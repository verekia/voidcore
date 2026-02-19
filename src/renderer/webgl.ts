// WebGL2 Renderer – Renders the scene using the WebGL2 API (fallback for non-WebGPU browsers).
//
// Functionally equivalent to the WebGPU renderer but uses the older WebGL2 API. The render
// pipeline is: frustum culling → sort → batch uniform upload → shadow pass →
// MSAA MRT draw (opaque only) → MSAA resolve → WBOIT transparent pass → OIT composite →
// bloom post-processing → final blit with gamma correction.
//
// Shadow mapping uses 3-cascade CSM (Cascaded Shadow Maps) with PCF 3×3 filtering, matching
// the WebGPU renderer. Shadow depth is rendered into a TEXTURE_2D_ARRAY with comparison mode.
//
// Key differences from WebGPU:
//   - Uses GLSL shaders instead of WGSL
//   - Uses Uniform Buffer Objects (UBOs) with bindBufferRange for per-object data
//   - Uses Vertex Array Objects (VAOs) to group vertex buffer bindings
//   - Uses framebuffer objects (FBOs) and renderbuffers for MSAA and MRT
//   - MSAA resolve is done via blitFramebuffer instead of resolve targets
//   - Clip space depth is [-1, 1] instead of WebGPU's [0, 1]
//   - Shadow uses mat4Ortho ([-1,1] depth) instead of mat4OrthoZO ([0,1] depth)
//
// Vertex packing: normals → snorm8, UVs → float16, material indices → uint8,
// joints → uint8, bone weights → unorm8. This reduces vertex buffer size by ~50%
// compared to using float32 for everything.
//
// WebGLRenderer.render()  – Draws one frame.
// WebGLRenderer.dispose() – Releases all GPU resources.

import {
  frustumFromViewProjection,
  mat4Create,
  mat4Invert,
  mat4Multiply,
  mat4Ortho,
  mat4Transpose,
  vec3Create,
} from '../math/index.ts'
import { Mesh } from '../scene/mesh.ts'
import { Node } from '../scene/node.ts'
import { packNormalsSnorm8, packUVsFloat16, packWeightsUnorm8 } from './pack.ts'
import {
  collectMeshes,
  computeLightDir,
  computeCascadeSplits,
  computeCascadeMatrix,
  defaultMaxDpr,
  findDirectionalLight,
  findTransparentStart,
  NUM_CASCADES,
} from './shared.ts'
import { createSortState, sortMeshes } from './sort.ts'
import {
  LAMBERT_VERT,
  LAMBERT_FRAG,
  LAMBERT_SKINNED_VERT,
  LAMBERT_TRANSPARENT_FRAG,
  BASIC_VERT,
  BASIC_FRAG,
  BASIC_SKINNED_VERT,
  BASIC_TRANSPARENT_FRAG,
  SHADOW_DEPTH_VERT,
  SHADOW_DEPTH_SKINNED_VERT,
  SHADOW_DEPTH_FRAG,
  FULLSCREEN_VERT,
  BLOOM_DOWNSAMPLE_FRAG,
  BLOOM_UPSAMPLE_FRAG,
  OIT_COMPOSITE_FRAG,
  BLIT_FRAG,
} from './webgl-shaders.ts'

import type { Geometry } from '../geometry/geometry.ts'
import type { Material, PaletteEntry } from '../materials/material.ts'
import type { AABB, Mat4, Vec3 } from '../math/index.ts'
import type { PerspectiveCamera } from '../scene/camera.ts'
import type { DirectionalLight } from '../scene/light.ts'
import type { Scene } from '../scene/scene.ts'
import type { Renderer, RendererConfig, FrameStats } from './renderer.ts'
import type { SortState } from './sort.ts'

// ─── WebGL2 GPU buffer handles ───────────────────────────────────────

interface GPUBuffers {
  position: WebGLBuffer
  normal: WebGLBuffer
  index: WebGLBuffer
  uv?: WebGLBuffer
  color?: WebGLBuffer
  materialIndex?: WebGLBuffer
  joints?: WebGLBuffer
  weights?: WebGLBuffer
  vao?: WebGLVertexArrayObject
}

// ─── Shader compilation ───────────────────────────────────────────────

const compileShader = (gl: WebGL2RenderingContext, type: number, source: string): WebGLShader => {
  const shader = gl.createShader(type)!
  gl.shaderSource(shader, source)
  gl.compileShader(shader)
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader)
    gl.deleteShader(shader)
    throw new Error(`Shader compile error: ${log}`)
  }
  return shader
}

const createProgram = (gl: WebGL2RenderingContext, vertSrc: string, fragSrc: string): WebGLProgram => {
  const vert = compileShader(gl, gl.VERTEX_SHADER, vertSrc)
  const frag = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc)
  const program = gl.createProgram()!
  gl.attachShader(program, vert)
  gl.attachShader(program, frag)
  gl.linkProgram(program)
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program)
    throw new Error(`Program link error: ${log}`)
  }
  gl.deleteShader(vert)
  gl.deleteShader(frag)
  return program
}

// ─── Uniform location cache ───────────────────────────────────────────
// Matrix and light uniforms are now in UBOs; only material uniforms remain.

interface SceneUniformLocs {
  u_baseColor: WebGLUniformLocation | null
  u_opacity: WebGLUniformLocation | null
  u_hasPalette: WebGLUniformLocation | null
  u_paletteColor: (WebGLUniformLocation | null)[]
  u_paletteEmissive: (WebGLUniformLocation | null)[]
  u_shadowMap: WebGLUniformLocation | null
  u_receiveShadow: WebGLUniformLocation | null
}

interface PostUniformLocs {
  u_srcTexture: WebGLUniformLocation | null
  u_texelSize: WebGLUniformLocation | null
  u_useKarisAverage: WebGLUniformLocation | null
}

interface OITCompositeUniformLocs {
  u_accumTexture: WebGLUniformLocation | null
  u_revealTexture: WebGLUniformLocation | null
  u_opaqueTexture: WebGLUniformLocation | null
}

interface BlitUniformLocs {
  u_sceneTexture: WebGLUniformLocation | null
  u_bloomTexture: WebGLUniformLocation | null
  u_bloomIntensity: WebGLUniformLocation | null
}

const cacheSceneLocs = (gl: WebGL2RenderingContext, program: WebGLProgram): SceneUniformLocs => {
  const paletteColor: (WebGLUniformLocation | null)[] = []
  const paletteEmissive: (WebGLUniformLocation | null)[] = []
  for (let i = 0; i < 32; i++) {
    paletteColor.push(gl.getUniformLocation(program, `u_palette[${i}].color`))
    paletteEmissive.push(gl.getUniformLocation(program, `u_palette[${i}].emissive`))
  }
  return {
    u_baseColor: gl.getUniformLocation(program, 'u_baseColor'),
    u_opacity: gl.getUniformLocation(program, 'u_opacity'),
    u_hasPalette: gl.getUniformLocation(program, 'u_hasPalette'),
    u_paletteColor: paletteColor,
    u_paletteEmissive: paletteEmissive,
    u_shadowMap: gl.getUniformLocation(program, 'u_shadowMap'),
    u_receiveShadow: gl.getUniformLocation(program, 'u_receiveShadow'),
  }
}

const cachePostLocs = (gl: WebGL2RenderingContext, program: WebGLProgram): PostUniformLocs => ({
  u_srcTexture: gl.getUniformLocation(program, 'u_srcTexture'),
  u_texelSize: gl.getUniformLocation(program, 'u_texelSize'),
  u_useKarisAverage: gl.getUniformLocation(program, 'u_useKarisAverage'),
})

const cacheOITCompositeLocs = (gl: WebGL2RenderingContext, program: WebGLProgram): OITCompositeUniformLocs => ({
  u_accumTexture: gl.getUniformLocation(program, 'u_accumTexture'),
  u_revealTexture: gl.getUniformLocation(program, 'u_revealTexture'),
  u_opaqueTexture: gl.getUniformLocation(program, 'u_opaqueTexture'),
})

const cacheBlitLocs = (gl: WebGL2RenderingContext, program: WebGLProgram): BlitUniformLocs => ({
  u_sceneTexture: gl.getUniformLocation(program, 'u_sceneTexture'),
  u_bloomTexture: gl.getUniformLocation(program, 'u_bloomTexture'),
  u_bloomIntensity: gl.getUniformLocation(program, 'u_bloomIntensity'),
})

// ─── GPU buffer management ────────────────────────────────────────────

const ensureGPUBuffers = (gl: WebGL2RenderingContext, geometry: Geometry) => {
  if (geometry._gpuBuffers) return

  // Unbind any active VAO to prevent corrupting its element array buffer
  // binding when we bind the new index buffer below
  gl.bindVertexArray(null)

  const posBuffer = gl.createBuffer()!
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer)
  gl.bufferData(gl.ARRAY_BUFFER, geometry.positions, gl.STATIC_DRAW)

  // Pack normals: float32x3 → snorm8x4 (12 → 4 bytes/vertex)
  const packedNormals = packNormalsSnorm8(geometry.normals, geometry.vertexCount)
  const normBuffer = gl.createBuffer()!
  gl.bindBuffer(gl.ARRAY_BUFFER, normBuffer)
  gl.bufferData(gl.ARRAY_BUFFER, packedNormals, gl.STATIC_DRAW)

  const idxBuffer = gl.createBuffer()!
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuffer)
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, geometry.indices, gl.STATIC_DRAW)

  // Pack UVs: float32x2 → float16x2 (8 → 4 bytes/vertex)
  let uvBuffer: WebGLBuffer | undefined
  if (geometry.uvs) {
    uvBuffer = gl.createBuffer()!
    gl.bindBuffer(gl.ARRAY_BUFFER, uvBuffer)
    gl.bufferData(gl.ARRAY_BUFFER, packUVsFloat16(geometry.uvs), gl.STATIC_DRAW)
  }

  // Material index — upload Uint8Array directly (no Float32 conversion needed)
  let matIdxBuffer: WebGLBuffer | undefined
  if (geometry.materialIndices) {
    matIdxBuffer = gl.createBuffer()!
    gl.bindBuffer(gl.ARRAY_BUFFER, matIdxBuffer)
    gl.bufferData(gl.ARRAY_BUFFER, geometry.materialIndices, gl.STATIC_DRAW)
  }

  // Joints buffer — upload Uint8Array directly (no Float32 conversion needed)
  let jointsBuffer: WebGLBuffer | undefined
  if (geometry.joints) {
    const jointsU8 = geometry.joints instanceof Uint8Array ? geometry.joints : new Uint8Array(geometry.joints)
    jointsBuffer = gl.createBuffer()!
    gl.bindBuffer(gl.ARRAY_BUFFER, jointsBuffer)
    gl.bufferData(gl.ARRAY_BUFFER, jointsU8, gl.STATIC_DRAW)
  }

  // Pack weights: float32x4 → unorm8x4 (16 → 4 bytes/vertex)
  let weightsBuffer: WebGLBuffer | undefined
  if (geometry.weights) {
    weightsBuffer = gl.createBuffer()!
    gl.bindBuffer(gl.ARRAY_BUFFER, weightsBuffer)
    gl.bufferData(gl.ARRAY_BUFFER, packWeightsUnorm8(geometry.weights), gl.STATIC_DRAW)
  }

  // Create VAO
  const vao = gl.createVertexArray()!
  gl.bindVertexArray(vao)

  // Position (location 0)
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer)
  gl.enableVertexAttribArray(0)
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0)

  // Normal (location 1) — snorm8x4: BYTE normalized, GPU auto-converts to [-1,1] float
  gl.bindBuffer(gl.ARRAY_BUFFER, normBuffer)
  gl.enableVertexAttribArray(1)
  gl.vertexAttribPointer(1, 4, gl.BYTE, true, 0, 0)

  // UV (location 2) — float16x2: HALF_FLOAT, GPU auto-converts to float
  if (uvBuffer) {
    gl.bindBuffer(gl.ARRAY_BUFFER, uvBuffer)
    gl.enableVertexAttribArray(2)
    gl.vertexAttribPointer(2, 2, gl.HALF_FLOAT, false, 0, 0)
  }

  // Material index (location 3) — uint8 not normalized: byte value N becomes float N.0
  if (matIdxBuffer) {
    gl.bindBuffer(gl.ARRAY_BUFFER, matIdxBuffer)
    gl.enableVertexAttribArray(3)
    gl.vertexAttribPointer(3, 1, gl.UNSIGNED_BYTE, false, 1, 0)
  } else {
    gl.disableVertexAttribArray(3)
    gl.vertexAttrib1f(3, 0.0)
  }

  // Joints (location 4) — uint8 not normalized: byte value N becomes float N.0
  if (jointsBuffer) {
    gl.bindBuffer(gl.ARRAY_BUFFER, jointsBuffer)
    gl.enableVertexAttribArray(4)
    gl.vertexAttribPointer(4, 4, gl.UNSIGNED_BYTE, false, 0, 0)
  } else {
    gl.disableVertexAttribArray(4)
  }

  // Weights (location 5) — unorm8x4: UNSIGNED_BYTE normalized, GPU maps [0,255]→[0.0,1.0]
  if (weightsBuffer) {
    gl.bindBuffer(gl.ARRAY_BUFFER, weightsBuffer)
    gl.enableVertexAttribArray(5)
    gl.vertexAttribPointer(5, 4, gl.UNSIGNED_BYTE, true, 0, 0)
  } else {
    gl.disableVertexAttribArray(5)
  }

  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuffer)
  gl.bindVertexArray(null)

  geometry._gpuBuffers = {
    position: posBuffer,
    normal: normBuffer,
    index: idxBuffer,
    uv: uvBuffer,
    materialIndex: matIdxBuffer,
    joints: jointsBuffer,
    weights: weightsBuffer,
    vao,
  }
}

// ─── Render targets ───────────────────────────────────────────────────

interface RenderTargets {
  // MSAA color + emissive + depth
  msaaFbo: WebGLFramebuffer
  msaaColorRb: WebGLRenderbuffer
  msaaEmissiveRb: WebGLRenderbuffer
  msaaDepthRb: WebGLRenderbuffer
  // Resolved (1x)
  resolvedColorFbo: WebGLFramebuffer
  resolvedColorTex: WebGLTexture
  resolvedEmissiveFbo: WebGLFramebuffer
  resolvedEmissiveTex: WebGLTexture
  // OIT (WBOIT transparency)
  oitFbo: WebGLFramebuffer
  oitAccumTex: WebGLTexture
  oitRevealTex: WebGLTexture
  resolvedDepthFbo: WebGLFramebuffer
  resolvedDepthRb: WebGLRenderbuffer
  opaqueColorFbo: WebGLFramebuffer
  opaqueColorTex: WebGLTexture
  // Bloom chain
  bloomFbos: WebGLFramebuffer[]
  bloomTextures: WebGLTexture[]
  bloomWidths: number[]
  bloomHeights: number[]
  width: number
  height: number
}

const createRenderTargets = (
  gl: WebGL2RenderingContext,
  w: number,
  h: number,
  samples: number,
  bloomLevels: number,
): RenderTargets => {
  // MSAA FBO with color + emissive + depth renderbuffers
  const msaaFbo = gl.createFramebuffer()!
  gl.bindFramebuffer(gl.FRAMEBUFFER, msaaFbo)

  const msaaColorRb = gl.createRenderbuffer()!
  gl.bindRenderbuffer(gl.RENDERBUFFER, msaaColorRb)
  gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, gl.RGBA8, w, h)
  gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, msaaColorRb)

  const msaaEmissiveRb = gl.createRenderbuffer()!
  gl.bindRenderbuffer(gl.RENDERBUFFER, msaaEmissiveRb)
  gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, gl.RGBA16F, w, h)
  gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.RENDERBUFFER, msaaEmissiveRb)

  const msaaDepthRb = gl.createRenderbuffer()!
  gl.bindRenderbuffer(gl.RENDERBUFFER, msaaDepthRb)
  gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, gl.DEPTH_COMPONENT24, w, h)
  gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, msaaDepthRb)

  gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1])

  // Resolved color texture
  const resolvedColorTex = gl.createTexture()!
  gl.bindTexture(gl.TEXTURE_2D, resolvedColorTex)
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)

  const resolvedColorFbo = gl.createFramebuffer()!
  gl.bindFramebuffer(gl.FRAMEBUFFER, resolvedColorFbo)
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, resolvedColorTex, 0)

  // Resolved emissive texture
  const resolvedEmissiveTex = gl.createTexture()!
  gl.bindTexture(gl.TEXTURE_2D, resolvedEmissiveTex)
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)

  const resolvedEmissiveFbo = gl.createFramebuffer()!
  gl.bindFramebuffer(gl.FRAMEBUFFER, resolvedEmissiveFbo)
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, resolvedEmissiveTex, 0)

  // Copy of resolved opaque color (read by OIT composite to avoid feedback loop)
  const opaqueColorTex = gl.createTexture()!
  gl.bindTexture(gl.TEXTURE_2D, opaqueColorTex)
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)

  const opaqueColorFbo = gl.createFramebuffer()!
  gl.bindFramebuffer(gl.FRAMEBUFFER, opaqueColorFbo)
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, opaqueColorTex, 0)

  // OIT accumulation texture (RGBA16F, non-MSAA)
  const oitAccumTex = gl.createTexture()!
  gl.bindTexture(gl.TEXTURE_2D, oitAccumTex)
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)

  // OIT revealage texture (RGBA8, non-MSAA — we only use the R channel)
  const oitRevealTex = gl.createTexture()!
  gl.bindTexture(gl.TEXTURE_2D, oitRevealTex)
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)

  // OIT FBO (accum + reveal as MRT, plus resolved depth for depth testing)
  const resolvedDepthRb = gl.createRenderbuffer()!
  gl.bindRenderbuffer(gl.RENDERBUFFER, resolvedDepthRb)
  gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h)

  const resolvedDepthFbo = gl.createFramebuffer()!
  gl.bindFramebuffer(gl.FRAMEBUFFER, resolvedDepthFbo)
  gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, resolvedDepthRb)

  const oitFbo = gl.createFramebuffer()!
  gl.bindFramebuffer(gl.FRAMEBUFFER, oitFbo)
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, oitAccumTex, 0)
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, oitRevealTex, 0)
  gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, resolvedDepthRb)
  gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1])

  // Bloom chain (half-res progressive downsample/upsample)
  const bloomFbos: WebGLFramebuffer[] = []
  const bloomTextures: WebGLTexture[] = []
  const bloomWidths: number[] = []
  const bloomHeights: number[] = []

  let bw = Math.max(1, Math.floor(w / 2))
  let bh = Math.max(1, Math.floor(h / 2))

  for (let i = 0; i < bloomLevels; i++) {
    const tex = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, tex)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, bw, bh, 0, gl.RGBA, gl.HALF_FLOAT, null)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)

    const fbo = gl.createFramebuffer()!
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo)
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0)

    bloomFbos.push(fbo)
    bloomTextures.push(tex)
    bloomWidths.push(bw)
    bloomHeights.push(bh)

    bw = Math.max(1, Math.floor(bw / 2))
    bh = Math.max(1, Math.floor(bh / 2))
  }

  gl.bindFramebuffer(gl.FRAMEBUFFER, null)

  return {
    msaaFbo,
    msaaColorRb,
    msaaEmissiveRb,
    msaaDepthRb,
    resolvedColorFbo,
    resolvedColorTex,
    resolvedEmissiveFbo,
    resolvedEmissiveTex,
    oitFbo,
    oitAccumTex,
    oitRevealTex,
    resolvedDepthFbo,
    resolvedDepthRb,
    opaqueColorFbo,
    opaqueColorTex,
    bloomFbos,
    bloomTextures,
    bloomWidths,
    bloomHeights,
    width: w,
    height: h,
  }
}

// ─── Renderer ─────────────────────────────────────────────────────────

export { type RendererConfig, type FrameStats } from './renderer.ts'

export class WebGLRenderer implements Renderer {
  readonly backend = 'webgl2' as const
  gl: WebGL2RenderingContext
  canvas: HTMLCanvasElement

  get maxDpr(): number {
    return this._maxDpr
  }

  set maxDpr(value: number) {
    this._maxDpr = value
  }

  private lambertProgram: WebGLProgram
  private basicProgram: WebGLProgram
  private lambertSkinnedProgram: WebGLProgram
  private basicSkinnedProgram: WebGLProgram
  private shadowDepthProgram: WebGLProgram
  private shadowDepthSkinnedProgram: WebGLProgram
  private lambertTransparentProgram: WebGLProgram
  private basicTransparentProgram: WebGLProgram
  private lambertSkinnedTransparentProgram: WebGLProgram
  private basicSkinnedTransparentProgram: WebGLProgram
  private oitCompositeProgram: WebGLProgram
  private bloomDownsampleProgram: WebGLProgram
  private bloomUpsampleProgram: WebGLProgram
  private blitProgram: WebGLProgram

  // Cached uniform locations
  private _lambertLocs!: SceneUniformLocs
  private _basicLocs!: SceneUniformLocs
  private _lambertSkinnedLocs!: SceneUniformLocs
  private _basicSkinnedLocs!: SceneUniformLocs
  private _lambertTransparentLocs!: SceneUniformLocs
  private _basicTransparentLocs!: SceneUniformLocs
  private _lambertSkinnedTransparentLocs!: SceneUniformLocs
  private _basicSkinnedTransparentLocs!: SceneUniformLocs
  private _oitCompositeLocs!: OITCompositeUniformLocs
  private _bloomDownLocs!: PostUniformLocs
  private _bloomUpLocs!: PostUniformLocs
  private _blitLocs!: BlitUniformLocs

  // Per-draw-buffer blend extension (for WBOIT)
  private _drawBuffersIndexed: OES_draw_buffers_indexed | null = null

  // Material tracking for skipping redundant uniform uploads
  private _lastMaterial: Material | null = null
  private _lastProgram: WebGLProgram | null = null

  private renderTargets: RenderTargets | null = null
  private samples: number
  private bloomLevels: number
  private bloomIntensity: number
  private bloomEnabled: boolean

  // Shadow config
  private shadowEnabled: boolean
  private shadowResolution: number
  private shadowLambda: number
  private shadowBackExtend: number
  private shadowConstantBias: number
  private shadowSlopeBias: number
  private shadowBlendRange: number

  // Shadow GPU resources
  private _shadowTexture!: WebGLTexture
  private _shadowFbos: WebGLFramebuffer[] = []
  private _shadowUBO!: WebGLBuffer
  private _shadowUBData = new Float32Array(16) // mat4 = 64 bytes
  private _shadowVAOs = new WeakMap<Geometry, WebGLVertexArrayObject>()
  private _shadowSkinnedVAOs = new WeakMap<Geometry, WebGLVertexArrayObject>()

  // Shadow scratch
  private _shadowMeshes: Mesh[] = []
  private _cascadeVPs: Mat4[] = [mat4Create(), mat4Create(), mat4Create()]
  private _cascadeSplits = new Float32Array(NUM_CASCADES)
  private _shadowLightView: Mat4 = mat4Create()
  private _shadowLightProj: Mat4 = mat4Create()
  private _frameNum = 0

  // Traversal
  private _traversalStack: Node[] = []

  // UBOs
  // FrameBlock (binding 0): 352 bytes = 88 floats (VP + light + shadow data)
  private _frameUBO!: WebGLBuffer
  private _frameData = new Float32Array(88)
  // ObjectBlock (binding 1, dynamic): mat4 worldMatrix + mat4 normalMatrix = 128 bytes
  // SkinnedObjectBlock (binding 1, dynamic): above + mat4[32] boneMatrices = 2176 bytes
  private _uboAlignment = 256
  private _alignedObjectSize = 256 // ceil(128 / alignment) * alignment
  private _alignedSkinnedSize = 2304 // ceil(2176 / alignment) * alignment
  private _objectDynBuf!: WebGLBuffer
  private _skinnedDynBuf!: WebGLBuffer
  private _objectBatchData!: Float32Array
  private _skinnedBatchData!: Float32Array
  private _objectCapacity = 2048
  private _skinnedCapacity = 2048

  // Scratch
  private _vpMatrix: Mat4 = mat4Create()
  private _invWorldMatrix: Mat4 = mat4Create()
  private _normalMatrix: Mat4 = mat4Create()
  private _frustumPlanes = new Float32Array(24)
  private _shadowFrustumPlanes = new Float32Array(24)
  private _worldAABB: AABB = new Float32Array(6)
  private _lightDir = vec3Create()
  private _tempVec3 = new Float32Array(3)
  private _meshes: Mesh[] = []
  private _sortState: SortState = createSortState(4096)

  // DPR limiting
  private _maxDpr: number

  // Cached canvas dimensions
  private _displayW = 0
  private _displayH = 0
  private _resizeObserver: ResizeObserver | null = null

  // Stats
  private _lastFrameTime = -1
  private _frameCount = 0
  private _fpsAccumulator = 0
  private _currentFps = 60
  stats: FrameStats = {
    fps: 60,
    frameTime: 0,
    drawCalls: 0,
    shadowDrawCalls: 0,
    triangles: 0,
    visibleObjects: 0,
    culledObjects: 0,
  }

  constructor(canvas: HTMLCanvasElement, config: RendererConfig = {}) {
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false, // We handle MSAA ourselves
      powerPreference: 'high-performance',
    })
    if (!gl) throw new Error('WebGL2 not supported')

    // Check for required extensions
    gl.getExtension('EXT_color_buffer_float')
    gl.getExtension('EXT_float_blend')
    const dbi = gl.getExtension('OES_draw_buffers_indexed')
    this._drawBuffersIndexed = dbi

    this.gl = gl
    this.canvas = canvas
    this._maxDpr = config.maxDpr === false ? Infinity : (config.maxDpr ?? defaultMaxDpr())
    this.samples = config.antialias !== false ? 4 : 1

    const bloomConfig = config.bloom
    if (bloomConfig === false) {
      this.bloomEnabled = false
      this.bloomIntensity = 0
      this.bloomLevels = 0
    } else if (typeof bloomConfig === 'object') {
      this.bloomEnabled = true
      this.bloomIntensity = bloomConfig.intensity ?? 0.5
      this.bloomLevels = bloomConfig.levels ?? 5
    } else {
      this.bloomEnabled = true
      this.bloomIntensity = 0.5
      this.bloomLevels = 5
    }

    // Shadow config
    const shadowConfig = config.shadows
    if (!shadowConfig) {
      this.shadowEnabled = false
      this.shadowResolution = 1024
      this.shadowLambda = 0.7
      this.shadowBackExtend = 75
      this.shadowConstantBias = 0.001
      this.shadowSlopeBias = 0.005
      this.shadowBlendRange = 0.1
    } else if (typeof shadowConfig === 'object') {
      this.shadowEnabled = shadowConfig.enabled !== false
      this.shadowResolution = shadowConfig.resolution ?? 1024
      this.shadowLambda = shadowConfig.lambda ?? 0.7
      this.shadowBackExtend = shadowConfig.backExtend ?? 75
      this.shadowConstantBias = shadowConfig.constantBias ?? 0.001
      this.shadowSlopeBias = shadowConfig.slopeBias ?? 0.005
      this.shadowBlendRange = shadowConfig.blendRange ?? 0.1
    } else {
      this.shadowEnabled = true
      this.shadowResolution = 1024
      this.shadowLambda = 0.7
      this.shadowBackExtend = 75
      this.shadowConstantBias = 0.001
      this.shadowSlopeBias = 0.005
      this.shadowBlendRange = 0.1
    }

    // Compile programs
    this.lambertProgram = createProgram(gl, LAMBERT_VERT, LAMBERT_FRAG)
    this.basicProgram = createProgram(gl, BASIC_VERT, BASIC_FRAG)
    this.lambertSkinnedProgram = createProgram(gl, LAMBERT_SKINNED_VERT, LAMBERT_FRAG)
    this.basicSkinnedProgram = createProgram(gl, BASIC_SKINNED_VERT, BASIC_FRAG)
    this.lambertTransparentProgram = createProgram(gl, LAMBERT_VERT, LAMBERT_TRANSPARENT_FRAG)
    this.basicTransparentProgram = createProgram(gl, BASIC_VERT, BASIC_TRANSPARENT_FRAG)
    this.lambertSkinnedTransparentProgram = createProgram(gl, LAMBERT_SKINNED_VERT, LAMBERT_TRANSPARENT_FRAG)
    this.basicSkinnedTransparentProgram = createProgram(gl, BASIC_SKINNED_VERT, BASIC_TRANSPARENT_FRAG)
    this.oitCompositeProgram = createProgram(gl, FULLSCREEN_VERT, OIT_COMPOSITE_FRAG)
    this.shadowDepthProgram = createProgram(gl, SHADOW_DEPTH_VERT, SHADOW_DEPTH_FRAG)
    this.shadowDepthSkinnedProgram = createProgram(gl, SHADOW_DEPTH_SKINNED_VERT, SHADOW_DEPTH_FRAG)
    this.bloomDownsampleProgram = createProgram(gl, FULLSCREEN_VERT, BLOOM_DOWNSAMPLE_FRAG)
    this.bloomUpsampleProgram = createProgram(gl, FULLSCREEN_VERT, BLOOM_UPSAMPLE_FRAG)
    this.blitProgram = createProgram(gl, FULLSCREEN_VERT, BLIT_FRAG)

    // Cache uniform locations
    this._lambertLocs = cacheSceneLocs(gl, this.lambertProgram)
    this._basicLocs = cacheSceneLocs(gl, this.basicProgram)
    this._lambertSkinnedLocs = cacheSceneLocs(gl, this.lambertSkinnedProgram)
    this._basicSkinnedLocs = cacheSceneLocs(gl, this.basicSkinnedProgram)
    this._lambertTransparentLocs = cacheSceneLocs(gl, this.lambertTransparentProgram)
    this._basicTransparentLocs = cacheSceneLocs(gl, this.basicTransparentProgram)
    this._lambertSkinnedTransparentLocs = cacheSceneLocs(gl, this.lambertSkinnedTransparentProgram)
    this._basicSkinnedTransparentLocs = cacheSceneLocs(gl, this.basicSkinnedTransparentProgram)
    this._oitCompositeLocs = cacheOITCompositeLocs(gl, this.oitCompositeProgram)
    this._bloomDownLocs = cachePostLocs(gl, this.bloomDownsampleProgram)
    this._bloomUpLocs = cachePostLocs(gl, this.bloomUpsampleProgram)
    this._blitLocs = cacheBlitLocs(gl, this.blitProgram)

    // Enable depth test
    gl.enable(gl.DEPTH_TEST)
    gl.depthFunc(gl.LEQUAL)
    gl.enable(gl.CULL_FACE)
    gl.cullFace(gl.BACK)

    // UBO setup
    this._uboAlignment = gl.getParameter(gl.UNIFORM_BUFFER_OFFSET_ALIGNMENT) as number
    this._alignedObjectSize = Math.ceil(128 / this._uboAlignment) * this._uboAlignment
    this._alignedSkinnedSize = Math.ceil(2176 / this._uboAlignment) * this._uboAlignment

    // Frame UBO (352 bytes = 88 floats)
    this._frameUBO = gl.createBuffer()!
    gl.bindBuffer(gl.UNIFORM_BUFFER, this._frameUBO)
    gl.bufferData(gl.UNIFORM_BUFFER, 352, gl.DYNAMIC_DRAW)

    // Dynamic object / skinned UBOs
    this._createDynamicBuffers()

    // Bind UBO block indices for all scene programs (done once at init)
    this._bindUBOBlocks(this.lambertProgram, false)
    this._bindUBOBlocks(this.basicProgram, false)
    this._bindUBOBlocks(this.lambertSkinnedProgram, true)
    this._bindUBOBlocks(this.basicSkinnedProgram, true)
    this._bindUBOBlocks(this.lambertTransparentProgram, false)
    this._bindUBOBlocks(this.basicTransparentProgram, false)
    this._bindUBOBlocks(this.lambertSkinnedTransparentProgram, true)
    this._bindUBOBlocks(this.basicSkinnedTransparentProgram, true)

    // Shadow UBO (binding 2, 64 bytes = mat4)
    this._shadowUBO = gl.createBuffer()!
    gl.bindBuffer(gl.UNIFORM_BUFFER, this._shadowUBO)
    gl.bufferData(gl.UNIFORM_BUFFER, 64, gl.DYNAMIC_DRAW)

    // Bind ShadowBlock for shadow programs
    this._bindShadowUBOBlocks(this.shadowDepthProgram, false)
    this._bindShadowUBOBlocks(this.shadowDepthSkinnedProgram, true)

    // Shadow texture (TEXTURE_2D_ARRAY with depth comparison)
    const shadowRes = this.shadowEnabled ? this.shadowResolution : 1
    this._shadowTexture = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this._shadowTexture)
    gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, gl.DEPTH_COMPONENT24, shadowRes, shadowRes, 3)
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_COMPARE_MODE, gl.COMPARE_REF_TO_TEXTURE)
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_COMPARE_FUNC, gl.LEQUAL)

    // Shadow FBOs (one per cascade layer)
    for (let i = 0; i < NUM_CASCADES; i++) {
      const fbo = gl.createFramebuffer()!
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo)
      gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, this._shadowTexture, 0, i)
      gl.drawBuffers([])
      gl.readBuffer(gl.NONE)
      this._shadowFbos.push(fbo)
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)

    // Cache canvas dimensions
    this._displayW = canvas.clientWidth
    this._displayH = canvas.clientHeight
    this._resizeObserver = new ResizeObserver(() => {
      this._displayW = this.canvas.clientWidth
      this._displayH = this.canvas.clientHeight
    })
    this._resizeObserver.observe(canvas)
  }

  private _bindUBOBlocks(program: WebGLProgram, skinned: boolean) {
    const gl = this.gl
    const frameIdx = gl.getUniformBlockIndex(program, 'FrameBlock')
    if (frameIdx !== gl.INVALID_INDEX) gl.uniformBlockBinding(program, frameIdx, 0)
    const objName = skinned ? 'SkinnedObjectBlock' : 'ObjectBlock'
    const objIdx = gl.getUniformBlockIndex(program, objName)
    if (objIdx !== gl.INVALID_INDEX) gl.uniformBlockBinding(program, objIdx, 1)
  }

  private _bindShadowUBOBlocks(program: WebGLProgram, skinned: boolean) {
    const gl = this.gl
    const shadowIdx = gl.getUniformBlockIndex(program, 'ShadowBlock')
    if (shadowIdx !== gl.INVALID_INDEX) gl.uniformBlockBinding(program, shadowIdx, 2)
    const objName = skinned ? 'SkinnedObjectBlock' : 'ObjectBlock'
    const objIdx = gl.getUniformBlockIndex(program, objName)
    if (objIdx !== gl.INVALID_INDEX) gl.uniformBlockBinding(program, objIdx, 1)
  }

  private ensureShadowVAO(geometry: Geometry, skinned: boolean): WebGLVertexArrayObject {
    const cache = skinned ? this._shadowSkinnedVAOs : this._shadowVAOs
    const cached = cache.get(geometry)
    if (cached) return cached

    const gl = this.gl
    ensureGPUBuffers(gl, geometry)
    const bufs = geometry._gpuBuffers as GPUBuffers

    const vao = gl.createVertexArray()!
    gl.bindVertexArray(vao)

    // Position (location 0)
    gl.bindBuffer(gl.ARRAY_BUFFER, bufs.position)
    gl.enableVertexAttribArray(0)
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0)

    if (skinned) {
      // Joints (location 4) — uint8 not normalized
      if (bufs.joints) {
        gl.bindBuffer(gl.ARRAY_BUFFER, bufs.joints)
        gl.enableVertexAttribArray(4)
        gl.vertexAttribPointer(4, 4, gl.UNSIGNED_BYTE, false, 0, 0)
      }
      // Weights (location 5) — unorm8 normalized
      if (bufs.weights) {
        gl.bindBuffer(gl.ARRAY_BUFFER, bufs.weights)
        gl.enableVertexAttribArray(5)
        gl.vertexAttribPointer(5, 4, gl.UNSIGNED_BYTE, true, 0, 0)
      }
    }

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, bufs.index)
    gl.bindVertexArray(null)

    cache.set(geometry, vao)
    return vao
  }

  // ─── Cascade shadow map computation ─────────────────────────────

  private _computeCascadeSplits(camera: PerspectiveCamera): void {
    computeCascadeSplits(this._cascadeSplits, camera, this.shadowLambda)
  }

  private _computeCascadeMatrix(
    cascadeIdx: number,
    camera: PerspectiveCamera,
    lightDir: Vec3,
    nearDist: number,
    farDist: number,
  ): void {
    computeCascadeMatrix(
      this._cascadeVPs[cascadeIdx]!,
      camera,
      lightDir,
      nearDist,
      farDist,
      this.shadowBackExtend,
      this.shadowResolution,
      this._shadowLightView,
      this._shadowLightProj,
      mat4Ortho,
    )
  }

  private _createDynamicBuffers() {
    const gl = this.gl
    this._objectDynBuf = gl.createBuffer()!
    gl.bindBuffer(gl.UNIFORM_BUFFER, this._objectDynBuf)
    gl.bufferData(gl.UNIFORM_BUFFER, this._objectCapacity * this._alignedObjectSize, gl.DYNAMIC_DRAW)
    this._objectBatchData = new Float32Array((this._objectCapacity * this._alignedObjectSize) / 4)

    this._skinnedDynBuf = gl.createBuffer()!
    gl.bindBuffer(gl.UNIFORM_BUFFER, this._skinnedDynBuf)
    gl.bufferData(gl.UNIFORM_BUFFER, this._skinnedCapacity * this._alignedSkinnedSize, gl.DYNAMIC_DRAW)
    this._skinnedBatchData = new Float32Array((this._skinnedCapacity * this._alignedSkinnedSize) / 4)
  }

  private _ensureDynamicCapacity(objCount: number, skinnedCount: number) {
    const gl = this.gl
    if (objCount > this._objectCapacity) {
      this._objectCapacity = Math.max(objCount, this._objectCapacity * 2)
      gl.deleteBuffer(this._objectDynBuf)
      this._objectDynBuf = gl.createBuffer()!
      gl.bindBuffer(gl.UNIFORM_BUFFER, this._objectDynBuf)
      gl.bufferData(gl.UNIFORM_BUFFER, this._objectCapacity * this._alignedObjectSize, gl.DYNAMIC_DRAW)
      this._objectBatchData = new Float32Array((this._objectCapacity * this._alignedObjectSize) / 4)
    }
    if (skinnedCount > this._skinnedCapacity) {
      this._skinnedCapacity = Math.max(skinnedCount, this._skinnedCapacity * 2)
      gl.deleteBuffer(this._skinnedDynBuf)
      this._skinnedDynBuf = gl.createBuffer()!
      gl.bindBuffer(gl.UNIFORM_BUFFER, this._skinnedDynBuf)
      gl.bufferData(gl.UNIFORM_BUFFER, this._skinnedCapacity * this._alignedSkinnedSize, gl.DYNAMIC_DRAW)
      this._skinnedBatchData = new Float32Array((this._skinnedCapacity * this._alignedSkinnedSize) / 4)
    }
  }

  private ensureRenderTargets() {
    const w = this.canvas.width
    const h = this.canvas.height
    if (this.renderTargets && this.renderTargets.width === w && this.renderTargets.height === h) return
    // Destroy old targets
    if (this.renderTargets) this.destroyRenderTargets()
    this.renderTargets = createRenderTargets(this.gl, w, h, this.samples, this.bloomLevels)
  }

  private destroyRenderTargets() {
    if (!this.renderTargets) return
    const gl = this.gl
    const rt = this.renderTargets
    gl.deleteFramebuffer(rt.msaaFbo)
    gl.deleteRenderbuffer(rt.msaaColorRb)
    gl.deleteRenderbuffer(rt.msaaEmissiveRb)
    gl.deleteRenderbuffer(rt.msaaDepthRb)
    gl.deleteFramebuffer(rt.resolvedColorFbo)
    gl.deleteTexture(rt.resolvedColorTex)
    gl.deleteFramebuffer(rt.resolvedEmissiveFbo)
    gl.deleteTexture(rt.resolvedEmissiveTex)
    gl.deleteFramebuffer(rt.oitFbo)
    gl.deleteTexture(rt.oitAccumTex)
    gl.deleteTexture(rt.oitRevealTex)
    gl.deleteFramebuffer(rt.resolvedDepthFbo)
    gl.deleteRenderbuffer(rt.resolvedDepthRb)
    gl.deleteFramebuffer(rt.opaqueColorFbo)
    gl.deleteTexture(rt.opaqueColorTex)
    for (const fbo of rt.bloomFbos) gl.deleteFramebuffer(fbo)
    for (const tex of rt.bloomTextures) gl.deleteTexture(tex)
    this.renderTargets = null
  }

  render(scene: Scene, camera: PerspectiveCamera) {
    const gl = this.gl
    const now = performance.now()
    if (this._lastFrameTime < 0) this._lastFrameTime = now
    const dt = now - this._lastFrameTime
    this._lastFrameTime = now

    // FPS tracking
    this._frameCount++
    this._fpsAccumulator += dt
    if (this._fpsAccumulator >= 1000) {
      this._currentFps = this._frameCount
      this._frameCount = 0
      this._fpsAccumulator = 0
    }

    // Resize canvas if needed
    const dpr = Math.min(window.devicePixelRatio, this._maxDpr)
    const displayW = Math.floor(this._displayW * dpr)
    const displayH = Math.floor(this._displayH * dpr)
    if (this.canvas.width !== displayW || this.canvas.height !== displayH) {
      this.canvas.width = displayW
      this.canvas.height = displayH
    }

    camera.aspect = this.canvas.width / this.canvas.height
    camera.updateProjection('neg-one-to-one') // WebGL2

    // Update scene graph (dirty flags)
    scene.updateGraph()

    // View-projection matrix (view matrix is set externally, e.g. by orbit controls)
    mat4Multiply(this._vpMatrix, camera._projectionMatrix, camera._viewMatrix)

    // Camera frustum
    frustumFromViewProjection(this._frustumPlanes, this._vpMatrix)

    // Find directional light (quick early-exit traversal)
    const dirLight = findDirectionalLight(scene, this._traversalStack)

    // Compute light direction from world matrix
    const lightDir = this._lightDir
    computeLightDir(lightDir, this._tempVec3, dirLight)

    this._frameNum++
    const frameNum = this._frameNum
    const shadowActive = this.shadowEnabled && !!dirLight

    let drawCalls = 0
    let shadowDrawCalls = 0
    let triangles = 0

    // ─── Cascade computation ─────────────────────────────────────
    // Compute shadow cascades BEFORE traversal so we can collect shadow-only
    // casters in the same pass as camera-visible meshes.
    let shadowFrustum: Float32Array | null = null
    if (shadowActive) {
      this._computeCascadeSplits(camera)
      for (let c = 0; c < NUM_CASCADES; c++) {
        const near = c === 0 ? camera.near : this._cascadeSplits[c - 1]!
        const far = this._cascadeSplits[c]!
        this._computeCascadeMatrix(c, camera, lightDir, near, far)
      }
      frustumFromViewProjection(this._shadowFrustumPlanes, this._cascadeVPs[2]!)
      shadowFrustum = this._shadowFrustumPlanes
    }

    // ─── Single merged traversal: camera meshes + shadow-only casters ───
    const meshes = this._meshes
    const shadowMeshes = this._shadowMeshes
    const culledCount = collectMeshes(
      scene,
      this._frustumPlanes,
      shadowFrustum,
      this._worldAABB,
      meshes,
      shadowMeshes,
      this._traversalStack,
    )

    // Radix sort meshes by layer > pipeline > material > depth
    sortMeshes(this._sortState, meshes, meshes.length, camera)
    const sortedIndices = this._sortState.indices

    // ─── Batch fill: camera-visible meshes ───────────────────────
    const alignedObjFloats = this._alignedObjectSize / 4
    const alignedSkinnedFloats = this._alignedSkinnedSize / 4

    // Count camera-visible meshes and mark _batchFrame
    let objCount = 0
    let skinnedCount = 0
    for (let si = 0; si < meshes.length; si++) {
      const mesh = meshes[sortedIndices[si]!]!
      mesh._batchFrame = frameNum
      if (mesh._isSkinned) skinnedCount++
      else objCount++
    }

    // Count shadow-only casters
    for (let i = 0; i < shadowMeshes.length; i++) {
      const sm = shadowMeshes[i]!
      sm._isSkinned = !!sm.skeleton
      if (sm._isSkinned) skinnedCount++
      else objCount++
    }

    this._ensureDynamicCapacity(objCount, skinnedCount)

    const objBatch = this._objectBatchData
    const skinnedBatch = this._skinnedBatchData
    let objIdx = 0
    let skinnedIdx = 0

    // Fill camera-visible meshes
    for (let si = 0; si < meshes.length; si++) {
      const mesh = meshes[sortedIndices[si]!]!
      if (mesh._isSkinned) {
        mesh.skeleton!.update()
        const off = skinnedIdx * alignedSkinnedFloats
        skinnedBatch.set(mesh._worldMatrix, off)
        if (mat4Invert(this._invWorldMatrix, mesh._worldMatrix)) {
          mat4Transpose(this._normalMatrix, this._invWorldMatrix)
        }
        skinnedBatch.set(this._normalMatrix, off + 16)
        skinnedBatch.set(mesh.skeleton!.boneMatrices, off + 32)
        mesh._batchIndex = skinnedIdx++
      } else {
        const off = objIdx * alignedObjFloats
        objBatch.set(mesh._worldMatrix, off)
        if (mat4Invert(this._invWorldMatrix, mesh._worldMatrix)) {
          mat4Transpose(this._normalMatrix, this._invWorldMatrix)
        }
        objBatch.set(this._normalMatrix, off + 16)
        mesh._batchIndex = objIdx++
      }
    }

    // Fill shadow-only meshes
    for (let i = 0; i < shadowMeshes.length; i++) {
      const mesh = shadowMeshes[i]!
      mesh._batchFrame = frameNum
      if (mesh._isSkinned) {
        mesh.skeleton!.update()
        const off = skinnedIdx * alignedSkinnedFloats
        skinnedBatch.set(mesh._worldMatrix, off)
        if (mat4Invert(this._invWorldMatrix, mesh._worldMatrix)) {
          mat4Transpose(this._normalMatrix, this._invWorldMatrix)
        }
        skinnedBatch.set(this._normalMatrix, off + 16)
        skinnedBatch.set(mesh.skeleton!.boneMatrices, off + 32)
        mesh._batchIndex = skinnedIdx++
      } else {
        const off = objIdx * alignedObjFloats
        objBatch.set(mesh._worldMatrix, off)
        if (mat4Invert(this._invWorldMatrix, mesh._worldMatrix)) {
          mat4Transpose(this._normalMatrix, this._invWorldMatrix)
        }
        objBatch.set(this._normalMatrix, off + 16)
        mesh._batchIndex = objIdx++
      }
    }

    // Upload batch data (1-2 bufferSubData calls instead of N*uniformMatrix4fv)
    if (objIdx > 0) {
      gl.bindBuffer(gl.UNIFORM_BUFFER, this._objectDynBuf)
      gl.bufferSubData(gl.UNIFORM_BUFFER, 0, objBatch, 0, objIdx * alignedObjFloats)
    }
    if (skinnedIdx > 0) {
      gl.bindBuffer(gl.UNIFORM_BUFFER, this._skinnedDynBuf)
      gl.bufferSubData(gl.UNIFORM_BUFFER, 0, skinnedBatch, 0, skinnedIdx * alignedSkinnedFloats)
    }

    // ─── Upload frame UBO (88 floats / 352 bytes) ─────────────────
    const fd = this._frameData
    fd.fill(0)
    fd.set(this._vpMatrix, 0)
    fd[16] = lightDir[0]!
    fd[17] = lightDir[1]!
    fd[18] = lightDir[2]!
    fd[19] = 0 // _lightPad
    fd[20] = dirLight ? (dirLight as DirectionalLight).color[0] : 0
    fd[21] = dirLight ? (dirLight as DirectionalLight).color[1] : 0
    fd[22] = dirLight ? (dirLight as DirectionalLight).color[2] : 0
    fd[23] = dirLight ? (dirLight as DirectionalLight).intensity : 0
    fd[24] = scene.ambientLight.color[0]
    fd[25] = scene.ambientLight.color[1]
    fd[26] = scene.ambientLight.color[2]
    fd[27] = scene.ambientLight.intensity
    fd[28] = shadowActive ? 1.0 : 0.0
    // fd[29-31] = 0 (padding, already zeroed)
    if (shadowActive) {
      fd.set(this._cascadeVPs[0]!, 32)
      fd.set(this._cascadeVPs[1]!, 48)
      fd.set(this._cascadeVPs[2]!, 64)
      fd[80] = this._cascadeSplits[0]!
      fd[81] = this._cascadeSplits[1]!
      fd[82] = this._cascadeSplits[2]!
      // fd[83] = 0 (padding)
      fd[84] = this.shadowConstantBias
      fd[85] = this.shadowSlopeBias
      fd[86] = 1.0 / this.shadowResolution
      fd[87] = this.shadowBlendRange
    }
    gl.bindBuffer(gl.UNIFORM_BUFFER, this._frameUBO)
    gl.bufferSubData(gl.UNIFORM_BUFFER, 0, fd)
    gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, this._frameUBO)

    // ─── Shadow render pass (3 cascades, depth-only) ─────────────
    if (shadowActive) {
      gl.cullFace(gl.FRONT)
      gl.colorMask(false, false, false, false)

      const shadowRes = this.shadowResolution
      const PAD = 0.3

      for (let c = 0; c < NUM_CASCADES; c++) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, this._shadowFbos[c]!)
        gl.viewport(0, 0, shadowRes, shadowRes)
        gl.clear(gl.DEPTH_BUFFER_BIT)

        // Write shadow UBO for this cascade
        this._shadowUBData.set(this._cascadeVPs[c]!)
        gl.bindBuffer(gl.UNIFORM_BUFFER, this._shadowUBO)
        gl.bufferSubData(gl.UNIFORM_BUFFER, 0, this._shadowUBData)
        gl.bindBufferBase(gl.UNIFORM_BUFFER, 2, this._shadowUBO)

        const cvp = this._cascadeVPs[c]!

        // Draw camera-visible shadow casters
        for (let si = 0; si < meshes.length; si++) {
          const mesh = meshes[sortedIndices[si]!]!
          if (!mesh.castShadow) continue

          // Light-space frustum cull
          const wm = mesh._worldMatrix
          const wx = wm[12]!,
            wy = wm[13]!,
            wz = wm[14]!
          const lx = cvp[0]! * wx + cvp[4]! * wy + cvp[8]! * wz + cvp[12]!
          const ly = cvp[1]! * wx + cvp[5]! * wy + cvp[9]! * wz + cvp[13]!
          const lz = cvp[2]! * wx + cvp[6]! * wy + cvp[10]! * wz + cvp[14]!
          if (lx < -(1 + PAD) || lx > 1 + PAD || ly < -(1 + PAD) || ly > 1 + PAD || lz < -(1 + PAD) || lz > 1 + PAD)
            continue

          ensureGPUBuffers(gl, mesh.geometry)
          const idxType = mesh.geometry.indices instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT

          if (mesh._isSkinned) {
            gl.useProgram(this.shadowDepthSkinnedProgram)
            gl.bindVertexArray(this.ensureShadowVAO(mesh.geometry, true))
            gl.bindBufferRange(
              gl.UNIFORM_BUFFER,
              1,
              this._skinnedDynBuf,
              mesh._batchIndex * this._alignedSkinnedSize,
              2176,
            )
          } else {
            gl.useProgram(this.shadowDepthProgram)
            gl.bindVertexArray(this.ensureShadowVAO(mesh.geometry, false))
            gl.bindBufferRange(
              gl.UNIFORM_BUFFER,
              1,
              this._objectDynBuf,
              mesh._batchIndex * this._alignedObjectSize,
              128,
            )
          }

          gl.drawElements(gl.TRIANGLES, mesh.geometry.indexCount, idxType, 0)
          shadowDrawCalls++
        }

        // Draw shadow-only casters
        for (let i = 0; i < shadowMeshes.length; i++) {
          const mesh = shadowMeshes[i]!

          const wm = mesh._worldMatrix
          const wx = wm[12]!,
            wy = wm[13]!,
            wz = wm[14]!
          const lx = cvp[0]! * wx + cvp[4]! * wy + cvp[8]! * wz + cvp[12]!
          const ly = cvp[1]! * wx + cvp[5]! * wy + cvp[9]! * wz + cvp[13]!
          const lz = cvp[2]! * wx + cvp[6]! * wy + cvp[10]! * wz + cvp[14]!
          if (lx < -(1 + PAD) || lx > 1 + PAD || ly < -(1 + PAD) || ly > 1 + PAD || lz < -(1 + PAD) || lz > 1 + PAD)
            continue

          ensureGPUBuffers(gl, mesh.geometry)
          const idxType = mesh.geometry.indices instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT

          if (mesh._isSkinned) {
            gl.useProgram(this.shadowDepthSkinnedProgram)
            gl.bindVertexArray(this.ensureShadowVAO(mesh.geometry, true))
            gl.bindBufferRange(
              gl.UNIFORM_BUFFER,
              1,
              this._skinnedDynBuf,
              mesh._batchIndex * this._alignedSkinnedSize,
              2176,
            )
          } else {
            gl.useProgram(this.shadowDepthProgram)
            gl.bindVertexArray(this.ensureShadowVAO(mesh.geometry, false))
            gl.bindBufferRange(
              gl.UNIFORM_BUFFER,
              1,
              this._objectDynBuf,
              mesh._batchIndex * this._alignedObjectSize,
              128,
            )
          }

          gl.drawElements(gl.TRIANGLES, mesh.geometry.indexCount, idxType, 0)
          shadowDrawCalls++
        }
      }

      gl.bindVertexArray(null)
      gl.cullFace(gl.BACK)
      gl.colorMask(true, true, true, true)
    }

    this.ensureRenderTargets()
    const rt = this.renderTargets!

    // ─── Opaque pass (MSAA MRT) ────────────────────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, rt.msaaFbo)
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1])
    gl.viewport(0, 0, rt.width, rt.height)
    gl.clearColor(0, 0, 0, 1.0)
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
    gl.enable(gl.DEPTH_TEST)
    gl.depthMask(true)

    // Bind shadow texture to unit 2
    gl.activeTexture(gl.TEXTURE2)
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this._shadowTexture)

    // Find where transparent meshes begin in the sorted array
    const transparentStart = findTransparentStart(this._sortState, meshes.length)

    // ─── Opaque draw loop ───────────────────────────────────────────
    this._lastMaterial = null
    this._lastProgram = null

    for (let si = 0; si < transparentStart; si++) {
      const mesh = meshes[sortedIndices[si]!]!
      let program: WebGLProgram
      let locs: SceneUniformLocs
      if (mesh._isSkinned) {
        if (mesh.material.type === 'lambert') {
          program = this.lambertSkinnedProgram
          locs = this._lambertSkinnedLocs
        } else {
          program = this.basicSkinnedProgram
          locs = this._basicSkinnedLocs
        }
      } else {
        if (mesh.material.type === 'lambert') {
          program = this.lambertProgram
          locs = this._lambertLocs
        } else {
          program = this.basicProgram
          locs = this._basicLocs
        }
      }

      const programChanged = program !== this._lastProgram
      if (programChanged) {
        gl.useProgram(program)
        this._lastProgram = program
        if (mesh.material.type === 'lambert') {
          gl.uniform1i(locs.u_shadowMap, 2)
        }
      }

      ensureGPUBuffers(gl, mesh.geometry)
      gl.bindVertexArray((mesh.geometry._gpuBuffers as GPUBuffers).vao!)

      if (mesh._isSkinned) {
        gl.bindBufferRange(gl.UNIFORM_BUFFER, 1, this._skinnedDynBuf, mesh._batchIndex * this._alignedSkinnedSize, 2176)
      } else {
        gl.bindBufferRange(gl.UNIFORM_BUFFER, 1, this._objectDynBuf, mesh._batchIndex * this._alignedObjectSize, 128)
      }

      const materialChanged = mesh.material !== this._lastMaterial || programChanged
      if (materialChanged) {
        this._lastMaterial = mesh.material
        gl.uniform3fv(locs.u_baseColor, mesh.material.color)
        gl.uniform1f(locs.u_opacity, mesh.material.opacity)
        if (mesh.material.type === 'lambert') {
          const hasPalette = !!mesh.material.palette && mesh.geometry.hasAttribute('materialIndex')
          gl.uniform1i(locs.u_hasPalette, hasPalette ? 1 : 0)
          gl.uniform1i(locs.u_receiveShadow, mesh.material.receiveShadow ? 1 : 0)
          if (hasPalette && mesh.material.palette) {
            for (let i = 0; i < 32; i++) {
              const entry: PaletteEntry = mesh.material.palette[i] ?? { color: [1, 1, 1] }
              gl.uniform4f(
                locs.u_paletteColor[i]!,
                entry.color[0],
                entry.color[1],
                entry.color[2],
                entry.opacity ?? 1.0,
              )
              gl.uniform4f(
                locs.u_paletteEmissive[i]!,
                entry.emissive?.[0] ?? 0,
                entry.emissive?.[1] ?? 0,
                entry.emissive?.[2] ?? 0,
                entry.emissiveIntensity ?? 0,
              )
            }
          }
        }
      }

      const idxType = mesh.geometry.indices instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT
      gl.drawElements(gl.TRIANGLES, mesh.geometry.indexCount, idxType, 0)
      drawCalls++
      triangles += mesh.geometry.indexCount / 3
    }

    gl.bindVertexArray(null)

    // ─── MSAA Resolve ──────────────────────────────────────────────
    // Resolve color
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, rt.msaaFbo)
    gl.readBuffer(gl.COLOR_ATTACHMENT0)
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, rt.resolvedColorFbo)
    gl.blitFramebuffer(0, 0, rt.width, rt.height, 0, 0, rt.width, rt.height, gl.COLOR_BUFFER_BIT, gl.NEAREST)

    // Resolve emissive
    gl.readBuffer(gl.COLOR_ATTACHMENT1)
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, rt.resolvedEmissiveFbo)
    gl.blitFramebuffer(0, 0, rt.width, rt.height, 0, 0, rt.width, rt.height, gl.COLOR_BUFFER_BIT, gl.NEAREST)

    // Resolve depth (MSAA → 1x for transparent pass depth testing)
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, rt.resolvedDepthFbo)
    gl.blitFramebuffer(0, 0, rt.width, rt.height, 0, 0, rt.width, rt.height, gl.DEPTH_BUFFER_BIT, gl.NEAREST)

    // ─── Transparent pass (WBOIT) ─────────────────────────────────
    if (transparentStart < meshes.length) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, rt.oitFbo)
      gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1])
      gl.viewport(0, 0, rt.width, rt.height)

      // Clear accum to (0,0,0,0), reveal to (1,0,0,1)
      gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0])
      gl.clearBufferfv(gl.COLOR, 1, [1, 0, 0, 1])

      // Depth test on (reads resolved depth), depth write off
      gl.enable(gl.DEPTH_TEST)
      gl.depthMask(false)
      gl.depthFunc(gl.LEQUAL)

      // WBOIT blending: accum = ONE/ONE additive, reveal = ZERO/ONE_MINUS_SRC_ALPHA
      gl.enable(gl.BLEND)
      const dbi = this._drawBuffersIndexed!
      dbi.blendFunciOES(0, gl.ONE, gl.ONE)
      dbi.blendFunciOES(1, gl.ZERO, gl.ONE_MINUS_SRC_ALPHA)

      // Disable back-face culling for transparent objects
      gl.disable(gl.CULL_FACE)

      // Bind shadow texture to unit 2
      gl.activeTexture(gl.TEXTURE2)
      gl.bindTexture(gl.TEXTURE_2D_ARRAY, this._shadowTexture)

      this._lastMaterial = null
      this._lastProgram = null

      for (let si = transparentStart; si < meshes.length; si++) {
        const mesh = meshes[sortedIndices[si]!]!
        let program: WebGLProgram
        let locs: SceneUniformLocs
        if (mesh._isSkinned) {
          if (mesh.material.type === 'lambert') {
            program = this.lambertSkinnedTransparentProgram
            locs = this._lambertSkinnedTransparentLocs
          } else {
            program = this.basicSkinnedTransparentProgram
            locs = this._basicSkinnedTransparentLocs
          }
        } else {
          if (mesh.material.type === 'lambert') {
            program = this.lambertTransparentProgram
            locs = this._lambertTransparentLocs
          } else {
            program = this.basicTransparentProgram
            locs = this._basicTransparentLocs
          }
        }

        const programChanged = program !== this._lastProgram
        if (programChanged) {
          gl.useProgram(program)
          this._lastProgram = program
          if (mesh.material.type === 'lambert') {
            gl.uniform1i(locs.u_shadowMap, 2)
          }
        }

        ensureGPUBuffers(gl, mesh.geometry)
        gl.bindVertexArray((mesh.geometry._gpuBuffers as GPUBuffers).vao!)

        if (mesh._isSkinned) {
          gl.bindBufferRange(
            gl.UNIFORM_BUFFER,
            1,
            this._skinnedDynBuf,
            mesh._batchIndex * this._alignedSkinnedSize,
            2176,
          )
        } else {
          gl.bindBufferRange(gl.UNIFORM_BUFFER, 1, this._objectDynBuf, mesh._batchIndex * this._alignedObjectSize, 128)
        }

        const materialChanged = mesh.material !== this._lastMaterial || programChanged
        if (materialChanged) {
          this._lastMaterial = mesh.material
          gl.uniform3fv(locs.u_baseColor, mesh.material.color)
          gl.uniform1f(locs.u_opacity, mesh.material.opacity)
          if (mesh.material.type === 'lambert') {
            const hasPalette = !!mesh.material.palette && mesh.geometry.hasAttribute('materialIndex')
            gl.uniform1i(locs.u_hasPalette, hasPalette ? 1 : 0)
            gl.uniform1i(locs.u_receiveShadow, mesh.material.receiveShadow ? 1 : 0)
            if (hasPalette && mesh.material.palette) {
              for (let i = 0; i < 32; i++) {
                const entry: PaletteEntry = mesh.material.palette[i] ?? { color: [1, 1, 1] }
                gl.uniform4f(
                  locs.u_paletteColor[i]!,
                  entry.color[0],
                  entry.color[1],
                  entry.color[2],
                  entry.opacity ?? 1.0,
                )
                gl.uniform4f(
                  locs.u_paletteEmissive[i]!,
                  entry.emissive?.[0] ?? 0,
                  entry.emissive?.[1] ?? 0,
                  entry.emissive?.[2] ?? 0,
                  entry.emissiveIntensity ?? 0,
                )
              }
            }
          }
        }

        const idxType = mesh.geometry.indices instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT
        gl.drawElements(gl.TRIANGLES, mesh.geometry.indexCount, idxType, 0)
        drawCalls++
        triangles += mesh.geometry.indexCount / 3
      }

      gl.bindVertexArray(null)
      gl.disable(gl.BLEND)
      gl.enable(gl.CULL_FACE)
      gl.depthMask(true)

      // ─── OIT Composite ───────────────────────────────────────────
      // Copy opaque color to avoid feedback loop (reads opaqueColorTex, writes resolvedColorFbo)
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, rt.resolvedColorFbo)
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, rt.opaqueColorFbo)
      gl.blitFramebuffer(0, 0, rt.width, rt.height, 0, 0, rt.width, rt.height, gl.COLOR_BUFFER_BIT, gl.NEAREST)

      // Composite: blend transparent result over opaque color
      gl.bindFramebuffer(gl.FRAMEBUFFER, rt.resolvedColorFbo)
      gl.viewport(0, 0, rt.width, rt.height)
      gl.disable(gl.DEPTH_TEST)
      gl.depthMask(false)

      gl.useProgram(this.oitCompositeProgram)
      const oitLocs = this._oitCompositeLocs

      gl.activeTexture(gl.TEXTURE0)
      gl.bindTexture(gl.TEXTURE_2D, rt.oitAccumTex)
      gl.uniform1i(oitLocs.u_accumTexture, 0)

      gl.activeTexture(gl.TEXTURE1)
      gl.bindTexture(gl.TEXTURE_2D, rt.oitRevealTex)
      gl.uniform1i(oitLocs.u_revealTexture, 1)

      gl.activeTexture(gl.TEXTURE2)
      gl.bindTexture(gl.TEXTURE_2D, rt.opaqueColorTex)
      gl.uniform1i(oitLocs.u_opaqueTexture, 2)

      gl.drawArrays(gl.TRIANGLES, 0, 3)
    }

    // ─── Bloom ─────────────────────────────────────────────────────
    if (this.bloomEnabled && this.bloomLevels > 0) {
      gl.disable(gl.DEPTH_TEST)
      gl.depthMask(false)

      // Downsample chain
      gl.useProgram(this.bloomDownsampleProgram)
      let srcTex = rt.resolvedEmissiveTex
      let srcW = rt.width
      let srcH = rt.height

      const downLocs = this._bloomDownLocs
      for (let i = 0; i < this.bloomLevels; i++) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, rt.bloomFbos[i]!)
        gl.viewport(0, 0, rt.bloomWidths[i]!, rt.bloomHeights[i]!)

        gl.activeTexture(gl.TEXTURE0)
        gl.bindTexture(gl.TEXTURE_2D, srcTex)
        gl.uniform1i(downLocs.u_srcTexture, 0)
        gl.uniform2f(downLocs.u_texelSize, 1 / srcW, 1 / srcH)
        gl.uniform1i(downLocs.u_useKarisAverage, i === 0 ? 1 : 0)

        gl.drawArrays(gl.TRIANGLES, 0, 3)

        srcTex = rt.bloomTextures[i]!
        srcW = rt.bloomWidths[i]!
        srcH = rt.bloomHeights[i]!
      }

      // Upsample chain (additive)
      gl.useProgram(this.bloomUpsampleProgram)
      gl.enable(gl.BLEND)
      gl.blendFunc(gl.ONE, gl.ONE)

      const upLocs = this._bloomUpLocs
      for (let i = this.bloomLevels - 1; i > 0; i--) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, rt.bloomFbos[i - 1]!)
        gl.viewport(0, 0, rt.bloomWidths[i - 1]!, rt.bloomHeights[i - 1]!)

        gl.activeTexture(gl.TEXTURE0)
        gl.bindTexture(gl.TEXTURE_2D, rt.bloomTextures[i]!)
        gl.uniform1i(upLocs.u_srcTexture, 0)
        gl.uniform2f(upLocs.u_texelSize, 1 / rt.bloomWidths[i]!, 1 / rt.bloomHeights[i]!)

        gl.drawArrays(gl.TRIANGLES, 0, 3)
      }

      gl.disable(gl.BLEND)
    }

    // ─── Final blit ────────────────────────────────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    gl.viewport(0, 0, rt.width, rt.height)
    gl.disable(gl.DEPTH_TEST)
    gl.depthMask(false)

    gl.useProgram(this.blitProgram)
    const blitLocs = this._blitLocs

    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, rt.resolvedColorTex)
    gl.uniform1i(blitLocs.u_sceneTexture, 0)

    gl.activeTexture(gl.TEXTURE1)
    if (this.bloomEnabled && this.bloomLevels > 0) {
      gl.bindTexture(gl.TEXTURE_2D, rt.bloomTextures[0]!)
    } else {
      gl.bindTexture(gl.TEXTURE_2D, rt.resolvedEmissiveTex)
    }
    gl.uniform1i(blitLocs.u_bloomTexture, 1)
    gl.uniform1f(blitLocs.u_bloomIntensity, this.bloomIntensity)

    gl.drawArrays(gl.TRIANGLES, 0, 3)

    // Re-enable depth for next frame
    gl.enable(gl.DEPTH_TEST)
    gl.depthMask(true)

    // Update stats
    this.stats.fps = this._currentFps
    this.stats.frameTime = dt
    this.stats.drawCalls = drawCalls
    this.stats.shadowDrawCalls = shadowDrawCalls
    this.stats.triangles = triangles
    this.stats.visibleObjects = meshes.length
    this.stats.culledObjects = culledCount
  }

  dispose() {
    this.destroyRenderTargets()
    const gl = this.gl
    gl.deleteBuffer(this._frameUBO)
    gl.deleteBuffer(this._objectDynBuf)
    gl.deleteBuffer(this._skinnedDynBuf)
    gl.deleteBuffer(this._shadowUBO)
    gl.deleteTexture(this._shadowTexture)
    for (const fbo of this._shadowFbos) gl.deleteFramebuffer(fbo)
    gl.deleteProgram(this.lambertProgram)
    gl.deleteProgram(this.basicProgram)
    gl.deleteProgram(this.lambertSkinnedProgram)
    gl.deleteProgram(this.basicSkinnedProgram)
    gl.deleteProgram(this.lambertTransparentProgram)
    gl.deleteProgram(this.basicTransparentProgram)
    gl.deleteProgram(this.lambertSkinnedTransparentProgram)
    gl.deleteProgram(this.basicSkinnedTransparentProgram)
    gl.deleteProgram(this.oitCompositeProgram)
    gl.deleteProgram(this.shadowDepthProgram)
    gl.deleteProgram(this.shadowDepthSkinnedProgram)
    gl.deleteProgram(this.bloomDownsampleProgram)
    gl.deleteProgram(this.bloomUpsampleProgram)
    gl.deleteProgram(this.blitProgram)
    this._resizeObserver?.disconnect()
    this._resizeObserver = null
  }
}
