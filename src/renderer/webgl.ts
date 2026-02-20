// WebGL2 Renderer – Renders the scene using the WebGL2 API (fallback for non-WebGPU browsers).
//
// Functionally equivalent to the WebGPU renderer but uses the older WebGL2 API. The render
// pipeline is: frustum culling → sort → batch uniform upload → shadow pass →
// MSAA MRT draw → occlusion queries → MSAA resolve → bloom post-processing → final blit.
//
// Shadow mapping uses a single shadow map with PCF 3×3 filtering, matching the WebGPU renderer.
// Shadow depth is rendered into a TEXTURE_2D with comparison mode.
//
// Dynamic geometry is supported via geometry.needsUpdate — position and normal buffers are
// re-uploaded with DYNAMIC_DRAW when the flag is set (used by helpers and procedural meshes).
//
// A GL state cache (_set* helpers) eliminates redundant state calls between consecutive draws.
// The radix sort groups meshes by pipeline > material > depth, so the cache yields significant
// savings — especially in the shadow pass (N draws with only 2 programs).
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
  aabbTransform,
  frustumFromViewProjection,
  mat4Create,
  mat4Invert,
  mat4Multiply,
  mat4Ortho,
  mat4Transpose,
  vec3Create,
} from '../math/index'
import { Mesh } from '../scene/mesh'
import { Node } from '../scene/node'
import { packNormalsSnorm8, packUVsFloat16, packWeightsUnorm8 } from './pack'
import {
  collectMeshes,
  computeLightDir,
  computeShadowMatrix,
  defaultMaxDpr,
  findAmbientLight,
  findDirectionalLight,
  findTransparentStart,
} from './shared'
import { createSortState, sortMeshes } from './sort'
import {
  LAMBERT_VERT,
  LAMBERT_FRAG,
  LAMBERT_TEXTURED_FRAG,
  LAMBERT_SKINNED_VERT,
  BASIC_VERT,
  BASIC_FRAG,
  BASIC_SKINNED_VERT,
  SHADOW_DEPTH_VERT,
  SHADOW_DEPTH_SKINNED_VERT,
  SHADOW_DEPTH_FRAG,
  OCCLUSION_BOX_VERT,
  OCCLUSION_BOX_FRAG,
  FULLSCREEN_VERT,
  BLOOM_DOWNSAMPLE_FRAG,
  BLOOM_UPSAMPLE_FRAG,
  BLIT_FRAG,
} from './webgl-shaders'

import type { Geometry } from '../geometry/geometry'
import type { Material, PaletteEntry } from '../materials/material'
import type { Texture } from '../materials/texture'
import type { AABB, Mat4, Vec3 } from '../math/index'
import type { PerspectiveCamera } from '../scene/camera'
import type { DirectionalLight } from '../scene/light'
import type { Scene } from '../scene/scene'
import type { Renderer, RendererConfig, FrameStats } from './renderer'
import type { SortState } from './sort'

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

const MAX_OCCLUSION_QUERIES = 64

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
  u_emissiveBrightness: WebGLUniformLocation | null
  u_colorMap: WebGLUniformLocation | null
  u_aoMap: WebGLUniformLocation | null
  u_aoIntensity: WebGLUniformLocation | null
}

interface PostUniformLocs {
  u_srcTexture: WebGLUniformLocation | null
  u_texelSize: WebGLUniformLocation | null
  u_useKarisAverage: WebGLUniformLocation | null
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
    u_emissiveBrightness: gl.getUniformLocation(program, 'u_emissiveBrightness'),
    u_colorMap: gl.getUniformLocation(program, 'u_colorMap'),
    u_aoMap: gl.getUniformLocation(program, 'u_aoMap'),
    u_aoIntensity: gl.getUniformLocation(program, 'u_aoIntensity'),
  }
}

const cachePostLocs = (gl: WebGL2RenderingContext, program: WebGLProgram): PostUniformLocs => ({
  u_srcTexture: gl.getUniformLocation(program, 'u_srcTexture'),
  u_texelSize: gl.getUniformLocation(program, 'u_texelSize'),
  u_useKarisAverage: gl.getUniformLocation(program, 'u_useKarisAverage'),
})

const cacheBlitLocs = (gl: WebGL2RenderingContext, program: WebGLProgram): BlitUniformLocs => ({
  u_sceneTexture: gl.getUniformLocation(program, 'u_sceneTexture'),
  u_bloomTexture: gl.getUniformLocation(program, 'u_bloomTexture'),
  u_bloomIntensity: gl.getUniformLocation(program, 'u_bloomIntensity'),
})

// ─── GPU buffer management ────────────────────────────────────────────

const ensureGPUBuffers = (gl: WebGL2RenderingContext, geometry: Geometry) => {
  if (geometry._gpuBuffers && !geometry.needsUpdate) return

  // Re-upload position and normal data into existing buffers for dynamic geometry
  if (geometry._gpuBuffers && geometry.needsUpdate) {
    const bufs = geometry._gpuBuffers as GPUBuffers
    gl.bindBuffer(gl.ARRAY_BUFFER, bufs.position)
    gl.bufferData(gl.ARRAY_BUFFER, geometry.positions, gl.DYNAMIC_DRAW)
    const packedNormals = packNormalsSnorm8(geometry.normals, geometry.vertexCount)
    gl.bindBuffer(gl.ARRAY_BUFFER, bufs.normal)
    gl.bufferData(gl.ARRAY_BUFFER, packedNormals, gl.DYNAMIC_DRAW)
    geometry.needsUpdate = false
    return
  }

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
  geometry.needsUpdate = false
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
    bloomFbos,
    bloomTextures,
    bloomWidths,
    bloomHeights,
    width: w,
    height: h,
  }
}

// ─── Renderer ─────────────────────────────────────────────────────────

export { type RendererConfig, type FrameStats } from './renderer'

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
  private lambertTexturedProgram: WebGLProgram
  private basicProgram: WebGLProgram
  private lambertSkinnedProgram: WebGLProgram
  private basicSkinnedProgram: WebGLProgram
  private shadowDepthProgram: WebGLProgram
  private shadowDepthSkinnedProgram: WebGLProgram
  private bloomDownsampleProgram: WebGLProgram
  private bloomUpsampleProgram: WebGLProgram
  private blitProgram: WebGLProgram

  // Cached uniform locations
  private _lambertLocs!: SceneUniformLocs
  private _lambertTexturedLocs!: SceneUniformLocs
  private _basicLocs!: SceneUniformLocs
  private _lambertSkinnedLocs!: SceneUniformLocs
  private _basicSkinnedLocs!: SceneUniformLocs

  // Texture map cache
  private _glTexCache = new WeakMap<Texture, WebGLTexture>()
  private _dummyWhiteTex!: WebGLTexture
  private _bloomDownLocs!: PostUniformLocs
  private _bloomUpLocs!: PostUniformLocs
  private _blitLocs!: BlitUniformLocs

  // ─── GL state cache ─────────────────────────────────────────────
  private _glProgram: WebGLProgram | null = null
  private _glVAO: WebGLVertexArrayObject | null = null
  private _glMaterial: Material | null = null
  private _glDepthTest = true
  private _glDepthMask = true
  private _glBlend = false
  private _glCullFace = true
  private _glCullMode = 0x0405 // gl.BACK
  private _glColorMaskAll = true // simplified: all channels same
  private _glViewportW = -1
  private _glViewportH = -1
  private _glFbo: WebGLFramebuffer | null | undefined = undefined // undefined = unknown

  private renderTargets: RenderTargets | null = null
  private samples: number
  private bloomLevels: number
  private bloomIntensity: number
  private bloomEnabled: boolean

  // Shadow config
  private shadowEnabled: boolean
  private shadowResolution: number
  shadowsBaked = false
  private _shadowIsBaked = false
  private _prevShadowsBaked = false

  // Shadow GPU resources
  private _shadowTexture!: WebGLTexture
  private _shadowFbo!: WebGLFramebuffer
  private _shadowUBO!: WebGLBuffer
  private _shadowUBData = new Float32Array(16) // mat4 = 64 bytes
  private _shadowVAOs = new WeakMap<Geometry, WebGLVertexArrayObject>()
  private _shadowSkinnedVAOs = new WeakMap<Geometry, WebGLVertexArrayObject>()

  // Occlusion query resources
  private _occlusionProgram!: WebGLProgram
  private _occlusionQueries: WebGLQuery[] = []
  private _occlusionUBO!: WebGLBuffer
  private _occlusionMVPData = new Float32Array(16) // single mat4
  private _occludees: Mesh[] = []
  private _occlusionPrevOccludees: Mesh[] = []
  private _occlusionPrevCount = 0
  private _occlusionWorldAABB: AABB = new Float32Array(6)
  private _occlusionMVP: Mat4 = mat4Create()
  private _occlusionBoxTransform: Mat4 = mat4Create()

  // Shadow scratch
  private _shadowMeshes: Mesh[] = []
  private _shadowVP: Mat4 = mat4Create()
  private _shadowLightView: Mat4 = mat4Create()
  private _shadowLightProj: Mat4 = mat4Create()
  private _frameNum = 0

  // Traversal
  private _traversalStack: Node[] = []

  // UBOs
  // FrameBlock (binding 0): 208 bytes = 52 floats (VP + light + shadow data)
  private _frameUBO!: WebGLBuffer
  private _frameData = new Float32Array(52)
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
      this.shadowResolution = 2048
    } else if (typeof shadowConfig === 'object') {
      this.shadowEnabled = shadowConfig.enabled !== false
      this.shadowResolution = shadowConfig.resolution ?? 2048
    } else {
      this.shadowEnabled = true
      this.shadowResolution = 2048
    }

    // Compile programs
    this.lambertProgram = createProgram(gl, LAMBERT_VERT, LAMBERT_FRAG)
    this.lambertTexturedProgram = createProgram(gl, LAMBERT_VERT, LAMBERT_TEXTURED_FRAG)
    this.basicProgram = createProgram(gl, BASIC_VERT, BASIC_FRAG)
    this.lambertSkinnedProgram = createProgram(gl, LAMBERT_SKINNED_VERT, LAMBERT_FRAG)
    this.basicSkinnedProgram = createProgram(gl, BASIC_SKINNED_VERT, BASIC_FRAG)
    this.shadowDepthProgram = createProgram(gl, SHADOW_DEPTH_VERT, SHADOW_DEPTH_FRAG)
    this.shadowDepthSkinnedProgram = createProgram(gl, SHADOW_DEPTH_SKINNED_VERT, SHADOW_DEPTH_FRAG)
    this.bloomDownsampleProgram = createProgram(gl, FULLSCREEN_VERT, BLOOM_DOWNSAMPLE_FRAG)
    this.bloomUpsampleProgram = createProgram(gl, FULLSCREEN_VERT, BLOOM_UPSAMPLE_FRAG)
    this.blitProgram = createProgram(gl, FULLSCREEN_VERT, BLIT_FRAG)

    // Cache uniform locations
    this._lambertLocs = cacheSceneLocs(gl, this.lambertProgram)
    this._lambertTexturedLocs = cacheSceneLocs(gl, this.lambertTexturedProgram)
    this._basicLocs = cacheSceneLocs(gl, this.basicProgram)
    this._lambertSkinnedLocs = cacheSceneLocs(gl, this.lambertSkinnedProgram)
    this._basicSkinnedLocs = cacheSceneLocs(gl, this.basicSkinnedProgram)
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

    // Frame UBO (208 bytes = 52 floats)
    this._frameUBO = gl.createBuffer()!
    gl.bindBuffer(gl.UNIFORM_BUFFER, this._frameUBO)
    gl.bufferData(gl.UNIFORM_BUFFER, 208, gl.DYNAMIC_DRAW)

    // Dynamic object / skinned UBOs
    this._createDynamicBuffers()

    // Bind UBO block indices for all scene programs (done once at init)
    this._bindUBOBlocks(this.lambertProgram, false)
    this._bindUBOBlocks(this.lambertTexturedProgram, false)
    this._bindUBOBlocks(this.basicProgram, false)
    this._bindUBOBlocks(this.lambertSkinnedProgram, true)
    this._bindUBOBlocks(this.basicSkinnedProgram, true)
    // Shadow UBO (binding 2, 64 bytes = mat4)
    this._shadowUBO = gl.createBuffer()!
    gl.bindBuffer(gl.UNIFORM_BUFFER, this._shadowUBO)
    gl.bufferData(gl.UNIFORM_BUFFER, 64, gl.DYNAMIC_DRAW)

    // Bind ShadowBlock for shadow programs
    this._bindShadowUBOBlocks(this.shadowDepthProgram, false)
    this._bindShadowUBOBlocks(this.shadowDepthSkinnedProgram, true)

    // Shadow texture (TEXTURE_2D with depth comparison)
    const shadowRes = this.shadowEnabled ? this.shadowResolution : 1
    this._shadowTexture = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, this._shadowTexture)
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.DEPTH_COMPONENT24, shadowRes, shadowRes)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_MODE, gl.COMPARE_REF_TO_TEXTURE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_FUNC, gl.LEQUAL)

    // Shadow FBO
    this._shadowFbo = gl.createFramebuffer()!
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._shadowFbo)
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, this._shadowTexture, 0)
    gl.drawBuffers([])
    gl.readBuffer(gl.NONE)
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)

    // Occlusion query program + resources
    this._occlusionProgram = createProgram(gl, OCCLUSION_BOX_VERT, OCCLUSION_BOX_FRAG)
    const occBlockIdx = gl.getUniformBlockIndex(this._occlusionProgram, 'OcclusionBlock')
    if (occBlockIdx !== gl.INVALID_INDEX) gl.uniformBlockBinding(this._occlusionProgram, occBlockIdx, 3)
    this._occlusionUBO = gl.createBuffer()!
    gl.bindBuffer(gl.UNIFORM_BUFFER, this._occlusionUBO)
    gl.bufferData(gl.UNIFORM_BUFFER, 64, gl.DYNAMIC_DRAW) // 1 mat4 = 64 bytes
    for (let i = 0; i < MAX_OCCLUSION_QUERIES; i++) {
      this._occlusionQueries.push(gl.createQuery()!)
    }

    // 1x1 white dummy texture for missing texture maps (sampling white = identity)
    this._dummyWhiteTex = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, this._dummyWhiteTex)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([255, 255, 255, 255]))
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.bindTexture(gl.TEXTURE_2D, null)

    // Cache canvas dimensions
    this._displayW = canvas.clientWidth
    this._displayH = canvas.clientHeight
    this._resizeObserver = new ResizeObserver(() => {
      this._displayW = this.canvas.clientWidth
      this._displayH = this.canvas.clientHeight
    })
    this._resizeObserver.observe(canvas)
  }

  private _ensureGLTexture(tex: Texture): WebGLTexture {
    const cached = this._glTexCache.get(tex)
    if (cached) return cached

    const gl = this.gl
    const glTex = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, glTex)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, tex.width, tex.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, tex.data)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.bindTexture(gl.TEXTURE_2D, null)

    this._glTexCache.set(tex, glTex)
    return glTex
  }

  private _bindMaterialTextures(material: Material, locs: SceneUniformLocs) {
    const gl = this.gl

    // Color map → texture unit 3
    gl.activeTexture(gl.TEXTURE3)
    gl.bindTexture(gl.TEXTURE_2D, material.colorMap ? this._ensureGLTexture(material.colorMap) : this._dummyWhiteTex)
    gl.uniform1i(locs.u_colorMap, 3)

    // AO map → texture unit 4
    gl.activeTexture(gl.TEXTURE4)
    gl.bindTexture(gl.TEXTURE_2D, material.aoMap ? this._ensureGLTexture(material.aoMap) : this._dummyWhiteTex)
    gl.uniform1i(locs.u_aoMap, 4)

    gl.uniform1f(locs.u_aoIntensity, material.aoIntensity)
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

  // ─── Shadow map computation ─────────────────────────────────────

  private _computeShadowMatrix(dirLight: DirectionalLight, lightDir: Vec3): void {
    computeShadowMatrix(
      this._shadowVP,
      lightDir,
      dirLight.shadowMapSize,
      dirLight.shadowNear,
      dirLight.shadowFar,
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
    for (const fbo of rt.bloomFbos) gl.deleteFramebuffer(fbo)
    for (const tex of rt.bloomTextures) gl.deleteTexture(tex)
    this.renderTargets = null
  }

  // ─── GL state cache helpers ──────────────────────────────────────

  private _setProgram(p: WebGLProgram): boolean {
    if (p === this._glProgram) return false
    this.gl.useProgram(p)
    this._glProgram = p
    this._glMaterial = null // force material re-upload on program change
    return true
  }

  private _setVAO(vao: WebGLVertexArrayObject | null): void {
    if (vao === this._glVAO) return
    this.gl.bindVertexArray(vao)
    this._glVAO = vao
  }

  private _setDepthTest(on: boolean): void {
    if (on === this._glDepthTest) return
    if (on) this.gl.enable(this.gl.DEPTH_TEST)
    else this.gl.disable(this.gl.DEPTH_TEST)
    this._glDepthTest = on
  }

  private _setDepthMask(on: boolean): void {
    if (on === this._glDepthMask) return
    this.gl.depthMask(on)
    this._glDepthMask = on
  }

  private _setBlend(on: boolean): void {
    if (on === this._glBlend) return
    if (on) this.gl.enable(this.gl.BLEND)
    else this.gl.disable(this.gl.BLEND)
    this._glBlend = on
  }

  private _setCullFace(on: boolean): void {
    if (on === this._glCullFace) return
    if (on) this.gl.enable(this.gl.CULL_FACE)
    else this.gl.disable(this.gl.CULL_FACE)
    this._glCullFace = on
  }

  private _setCullMode(mode: number): void {
    if (mode === this._glCullMode) return
    this.gl.cullFace(mode)
    this._glCullMode = mode
  }

  private _setColorMask(on: boolean): void {
    if (on === this._glColorMaskAll) return
    this.gl.colorMask(on, on, on, on)
    this._glColorMaskAll = on
  }

  private _setViewport(w: number, h: number): void {
    if (w === this._glViewportW && h === this._glViewportH) return
    this.gl.viewport(0, 0, w, h)
    this._glViewportW = w
    this._glViewportH = h
  }

  private _setFbo(fbo: WebGLFramebuffer | null): void {
    if (fbo === this._glFbo) return
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, fbo)
    this._glFbo = fbo
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

    // Reset GL state cache (matches state left by previous frame's cleanup)
    this._glProgram = null
    this._glVAO = null
    this._glMaterial = null
    this._glDepthTest = true
    this._glDepthMask = true
    this._glBlend = false
    this._glCullFace = true
    this._glCullMode = 0x0405
    this._glColorMaskAll = true
    this._glViewportW = -1
    this._glViewportH = -1
    this._glFbo = undefined

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

    // Find lights (quick early-exit traversals)
    const dirLight = findDirectionalLight(scene, this._traversalStack)
    const ambLight = findAmbientLight(scene, this._traversalStack)

    // Compute light direction from world matrix
    const lightDir = this._lightDir
    computeLightDir(lightDir, this._tempVec3, dirLight)

    this._frameNum++
    const frameNum = this._frameNum
    const shadowActive = this.shadowEnabled && !!dirLight && dirLight.castShadow

    let drawCalls = 0
    let shadowDrawCalls = 0
    let triangles = 0

    // ─── Shadow computation ────────────────────────────────────────
    // Compute shadow matrix BEFORE traversal so we can collect shadow-only
    // casters in the same pass as camera-visible meshes.
    let shadowFrustum: Float32Array | null = null
    if (shadowActive) {
      this._computeShadowMatrix(dirLight!, lightDir)
      frustumFromViewProjection(this._shadowFrustumPlanes, this._shadowVP)
      shadowFrustum = this._shadowFrustumPlanes
    }

    // ─── Apply previous frame's occlusion query results ─────────
    const gl2 = gl
    for (let i = 0; i < this._occlusionPrevCount; i++) {
      const query = this._occlusionQueries[i]!
      if (gl2.getQueryParameter(query, gl2.QUERY_RESULT_AVAILABLE)) {
        const visible = gl2.getQueryParameter(query, gl2.QUERY_RESULT) as number
        this._occlusionPrevOccludees[i]!._occluded = visible === 0
      }
    }

    // ─── Single merged traversal: camera meshes + shadow-only casters ───
    const meshes = this._meshes
    const shadowMeshes = this._shadowMeshes
    const occludees = this._occludees
    const culledCount = collectMeshes(
      scene,
      this._frustumPlanes,
      shadowFrustum,
      this._worldAABB,
      meshes,
      shadowMeshes,
      occludees,
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

    // ─── Upload frame UBO (52 floats / 208 bytes) ─────────────────
    // std140 layout:
    //   mat4 u_viewProjection       (floats 0-15)
    //   vec3 u_lightDirection + pad  (floats 16-19)
    //   vec3 u_lightColor + intensity (floats 20-23)
    //   vec3 u_ambientColor + ambientIntensity (floats 24-27)
    //   float u_shadowEnabled + 3 pad (floats 28-31)
    //   mat4 u_shadowVP              (floats 32-47)
    //   float constantBias, slopeBias, invMapSize, pad (floats 48-51)
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
    fd[24] = ambLight ? ambLight.color[0] : 0
    fd[25] = ambLight ? ambLight.color[1] : 0
    fd[26] = ambLight ? ambLight.color[2] : 0
    fd[27] = ambLight ? ambLight.intensity : 0
    fd[28] = shadowActive ? 1.0 : 0.0
    // fd[29-31] = 0 (padding for mat4 alignment, already zeroed)
    if (shadowActive) {
      fd.set(this._shadowVP, 32)
      fd[48] = dirLight!.shadowBias
      fd[49] = dirLight!.shadowSlopeBias
      fd[50] = 1.0 / this.shadowResolution
    }
    gl.bindBuffer(gl.UNIFORM_BUFFER, this._frameUBO)
    gl.bufferSubData(gl.UNIFORM_BUFFER, 0, fd)
    gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, this._frameUBO)

    // ─── Shadow baking ──────────────────────────────────────────────
    // When shadowsBaked transitions false→true, force one final shadow render
    // so the shadow map captures the current scene (e.g. meshes that just mounted).
    if (!this.shadowsBaked) {
      this._shadowIsBaked = false
    } else if (!this._prevShadowsBaked) {
      this._shadowIsBaked = false
    }
    this._prevShadowsBaked = this.shadowsBaked

    // ─── Shadow render pass (single depth-only pass) ──────────────
    if (shadowActive && !(this.shadowsBaked && this._shadowIsBaked)) {
      this._setCullMode(gl.FRONT)
      this._setColorMask(false)

      const shadowRes = this.shadowResolution
      const PAD = 0.3

      this._setFbo(this._shadowFbo)
      this._setViewport(shadowRes, shadowRes)
      gl.clear(gl.DEPTH_BUFFER_BIT)

      // Write shadow UBO
      this._shadowUBData.set(this._shadowVP)
      gl.bindBuffer(gl.UNIFORM_BUFFER, this._shadowUBO)
      gl.bufferSubData(gl.UNIFORM_BUFFER, 0, this._shadowUBData)
      gl.bindBufferBase(gl.UNIFORM_BUFFER, 2, this._shadowUBO)

      const svp = this._shadowVP

      // Draw camera-visible shadow casters
      for (let si = 0; si < meshes.length; si++) {
        const mesh = meshes[sortedIndices[si]!]!
        if (!mesh.castShadow) continue

        // Light-space frustum cull
        const wm = mesh._worldMatrix
        const wx = wm[12]!,
          wy = wm[13]!,
          wz = wm[14]!
        const lx = svp[0]! * wx + svp[4]! * wy + svp[8]! * wz + svp[12]!
        const ly = svp[1]! * wx + svp[5]! * wy + svp[9]! * wz + svp[13]!
        const lz = svp[2]! * wx + svp[6]! * wy + svp[10]! * wz + svp[14]!
        if (lx < -(1 + PAD) || lx > 1 + PAD || ly < -(1 + PAD) || ly > 1 + PAD || lz < -(1 + PAD) || lz > 1 + PAD)
          continue

        ensureGPUBuffers(gl, mesh.geometry)
        const idxType = mesh.geometry.indices instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT

        if (mesh._isSkinned) {
          this._setProgram(this.shadowDepthSkinnedProgram)
          this._setVAO(this.ensureShadowVAO(mesh.geometry, true))
          gl.bindBufferRange(
            gl.UNIFORM_BUFFER,
            1,
            this._skinnedDynBuf,
            mesh._batchIndex * this._alignedSkinnedSize,
            2176,
          )
        } else {
          this._setProgram(this.shadowDepthProgram)
          this._setVAO(this.ensureShadowVAO(mesh.geometry, false))
          gl.bindBufferRange(gl.UNIFORM_BUFFER, 1, this._objectDynBuf, mesh._batchIndex * this._alignedObjectSize, 128)
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
        const lx = svp[0]! * wx + svp[4]! * wy + svp[8]! * wz + svp[12]!
        const ly = svp[1]! * wx + svp[5]! * wy + svp[9]! * wz + svp[13]!
        const lz = svp[2]! * wx + svp[6]! * wy + svp[10]! * wz + svp[14]!
        if (lx < -(1 + PAD) || lx > 1 + PAD || ly < -(1 + PAD) || ly > 1 + PAD || lz < -(1 + PAD) || lz > 1 + PAD)
          continue

        ensureGPUBuffers(gl, mesh.geometry)
        const idxType = mesh.geometry.indices instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT

        if (mesh._isSkinned) {
          this._setProgram(this.shadowDepthSkinnedProgram)
          this._setVAO(this.ensureShadowVAO(mesh.geometry, true))
          gl.bindBufferRange(
            gl.UNIFORM_BUFFER,
            1,
            this._skinnedDynBuf,
            mesh._batchIndex * this._alignedSkinnedSize,
            2176,
          )
        } else {
          this._setProgram(this.shadowDepthProgram)
          this._setVAO(this.ensureShadowVAO(mesh.geometry, false))
          gl.bindBufferRange(gl.UNIFORM_BUFFER, 1, this._objectDynBuf, mesh._batchIndex * this._alignedObjectSize, 128)
        }

        gl.drawElements(gl.TRIANGLES, mesh.geometry.indexCount, idxType, 0)
        shadowDrawCalls++
      }

      this._setVAO(null)
      this._setCullMode(gl.BACK)
      this._setColorMask(true)
      this._shadowIsBaked = true
    }

    this.ensureRenderTargets()
    const rt = this.renderTargets!

    // ─── Scene pass (MSAA MRT) ────────────────────────────────────
    this._setFbo(rt.msaaFbo)
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1])
    this._setViewport(rt.width, rt.height)
    gl.clearColor(0, 0, 0, 1.0)
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
    this._setDepthTest(true)
    this._setDepthMask(true)

    // Bind shadow texture to unit 2
    gl.activeTexture(gl.TEXTURE2)
    gl.bindTexture(gl.TEXTURE_2D, this._shadowTexture)

    // Find the split between opaque and transparent meshes
    const transparentStart = findTransparentStart(this._sortState, meshes.length)

    // ─── Draw helper (shared by opaque + transparent loops) ──────
    const drawMesh = (si: number, locs: SceneUniformLocs, programChanged: boolean, mesh: Mesh) => {
      ensureGPUBuffers(gl, mesh.geometry)
      this._setVAO((mesh.geometry._gpuBuffers as GPUBuffers).vao!)

      if (mesh._isSkinned) {
        gl.bindBufferRange(gl.UNIFORM_BUFFER, 1, this._skinnedDynBuf, mesh._batchIndex * this._alignedSkinnedSize, 2176)
      } else {
        gl.bindBufferRange(gl.UNIFORM_BUFFER, 1, this._objectDynBuf, mesh._batchIndex * this._alignedObjectSize, 128)
      }

      const materialChanged = mesh.material !== this._glMaterial || programChanged
      if (materialChanged) {
        this._glMaterial = mesh.material
        gl.uniform3fv(locs.u_baseColor, mesh.material.color)
        gl.uniform1f(locs.u_opacity, mesh.material.opacity)
        if (mesh.material.type === 'lambert') {
          const hasPalette = !!mesh.material.palette && mesh.geometry.hasAttribute('materialIndex')
          gl.uniform1i(locs.u_hasPalette, hasPalette ? 1 : 0)
          gl.uniform1i(locs.u_receiveShadow, mesh.material.receiveShadow ? 1 : 0)
          gl.uniform1f(locs.u_emissiveBrightness, mesh.material.emissiveBrightness)
          if (hasPalette && mesh.material.palette) {
            for (let i = 0; i < 32; i++) {
              const entry: PaletteEntry = mesh.material.palette[i] ?? { color: [1, 1, 1] }
              gl.uniform4f(locs.u_paletteColor[i]!, entry.color[0], entry.color[1], entry.color[2], 1.0)
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

    // ─── Opaque draw loop ────────────────────────────────────────
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
      } else if (mesh.material._hasTextures && mesh.material.type === 'lambert') {
        program = this.lambertTexturedProgram
        locs = this._lambertTexturedLocs
      } else {
        if (mesh.material.type === 'lambert') {
          program = this.lambertProgram
          locs = this._lambertLocs
        } else {
          program = this.basicProgram
          locs = this._basicLocs
        }
      }

      const programChanged = this._setProgram(program)
      if (programChanged && mesh.material.type === 'lambert') {
        gl.uniform1i(locs.u_shadowMap, 2)
      }
      if (mesh.material._hasTextures && !mesh._isSkinned) {
        this._bindMaterialTextures(mesh.material, locs)
      }

      drawMesh(si, locs, programChanged, mesh)
    }

    // ─── Transparent draw loop (back-to-front, blend on, depth write off) ──
    if (transparentStart < meshes.length) {
      this._setBlend(true)
      gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA)
      this._setDepthMask(false)
      this._setCullFace(false)

      for (let si = transparentStart; si < meshes.length; si++) {
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
        } else if (mesh.material._hasTextures && mesh.material.type === 'lambert') {
          program = this.lambertTexturedProgram
          locs = this._lambertTexturedLocs
        } else {
          if (mesh.material.type === 'lambert') {
            program = this.lambertProgram
            locs = this._lambertLocs
          } else {
            program = this.basicProgram
            locs = this._basicLocs
          }
        }

        const programChanged = this._setProgram(program)
        if (programChanged && mesh.material.type === 'lambert') {
          gl.uniform1i(locs.u_shadowMap, 2)
        }
        if (mesh.material._hasTextures && !mesh._isSkinned) {
          this._bindMaterialTextures(mesh.material, locs)
        }

        drawMesh(si, locs, programChanged, mesh)
      }

      // Restore state
      this._setBlend(false)
      this._setDepthMask(true)
      this._setCullFace(true)
    }

    // ─── Occlusion query pass (test AABB boxes against scene depth) ──
    const occludeeCount = Math.min(occludees.length, MAX_OCCLUSION_QUERIES)
    if (occludeeCount > 0) {
      this._setColorMask(false)
      this._setDepthMask(false)
      this._setCullFace(false)
      this._setProgram(this._occlusionProgram)
      this._setVAO(null) // No VAO needed — vertices generated from gl_VertexID

      for (let i = 0; i < occludeeCount; i++) {
        const mesh = occludees[i]!
        // Transform geometry AABB to world space
        aabbTransform(this._occlusionWorldAABB, mesh.geometry.aabb, mesh._worldMatrix)
        const aabb = this._occlusionWorldAABB
        const cx = (aabb[0]! + aabb[3]!) * 0.5
        const cy = (aabb[1]! + aabb[4]!) * 0.5
        const cz = (aabb[2]! + aabb[5]!) * 0.5
        const hx = (aabb[3]! - aabb[0]!) * 0.5
        const hy = (aabb[4]! - aabb[1]!) * 0.5
        const hz = (aabb[5]! - aabb[2]!) * 0.5
        // Box transform: T(center) * S(halfExtent)
        const bt = this._occlusionBoxTransform
        bt[0] = hx
        bt[1] = 0
        bt[2] = 0
        bt[3] = 0
        bt[4] = 0
        bt[5] = hy
        bt[6] = 0
        bt[7] = 0
        bt[8] = 0
        bt[9] = 0
        bt[10] = hz
        bt[11] = 0
        bt[12] = cx
        bt[13] = cy
        bt[14] = cz
        bt[15] = 1
        mat4Multiply(this._occlusionMVP, this._vpMatrix, bt)
        this._occlusionMVPData.set(this._occlusionMVP)

        gl.bindBuffer(gl.UNIFORM_BUFFER, this._occlusionUBO)
        gl.bufferSubData(gl.UNIFORM_BUFFER, 0, this._occlusionMVPData)
        gl.bindBufferBase(gl.UNIFORM_BUFFER, 3, this._occlusionUBO)

        gl.beginQuery(gl.ANY_SAMPLES_PASSED_CONSERVATIVE, this._occlusionQueries[i]!)
        gl.drawArrays(gl.TRIANGLES, 0, 36)
        gl.endQuery(gl.ANY_SAMPLES_PASSED_CONSERVATIVE)
      }

      // Store occludees for next frame's result application
      for (let i = 0; i < occludeeCount; i++) {
        this._occlusionPrevOccludees[i] = occludees[i]!
      }
      this._occlusionPrevCount = occludeeCount

      // Restore state
      this._setColorMask(true)
      this._setDepthMask(true)
      this._setCullFace(true)
    }

    this._setVAO(null)

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
    this._glFbo = undefined // DRAW_FRAMEBUFFER changed outside cache

    // ─── Bloom ─────────────────────────────────────────────────────
    if (this.bloomEnabled && this.bloomLevels > 0) {
      this._setDepthTest(false)
      this._setDepthMask(false)

      // Downsample chain
      this._setProgram(this.bloomDownsampleProgram)
      let srcTex = rt.resolvedEmissiveTex
      let srcW = rt.width
      let srcH = rt.height

      const downLocs = this._bloomDownLocs
      for (let i = 0; i < this.bloomLevels; i++) {
        this._setFbo(rt.bloomFbos[i]!)
        this._setViewport(rt.bloomWidths[i]!, rt.bloomHeights[i]!)

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
      this._setProgram(this.bloomUpsampleProgram)
      this._setBlend(true)
      gl.blendFunc(gl.ONE, gl.ONE)

      const upLocs = this._bloomUpLocs
      for (let i = this.bloomLevels - 1; i > 0; i--) {
        this._setFbo(rt.bloomFbos[i - 1]!)
        this._setViewport(rt.bloomWidths[i - 1]!, rt.bloomHeights[i - 1]!)

        gl.activeTexture(gl.TEXTURE0)
        gl.bindTexture(gl.TEXTURE_2D, rt.bloomTextures[i]!)
        gl.uniform1i(upLocs.u_srcTexture, 0)
        gl.uniform2f(upLocs.u_texelSize, 1 / rt.bloomWidths[i]!, 1 / rt.bloomHeights[i]!)

        gl.drawArrays(gl.TRIANGLES, 0, 3)
      }

      this._setBlend(false)
    }

    // ─── Final blit ────────────────────────────────────────────────
    this._setFbo(null)
    this._setViewport(rt.width, rt.height)
    this._setDepthTest(false)
    this._setDepthMask(false)

    this._setProgram(this.blitProgram)
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
    this._setDepthTest(true)
    this._setDepthMask(true)

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
    gl.deleteFramebuffer(this._shadowFbo)
    gl.deleteProgram(this.lambertProgram)
    gl.deleteProgram(this.lambertTexturedProgram)
    gl.deleteProgram(this.basicProgram)
    gl.deleteProgram(this.lambertSkinnedProgram)
    gl.deleteProgram(this.basicSkinnedProgram)
    gl.deleteProgram(this.shadowDepthProgram)
    gl.deleteProgram(this.shadowDepthSkinnedProgram)
    gl.deleteProgram(this._occlusionProgram)
    gl.deleteBuffer(this._occlusionUBO)
    for (const q of this._occlusionQueries) gl.deleteQuery(q)
    gl.deleteProgram(this.bloomDownsampleProgram)
    gl.deleteProgram(this.bloomUpsampleProgram)
    gl.deleteProgram(this.blitProgram)
    this._resizeObserver?.disconnect()
    this._resizeObserver = null
  }
}
