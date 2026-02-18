import {
  aabbTransform,
  frustumContainsAABB,
  frustumFromViewProjection,
  mat4Create,
  mat4Invert,
  mat4Multiply,
  mat4Transpose,
  vec3Create,
  vec3Normalize,
} from '../math/index.ts'
import { Mesh } from '../scene/mesh.ts'
import { Node } from '../scene/node.ts'
import {
  LAMBERT_VERT,
  LAMBERT_FRAG,
  LAMBERT_SKINNED_VERT,
  BASIC_VERT,
  BASIC_FRAG,
  BASIC_SKINNED_VERT,
  FULLSCREEN_VERT,
  BLOOM_DOWNSAMPLE_FRAG,
  BLOOM_UPSAMPLE_FRAG,
  BLIT_FRAG,
} from './shaders.ts'

import type { Geometry } from '../geometry/geometry.ts'
import type { PaletteEntry } from '../materials/material.ts'
import type { AABB, Mat4 } from '../math/index.ts'
import type { PerspectiveCamera } from '../scene/camera.ts'
import type { DirectionalLight } from '../scene/light.ts'
import type { Scene } from '../scene/scene.ts'
import type { Renderer, RendererConfig, FrameStats } from './renderer.ts'

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

// ─── GPU buffer management ────────────────────────────────────────────

const ensureGPUBuffers = (gl: WebGL2RenderingContext, geometry: Geometry, _program: WebGLProgram) => {
  if (geometry._gpuBuffers) return

  const posBuffer = gl.createBuffer()!
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer)
  gl.bufferData(gl.ARRAY_BUFFER, geometry.positions, gl.STATIC_DRAW)

  const normBuffer = gl.createBuffer()!
  gl.bindBuffer(gl.ARRAY_BUFFER, normBuffer)
  gl.bufferData(gl.ARRAY_BUFFER, geometry.normals, gl.STATIC_DRAW)

  const idxBuffer = gl.createBuffer()!
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuffer)
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, geometry.indices, gl.STATIC_DRAW)

  let uvBuffer: WebGLBuffer | undefined
  if (geometry.uvs) {
    uvBuffer = gl.createBuffer()!
    gl.bindBuffer(gl.ARRAY_BUFFER, uvBuffer)
    gl.bufferData(gl.ARRAY_BUFFER, geometry.uvs, gl.STATIC_DRAW)
  }

  let matIdxBuffer: WebGLBuffer | undefined
  if (geometry.materialIndices) {
    matIdxBuffer = gl.createBuffer()!
    gl.bindBuffer(gl.ARRAY_BUFFER, matIdxBuffer)
    // Convert Uint8 to Float32 for the shader attribute
    const floatIndices = new Float32Array(geometry.materialIndices.length)
    for (let i = 0; i < geometry.materialIndices.length; i++) {
      floatIndices[i] = geometry.materialIndices[i]!
    }
    gl.bufferData(gl.ARRAY_BUFFER, floatIndices, gl.STATIC_DRAW)
  }

  // Joints buffer (location 4) — convert to float for vertexAttribPointer
  let jointsBuffer: WebGLBuffer | undefined
  if (geometry.joints) {
    jointsBuffer = gl.createBuffer()!
    gl.bindBuffer(gl.ARRAY_BUFFER, jointsBuffer)
    const floatJoints = new Float32Array(geometry.joints.length)
    for (let i = 0; i < geometry.joints.length; i++) {
      floatJoints[i] = geometry.joints[i]!
    }
    gl.bufferData(gl.ARRAY_BUFFER, floatJoints, gl.STATIC_DRAW)
  }

  // Weights buffer (location 5)
  let weightsBuffer: WebGLBuffer | undefined
  if (geometry.weights) {
    weightsBuffer = gl.createBuffer()!
    gl.bindBuffer(gl.ARRAY_BUFFER, weightsBuffer)
    gl.bufferData(gl.ARRAY_BUFFER, geometry.weights, gl.STATIC_DRAW)
  }

  // Create VAO
  const vao = gl.createVertexArray()!
  gl.bindVertexArray(vao)

  // Position (location 0)
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer)
  gl.enableVertexAttribArray(0)
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0)

  // Normal (location 1)
  gl.bindBuffer(gl.ARRAY_BUFFER, normBuffer)
  gl.enableVertexAttribArray(1)
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0)

  // UV (location 2)
  if (uvBuffer) {
    gl.bindBuffer(gl.ARRAY_BUFFER, uvBuffer)
    gl.enableVertexAttribArray(2)
    gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 0, 0)
  }

  // Material index (location 3)
  if (matIdxBuffer) {
    gl.bindBuffer(gl.ARRAY_BUFFER, matIdxBuffer)
    gl.enableVertexAttribArray(3)
    gl.vertexAttribPointer(3, 1, gl.FLOAT, false, 0, 0)
  } else {
    gl.disableVertexAttribArray(3)
    gl.vertexAttrib1f(3, 0.0)
  }

  // Joints (location 4) — as vec4 float
  if (jointsBuffer) {
    gl.bindBuffer(gl.ARRAY_BUFFER, jointsBuffer)
    gl.enableVertexAttribArray(4)
    gl.vertexAttribPointer(4, 4, gl.FLOAT, false, 0, 0)
  } else {
    gl.disableVertexAttribArray(4)
  }

  // Weights (location 5) — as vec4 float
  if (weightsBuffer) {
    gl.bindBuffer(gl.ARRAY_BUFFER, weightsBuffer)
    gl.enableVertexAttribArray(5)
    gl.vertexAttribPointer(5, 4, gl.FLOAT, false, 0, 0)
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

export { type RendererConfig, type FrameStats } from './renderer.ts'

export class WebGL2Renderer implements Renderer {
  readonly backend = 'webgl2' as const
  gl: WebGL2RenderingContext
  canvas: HTMLCanvasElement

  private lambertProgram: WebGLProgram
  private basicProgram: WebGLProgram
  private lambertSkinnedProgram: WebGLProgram
  private basicSkinnedProgram: WebGLProgram
  private bloomDownsampleProgram: WebGLProgram
  private bloomUpsampleProgram: WebGLProgram
  private blitProgram: WebGLProgram

  private renderTargets: RenderTargets | null = null
  private samples: number
  private bloomLevels: number
  private bloomIntensity: number
  private bloomEnabled: boolean

  // Scratch
  private _vpMatrix: Mat4 = mat4Create()
  private _invWorldMatrix: Mat4 = mat4Create()
  private _normalMatrix: Mat4 = mat4Create()
  private _frustumPlanes = new Float32Array(24)
  private _worldAABB: AABB = new Float32Array(6)

  // Stats
  private _lastFrameTime = 0
  private _frameCount = 0
  private _fpsAccumulator = 0
  private _currentFps = 60
  stats: FrameStats = {
    fps: 60,
    frameTime: 0,
    drawCalls: 0,
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

    this.gl = gl
    this.canvas = canvas
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

    // Compile programs
    this.lambertProgram = createProgram(gl, LAMBERT_VERT, LAMBERT_FRAG)
    this.basicProgram = createProgram(gl, BASIC_VERT, BASIC_FRAG)
    this.lambertSkinnedProgram = createProgram(gl, LAMBERT_SKINNED_VERT, LAMBERT_FRAG)
    this.basicSkinnedProgram = createProgram(gl, BASIC_SKINNED_VERT, BASIC_FRAG)
    this.bloomDownsampleProgram = createProgram(gl, FULLSCREEN_VERT, BLOOM_DOWNSAMPLE_FRAG)
    this.bloomUpsampleProgram = createProgram(gl, FULLSCREEN_VERT, BLOOM_UPSAMPLE_FRAG)
    this.blitProgram = createProgram(gl, FULLSCREEN_VERT, BLIT_FRAG)

    // Enable depth test
    gl.enable(gl.DEPTH_TEST)
    gl.depthFunc(gl.LEQUAL)
    gl.enable(gl.CULL_FACE)
    gl.cullFace(gl.BACK)
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

  render(scene: Scene, camera: PerspectiveCamera) {
    const gl = this.gl
    const now = performance.now()
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
    const dpr = Math.min(window.devicePixelRatio, 2)
    const displayW = Math.floor(this.canvas.clientWidth * dpr)
    const displayH = Math.floor(this.canvas.clientHeight * dpr)
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

    // Frustum culling
    frustumFromViewProjection(this._frustumPlanes, this._vpMatrix)

    // Collect visible meshes
    const meshes: Mesh[] = []
    let culledCount = 0
    scene.traverse((node: Node) => {
      if (!node.visible) return
      if (node instanceof Mesh) {
        if (node.frustumCulled) {
          aabbTransform(this._worldAABB, node.geometry.aabb, node._worldMatrix)
          if (!frustumContainsAABB(this._frustumPlanes, this._worldAABB)) {
            culledCount++
            return
          }
        }
        meshes.push(node)
      }
    })

    // Find directional light
    let dirLight: DirectionalLight | null = null
    scene.traverse((node: Node) => {
      if (node.type === 'directionalLight' && !dirLight) {
        dirLight = node as DirectionalLight
      }
    })

    // Compute light direction from world matrix
    const lightDir = vec3Create()
    if (dirLight) {
      // Direction = normalize(light position) as directional light
      const lp = (dirLight as DirectionalLight)._worldMatrix
      const lx = lp[12]!,
        ly = lp[13]!,
        lz = lp[14]!
      vec3Normalize(lightDir, new Float32Array([lx, ly, lz]))
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

    let drawCalls = 0
    let triangles = 0

    for (const mesh of meshes) {
      const isSkinned = !!mesh.skeleton && !!mesh.geometry.joints && !!mesh.geometry.weights
      let program: WebGLProgram
      if (isSkinned) {
        program = mesh.material.type === 'lambert' ? this.lambertSkinnedProgram : this.basicSkinnedProgram
      } else {
        program = mesh.material.type === 'lambert' ? this.lambertProgram : this.basicProgram
      }
      gl.useProgram(program)

      ensureGPUBuffers(gl, mesh.geometry, program)
      gl.bindVertexArray(mesh.geometry._gpuBuffers!.vao!)

      // Per-frame uniforms
      gl.uniformMatrix4fv(gl.getUniformLocation(program, 'u_viewProjection'), false, this._vpMatrix)

      // Per-object uniforms
      gl.uniformMatrix4fv(gl.getUniformLocation(program, 'u_worldMatrix'), false, mesh._worldMatrix)

      // Upload bone matrices for skinned meshes
      if (isSkinned) {
        mesh.skeleton!.update()
        gl.uniformMatrix4fv(gl.getUniformLocation(program, 'u_boneMatrices[0]'), false, mesh.skeleton!.boneMatrices)
      }

      if (mesh.material.type === 'lambert') {
        // Normal matrix = transpose(inverse(worldMatrix))
        if (mat4Invert(this._invWorldMatrix, mesh._worldMatrix)) {
          mat4Transpose(this._normalMatrix, this._invWorldMatrix)
        }
        gl.uniformMatrix4fv(gl.getUniformLocation(program, 'u_normalMatrix'), false, this._normalMatrix)

        // Light uniforms
        gl.uniform3fv(gl.getUniformLocation(program, 'u_lightDirection'), lightDir)
        if (dirLight) {
          gl.uniform3fv(gl.getUniformLocation(program, 'u_lightColor'), (dirLight as DirectionalLight).color)
          gl.uniform1f(gl.getUniformLocation(program, 'u_lightIntensity'), (dirLight as DirectionalLight).intensity)
        }

        // Ambient
        gl.uniform3fv(gl.getUniformLocation(program, 'u_ambientColor'), scene.ambientLight.color)
        gl.uniform1f(gl.getUniformLocation(program, 'u_ambientIntensity'), scene.ambientLight.intensity)

        // Material
        gl.uniform3fv(gl.getUniformLocation(program, 'u_baseColor'), mesh.material.color)
        gl.uniform1f(gl.getUniformLocation(program, 'u_opacity'), mesh.material.opacity)

        // Palette
        const hasPalette = !!mesh.material.palette && mesh.geometry.hasAttribute('materialIndex')
        gl.uniform1i(gl.getUniformLocation(program, 'u_hasPalette'), hasPalette ? 1 : 0)

        if (hasPalette && mesh.material.palette) {
          for (let i = 0; i < 32; i++) {
            const entry: PaletteEntry = mesh.material.palette[i] ?? { color: [1, 1, 1] }
            gl.uniform4f(
              gl.getUniformLocation(program, `u_palette[${i}].color`),
              entry.color[0],
              entry.color[1],
              entry.color[2],
              entry.opacity ?? 1.0,
            )
            gl.uniform4f(
              gl.getUniformLocation(program, `u_palette[${i}].emissive`),
              entry.emissive?.[0] ?? 0,
              entry.emissive?.[1] ?? 0,
              entry.emissive?.[2] ?? 0,
              entry.emissiveIntensity ?? 0,
            )
          }
        }
      } else {
        // Basic material
        gl.uniform3fv(gl.getUniformLocation(program, 'u_baseColor'), mesh.material.color)
        gl.uniform1f(gl.getUniformLocation(program, 'u_opacity'), mesh.material.opacity)
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

    // ─── Bloom ─────────────────────────────────────────────────────
    if (this.bloomEnabled && this.bloomLevels > 0) {
      gl.disable(gl.DEPTH_TEST)
      gl.depthMask(false)

      // Downsample chain
      gl.useProgram(this.bloomDownsampleProgram)
      let srcTex = rt.resolvedEmissiveTex
      let srcW = rt.width
      let srcH = rt.height

      for (let i = 0; i < this.bloomLevels; i++) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, rt.bloomFbos[i]!)
        gl.viewport(0, 0, rt.bloomWidths[i]!, rt.bloomHeights[i]!)

        gl.activeTexture(gl.TEXTURE0)
        gl.bindTexture(gl.TEXTURE_2D, srcTex)
        gl.uniform1i(gl.getUniformLocation(this.bloomDownsampleProgram, 'u_srcTexture'), 0)
        gl.uniform2f(gl.getUniformLocation(this.bloomDownsampleProgram, 'u_texelSize'), 1 / srcW, 1 / srcH)
        gl.uniform1i(gl.getUniformLocation(this.bloomDownsampleProgram, 'u_useKarisAverage'), i === 0 ? 1 : 0)

        gl.drawArrays(gl.TRIANGLES, 0, 3)

        srcTex = rt.bloomTextures[i]!
        srcW = rt.bloomWidths[i]!
        srcH = rt.bloomHeights[i]!
      }

      // Upsample chain (additive)
      gl.useProgram(this.bloomUpsampleProgram)
      gl.enable(gl.BLEND)
      gl.blendFunc(gl.ONE, gl.ONE)

      for (let i = this.bloomLevels - 1; i > 0; i--) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, rt.bloomFbos[i - 1]!)
        gl.viewport(0, 0, rt.bloomWidths[i - 1]!, rt.bloomHeights[i - 1]!)

        gl.activeTexture(gl.TEXTURE0)
        gl.bindTexture(gl.TEXTURE_2D, rt.bloomTextures[i]!)
        gl.uniform1i(gl.getUniformLocation(this.bloomUpsampleProgram, 'u_srcTexture'), 0)
        gl.uniform2f(
          gl.getUniformLocation(this.bloomUpsampleProgram, 'u_texelSize'),
          1 / rt.bloomWidths[i]!,
          1 / rt.bloomHeights[i]!,
        )

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

    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, rt.resolvedColorTex)
    gl.uniform1i(gl.getUniformLocation(this.blitProgram, 'u_sceneTexture'), 0)

    gl.activeTexture(gl.TEXTURE1)
    if (this.bloomEnabled && this.bloomLevels > 0) {
      gl.bindTexture(gl.TEXTURE_2D, rt.bloomTextures[0]!)
    } else {
      // Bind a dummy (resolved color serves as zero bloom)
      gl.bindTexture(gl.TEXTURE_2D, rt.resolvedEmissiveTex)
    }
    gl.uniform1i(gl.getUniformLocation(this.blitProgram, 'u_bloomTexture'), 1)
    gl.uniform1f(gl.getUniformLocation(this.blitProgram, 'u_bloomIntensity'), this.bloomIntensity)

    gl.drawArrays(gl.TRIANGLES, 0, 3)

    // Re-enable depth for next frame
    gl.enable(gl.DEPTH_TEST)
    gl.depthMask(true)

    // Update stats
    this.stats.fps = this._currentFps
    this.stats.frameTime = dt
    this.stats.drawCalls = drawCalls
    this.stats.triangles = triangles
    this.stats.visibleObjects = meshes.length
    this.stats.culledObjects = culledCount
  }

  dispose() {
    this.destroyRenderTargets()
    const gl = this.gl
    gl.deleteProgram(this.lambertProgram)
    gl.deleteProgram(this.basicProgram)
    gl.deleteProgram(this.lambertSkinnedProgram)
    gl.deleteProgram(this.basicSkinnedProgram)
    gl.deleteProgram(this.bloomDownsampleProgram)
    gl.deleteProgram(this.bloomUpsampleProgram)
    gl.deleteProgram(this.blitProgram)
  }
}
