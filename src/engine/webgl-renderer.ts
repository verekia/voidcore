import {
  vertexShaderGLSL,
  fragmentShaderGLSL,
  skinnedVertexShaderGLSL,
  texturedVertexShaderGLSL,
  texturedFragmentShaderGLSL,
  texturedMrtFragmentShaderGLSL,
  mrtFragmentShaderGLSL,
  fullscreenVertexGLSL,
  downsampleFragmentGLSL,
  upsampleFragmentGLSL,
  compositeFragmentGLSL,
} from './webgl-shaders.ts'

import type { BloomConfig, DrawEntity, Renderer } from './renderer.ts'

interface GLGeometry {
  vao: WebGLVertexArrayObject
  indexCount: number
  indexType: GLenum
  skinned: boolean
  textured?: boolean
}

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)!
  gl.shaderSource(shader, source)
  gl.compileShader(shader)
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader)
    gl.deleteShader(shader)
    throw new Error(`Shader compile error: ${info}`)
  }
  return shader
}

function createProgram(gl: WebGL2RenderingContext, vsSource: string, fsSource: string): WebGLProgram {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSource)
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSource)
  const program = gl.createProgram()!
  gl.attachShader(program, vs)
  gl.attachShader(program, fs)
  gl.linkProgram(program)
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program)
    throw new Error(`Program link error: ${info}`)
  }
  gl.deleteShader(vs)
  gl.deleteShader(fs)
  return program
}

const MAX_JOINTS = 128
const JOINT_UBO_SIZE = MAX_JOINTS * 64 // 128 mat4 × 64 bytes each = 8192 bytes
const BLOOM_MIP_LEVELS = 5

export function createWebGLRenderer(canvas: HTMLCanvasElement): Renderer {
  const gl = canvas.getContext('webgl2', { antialias: true })!
  if (!gl) throw new Error('WebGL2 not supported')

  gl.enable(gl.DEPTH_TEST)
  gl.depthFunc(gl.LESS)
  gl.enable(gl.CULL_FACE)
  gl.cullFace(gl.BACK)
  gl.clearColor(0.1, 0.1, 0.1, 1.0)

  // Non-skinned program
  const program = createProgram(gl, vertexShaderGLSL, fragmentShaderGLSL)

  const cameraBlockIdx = gl.getUniformBlockIndex(program, 'CameraUniforms')
  const modelBlockIdx = gl.getUniformBlockIndex(program, 'ModelUniforms')
  const lightBlockIdx = gl.getUniformBlockIndex(program, 'LightUniforms')
  gl.uniformBlockBinding(program, cameraBlockIdx, 0)
  gl.uniformBlockBinding(program, modelBlockIdx, 1)
  gl.uniformBlockBinding(program, lightBlockIdx, 2)

  // Skinned program
  const skinnedProgram = createProgram(gl, skinnedVertexShaderGLSL, fragmentShaderGLSL)

  const skCameraBlockIdx = gl.getUniformBlockIndex(skinnedProgram, 'CameraUniforms')
  const skModelBlockIdx = gl.getUniformBlockIndex(skinnedProgram, 'ModelUniforms')
  const skLightBlockIdx = gl.getUniformBlockIndex(skinnedProgram, 'LightUniforms')
  const skJointBlockIdx = gl.getUniformBlockIndex(skinnedProgram, 'JointUniforms')
  gl.uniformBlockBinding(skinnedProgram, skCameraBlockIdx, 0)
  gl.uniformBlockBinding(skinnedProgram, skModelBlockIdx, 1)
  gl.uniformBlockBinding(skinnedProgram, skLightBlockIdx, 2)
  gl.uniformBlockBinding(skinnedProgram, skJointBlockIdx, 3)

  // Textured program
  const texturedProgram = createProgram(gl, texturedVertexShaderGLSL, texturedFragmentShaderGLSL)
  const txCameraBlockIdx = gl.getUniformBlockIndex(texturedProgram, 'CameraUniforms')
  const txModelBlockIdx = gl.getUniformBlockIndex(texturedProgram, 'ModelUniforms')
  const txLightBlockIdx = gl.getUniformBlockIndex(texturedProgram, 'LightUniforms')
  gl.uniformBlockBinding(texturedProgram, txCameraBlockIdx, 0)
  gl.uniformBlockBinding(texturedProgram, txModelBlockIdx, 1)
  gl.uniformBlockBinding(texturedProgram, txLightBlockIdx, 2)
  const uAoMapLoc = gl.getUniformLocation(texturedProgram, 'u_aoMap')

  // Texture storage
  const glTextures = new Map<number, WebGLTexture>()

  // Camera UBO (2 × mat4 = 128 bytes)
  const cameraUBO = gl.createBuffer()!
  gl.bindBuffer(gl.UNIFORM_BUFFER, cameraUBO)
  gl.bufferData(gl.UNIFORM_BUFFER, 128, gl.DYNAMIC_DRAW)

  // Model UBO (mat4 + vec4 + vec4 = 96 bytes)
  const MODEL_UBO_SIZE = 96
  const modelUBO = gl.createBuffer()!
  gl.bindBuffer(gl.UNIFORM_BUFFER, modelUBO)
  gl.bufferData(gl.UNIFORM_BUFFER, MODEL_UBO_SIZE, gl.DYNAMIC_DRAW)

  // Light UBO (3 × vec4 = 48 bytes)
  const lightUBO = gl.createBuffer()!
  gl.bindBuffer(gl.UNIFORM_BUFFER, lightUBO)
  gl.bufferData(gl.UNIFORM_BUFFER, 48, gl.DYNAMIC_DRAW)

  // Joint UBO (128 × mat4 = 8192 bytes)
  const jointUBO = gl.createBuffer()!
  gl.bindBuffer(gl.UNIFORM_BUFFER, jointUBO)
  gl.bufferData(gl.UNIFORM_BUFFER, JOINT_UBO_SIZE, gl.DYNAMIC_DRAW)

  const geometries = new Map<number, GLGeometry>()
  const modelData = new Float32Array(24) // mat4(16) + vec4(4) + vec4(4)

  // Fullscreen triangle VAO (no attributes, vertex_index based)
  const fullscreenVAO = gl.createVertexArray()!

  // ── Bloom state ──────────────────────────────────────────────────
  let bloomEnabled = false
  let bloomIntensity = 1.0
  let bloomRadius = 1.0

  // MRT programs (created lazily)
  let mrtProgram: WebGLProgram | null = null
  let mrtSkinnedProgram: WebGLProgram | null = null
  let mrtTexturedProgram: WebGLProgram | null = null
  let uMrtAoMapLoc: WebGLUniformLocation | null = null

  // Post-processing programs
  let downsampleProgram: WebGLProgram | null = null
  let upsampleProgram: WebGLProgram | null = null
  let compositeProgram: WebGLProgram | null = null

  // Bloom FBOs and textures
  let mrtFBO: WebGLFramebuffer | null = null
  let mrtColorRB: WebGLRenderbuffer | null = null
  let mrtBloomRB: WebGLRenderbuffer | null = null
  let mrtDepthRB: WebGLRenderbuffer | null = null

  let resolveSceneFBO: WebGLFramebuffer | null = null
  let resolveBloomFBO: WebGLFramebuffer | null = null
  let resolvedSceneTexture: WebGLTexture | null = null
  let resolvedBloomTexture: WebGLTexture | null = null

  let bloomMipFBOs: WebGLFramebuffer[] = []
  let bloomMipTextures: WebGLTexture[] = []

  // Uniform locations (cached)
  let uDownsampleInput: WebGLUniformLocation | null = null
  let uUpsampleInput: WebGLUniformLocation | null = null
  let uUpsampleRadius: WebGLUniformLocation | null = null
  let uCompositeScene: WebGLUniformLocation | null = null
  let uCompositeBloom: WebGLUniformLocation | null = null
  let uCompositeIntensity: WebGLUniformLocation | null = null

  let currentWidth = canvas.width || 1
  let currentHeight = canvas.height || 1

  function setupVAO(
    vao: WebGLVertexArrayObject,
    vertices: Float32Array,
    indices: Uint16Array | Uint32Array,
    skinned: boolean,
    joints?: Uint8Array,
    weights?: Float32Array,
  ) {
    gl.bindVertexArray(vao)

    // Buffer 0: geometry [pos, normal, vertColor, bloom] = 40 bytes
    const vbo = gl.createBuffer()!
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo)
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW)

    gl.enableVertexAttribArray(0)
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 40, 0)
    gl.enableVertexAttribArray(1)
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 40, 12)
    gl.enableVertexAttribArray(2)
    gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 40, 24)
    gl.enableVertexAttribArray(3)
    gl.vertexAttribPointer(3, 1, gl.FLOAT, false, 40, 36)

    if (skinned && joints && weights) {
      // Buffer 1: skin data [joints u8x4, weights f32x4] = 20 bytes
      const vertexCount = joints.length / 4
      const skinBuf = new ArrayBuffer(vertexCount * 20)
      const skinView = new DataView(skinBuf)
      for (let i = 0; i < vertexCount; i++) {
        const o = i * 20
        skinView.setUint8(o, joints[i * 4]!)
        skinView.setUint8(o + 1, joints[i * 4 + 1]!)
        skinView.setUint8(o + 2, joints[i * 4 + 2]!)
        skinView.setUint8(o + 3, joints[i * 4 + 3]!)
        skinView.setFloat32(o + 4, weights[i * 4]!, true)
        skinView.setFloat32(o + 8, weights[i * 4 + 1]!, true)
        skinView.setFloat32(o + 12, weights[i * 4 + 2]!, true)
        skinView.setFloat32(o + 16, weights[i * 4 + 3]!, true)
      }

      const skinVBO = gl.createBuffer()!
      gl.bindBuffer(gl.ARRAY_BUFFER, skinVBO)
      gl.bufferData(gl.ARRAY_BUFFER, skinBuf, gl.STATIC_DRAW)

      // joints as integer attribute (uvec4)
      gl.enableVertexAttribArray(4)
      gl.vertexAttribIPointer(4, 4, gl.UNSIGNED_BYTE, 20, 0)
      // weights as float attribute (vec4)
      gl.enableVertexAttribArray(5)
      gl.vertexAttribPointer(5, 4, gl.FLOAT, false, 20, 4)
    }

    const ebo = gl.createBuffer()!
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo)
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW)

    gl.bindVertexArray(null)
  }

  function ensureBloomPrograms() {
    if (mrtProgram) return

    // MRT programs (use MRT fragment shader)
    mrtProgram = createProgram(gl, vertexShaderGLSL, mrtFragmentShaderGLSL)
    const mrtCameraBlockIdx = gl.getUniformBlockIndex(mrtProgram, 'CameraUniforms')
    const mrtModelBlockIdx = gl.getUniformBlockIndex(mrtProgram, 'ModelUniforms')
    const mrtLightBlockIdx = gl.getUniformBlockIndex(mrtProgram, 'LightUniforms')
    gl.uniformBlockBinding(mrtProgram, mrtCameraBlockIdx, 0)
    gl.uniformBlockBinding(mrtProgram, mrtModelBlockIdx, 1)
    gl.uniformBlockBinding(mrtProgram, mrtLightBlockIdx, 2)

    mrtSkinnedProgram = createProgram(gl, skinnedVertexShaderGLSL, mrtFragmentShaderGLSL)
    const mrtSkCameraBlockIdx = gl.getUniformBlockIndex(mrtSkinnedProgram, 'CameraUniforms')
    const mrtSkModelBlockIdx = gl.getUniformBlockIndex(mrtSkinnedProgram, 'ModelUniforms')
    const mrtSkLightBlockIdx = gl.getUniformBlockIndex(mrtSkinnedProgram, 'LightUniforms')
    const mrtSkJointBlockIdx = gl.getUniformBlockIndex(mrtSkinnedProgram, 'JointUniforms')
    gl.uniformBlockBinding(mrtSkinnedProgram, mrtSkCameraBlockIdx, 0)
    gl.uniformBlockBinding(mrtSkinnedProgram, mrtSkModelBlockIdx, 1)
    gl.uniformBlockBinding(mrtSkinnedProgram, mrtSkLightBlockIdx, 2)
    gl.uniformBlockBinding(mrtSkinnedProgram, mrtSkJointBlockIdx, 3)

    mrtTexturedProgram = createProgram(gl, texturedVertexShaderGLSL, texturedMrtFragmentShaderGLSL)
    const mrtTxCameraBlockIdx = gl.getUniformBlockIndex(mrtTexturedProgram, 'CameraUniforms')
    const mrtTxModelBlockIdx = gl.getUniformBlockIndex(mrtTexturedProgram, 'ModelUniforms')
    const mrtTxLightBlockIdx = gl.getUniformBlockIndex(mrtTexturedProgram, 'LightUniforms')
    gl.uniformBlockBinding(mrtTexturedProgram, mrtTxCameraBlockIdx, 0)
    gl.uniformBlockBinding(mrtTexturedProgram, mrtTxModelBlockIdx, 1)
    gl.uniformBlockBinding(mrtTexturedProgram, mrtTxLightBlockIdx, 2)
    uMrtAoMapLoc = gl.getUniformLocation(mrtTexturedProgram, 'u_aoMap')

    // Post-processing programs
    downsampleProgram = createProgram(gl, fullscreenVertexGLSL, downsampleFragmentGLSL)
    uDownsampleInput = gl.getUniformLocation(downsampleProgram, 'u_input')

    upsampleProgram = createProgram(gl, fullscreenVertexGLSL, upsampleFragmentGLSL)
    uUpsampleInput = gl.getUniformLocation(upsampleProgram, 'u_input')
    uUpsampleRadius = gl.getUniformLocation(upsampleProgram, 'u_radius')

    compositeProgram = createProgram(gl, fullscreenVertexGLSL, compositeFragmentGLSL)
    uCompositeScene = gl.getUniformLocation(compositeProgram, 'u_scene')
    uCompositeBloom = gl.getUniformLocation(compositeProgram, 'u_bloom')
    uCompositeIntensity = gl.getUniformLocation(compositeProgram, 'u_intensity')
  }

  function createBloomFBOs() {
    const w = currentWidth
    const h = currentHeight
    const samples = gl.getParameter(gl.MAX_SAMPLES) as number
    const msaaSamples = Math.min(samples, 4)

    // Clean up old resources
    if (mrtFBO) gl.deleteFramebuffer(mrtFBO)
    if (mrtColorRB) gl.deleteRenderbuffer(mrtColorRB)
    if (mrtBloomRB) gl.deleteRenderbuffer(mrtBloomRB)
    if (mrtDepthRB) gl.deleteRenderbuffer(mrtDepthRB)
    if (resolveSceneFBO) gl.deleteFramebuffer(resolveSceneFBO)
    if (resolveBloomFBO) gl.deleteFramebuffer(resolveBloomFBO)
    if (resolvedSceneTexture) gl.deleteTexture(resolvedSceneTexture)
    if (resolvedBloomTexture) gl.deleteTexture(resolvedBloomTexture)
    for (const fbo of bloomMipFBOs) gl.deleteFramebuffer(fbo)
    for (const tex of bloomMipTextures) gl.deleteTexture(tex)

    // MSAA MRT FBO with 2 color renderbuffers + depth
    mrtFBO = gl.createFramebuffer()!
    gl.bindFramebuffer(gl.FRAMEBUFFER, mrtFBO)

    mrtColorRB = gl.createRenderbuffer()!
    gl.bindRenderbuffer(gl.RENDERBUFFER, mrtColorRB)
    gl.renderbufferStorageMultisample(gl.RENDERBUFFER, msaaSamples, gl.RGBA8, w, h)
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, mrtColorRB)

    mrtBloomRB = gl.createRenderbuffer()!
    gl.bindRenderbuffer(gl.RENDERBUFFER, mrtBloomRB)
    gl.renderbufferStorageMultisample(gl.RENDERBUFFER, msaaSamples, gl.RGBA8, w, h)
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.RENDERBUFFER, mrtBloomRB)

    mrtDepthRB = gl.createRenderbuffer()!
    gl.bindRenderbuffer(gl.RENDERBUFFER, mrtDepthRB)
    gl.renderbufferStorageMultisample(gl.RENDERBUFFER, msaaSamples, gl.DEPTH_COMPONENT24, w, h)
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, mrtDepthRB)

    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1])

    // Resolve FBOs with textures
    resolvedSceneTexture = createTexture2D(gl, w, h)
    resolveSceneFBO = gl.createFramebuffer()!
    gl.bindFramebuffer(gl.FRAMEBUFFER, resolveSceneFBO)
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, resolvedSceneTexture, 0)

    resolvedBloomTexture = createTexture2D(gl, w, h)
    resolveBloomFBO = gl.createFramebuffer()!
    gl.bindFramebuffer(gl.FRAMEBUFFER, resolveBloomFBO)
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, resolvedBloomTexture, 0)

    // Mip chain for bloom blur
    bloomMipFBOs = []
    bloomMipTextures = []
    let mw = Math.max(1, w >> 1)
    let mh = Math.max(1, h >> 1)
    for (let i = 0; i < BLOOM_MIP_LEVELS; i++) {
      const tex = createTexture2D(gl, mw, mh)
      const fbo = gl.createFramebuffer()!
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo)
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0)
      bloomMipFBOs.push(fbo)
      bloomMipTextures.push(tex)
      mw = Math.max(1, mw >> 1)
      mh = Math.max(1, mh >> 1)
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
  }

  function drawScene(entities: DrawEntity[], count: number, useMrt: boolean) {
    gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, cameraUBO)
    gl.bindBufferBase(gl.UNIFORM_BUFFER, 2, lightUBO)

    // Draw non-skinned, non-textured entities
    gl.useProgram(useMrt ? mrtProgram : program)
    for (let i = 0; i < count; i++) {
      const entity = entities[i]!
      if (entity.isTextured) continue
      const geo = geometries.get(entity.geometryId)
      if (!geo || geo.skinned) continue

      modelData.set(entity.worldMatrix, 0)
      modelData.set(entity.color, 16)
      modelData[20] = entity.unlit ? 1.0 : 0.0
      modelData[21] = entity.aoIntensity ?? 0.0
      gl.bindBuffer(gl.UNIFORM_BUFFER, modelUBO)
      gl.bufferSubData(gl.UNIFORM_BUFFER, 0, modelData)
      gl.bindBufferBase(gl.UNIFORM_BUFFER, 1, modelUBO)

      gl.bindVertexArray(geo.vao)
      gl.drawElements(gl.TRIANGLES, geo.indexCount, geo.indexType, 0)
    }

    // Draw skinned entities
    gl.useProgram(useMrt ? mrtSkinnedProgram : skinnedProgram)
    gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, cameraUBO)
    gl.bindBufferBase(gl.UNIFORM_BUFFER, 2, lightUBO)

    for (let i = 0; i < count; i++) {
      const entity = entities[i]!
      const geo = geometries.get(entity.geometryId)
      if (!geo?.skinned || !entity.jointMatrices) continue

      modelData.set(entity.worldMatrix, 0)
      modelData.set(entity.color, 16)
      modelData[20] = entity.unlit ? 1.0 : 0.0
      modelData[21] = entity.aoIntensity ?? 0.0
      gl.bindBuffer(gl.UNIFORM_BUFFER, modelUBO)
      gl.bufferSubData(gl.UNIFORM_BUFFER, 0, modelData)
      gl.bindBufferBase(gl.UNIFORM_BUFFER, 1, modelUBO)

      // Upload joint matrices
      gl.bindBuffer(gl.UNIFORM_BUFFER, jointUBO)
      gl.bufferSubData(gl.UNIFORM_BUFFER, 0, entity.jointMatrices)
      gl.bindBufferBase(gl.UNIFORM_BUFFER, 3, jointUBO)

      gl.bindVertexArray(geo.vao)
      gl.drawElements(gl.TRIANGLES, geo.indexCount, geo.indexType, 0)
    }

    // Draw textured entities
    let texturedProgramBound = false
    for (let i = 0; i < count; i++) {
      const entity = entities[i]!
      if (!entity.isTextured || entity.textureId === undefined) continue
      const geo = geometries.get(entity.geometryId)
      const tex = glTextures.get(entity.textureId)
      if (!geo?.textured || !tex) continue

      if (!texturedProgramBound) {
        gl.useProgram(useMrt ? mrtTexturedProgram : texturedProgram)
        gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, cameraUBO)
        gl.bindBufferBase(gl.UNIFORM_BUFFER, 2, lightUBO)
        texturedProgramBound = true
      }

      modelData.set(entity.worldMatrix, 0)
      modelData.set(entity.color, 16)
      modelData[20] = entity.unlit ? 1.0 : 0.0
      modelData[21] = entity.aoIntensity ?? 0.0
      gl.bindBuffer(gl.UNIFORM_BUFFER, modelUBO)
      gl.bufferSubData(gl.UNIFORM_BUFFER, 0, modelData)
      gl.bindBufferBase(gl.UNIFORM_BUFFER, 1, modelUBO)

      gl.activeTexture(gl.TEXTURE0)
      gl.bindTexture(gl.TEXTURE_2D, tex)
      gl.uniform1i(useMrt ? uMrtAoMapLoc : uAoMapLoc, 0)

      gl.bindVertexArray(geo.vao)
      gl.drawElements(gl.TRIANGLES, geo.indexCount, geo.indexType, 0)
    }

    gl.bindVertexArray(null)
  }

  const renderer: Renderer = {
    backend: 'webgl',

    registerGeometry(id, vertices, indices) {
      const vao = gl.createVertexArray()!
      setupVAO(vao, vertices, indices, false)
      geometries.set(id, {
        vao,
        indexCount: indices.length,
        indexType: indices instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT,
        skinned: false,
      })
    },

    registerSkinnedGeometry(id, vertices, indices, joints, weights) {
      const vao = gl.createVertexArray()!
      setupVAO(vao, vertices, indices, true, joints, weights)
      geometries.set(id, {
        vao,
        indexCount: indices.length,
        indexType: indices instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT,
        skinned: true,
      })
    },

    registerTexturedGeometry(id, vertices, indices, uvs) {
      const vao = gl.createVertexArray()!
      gl.bindVertexArray(vao)

      // Buffer 0: geometry [pos, normal, vertColor, bloom] = 40 bytes
      const vbo = gl.createBuffer()!
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo)
      gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW)

      gl.enableVertexAttribArray(0)
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 40, 0)
      gl.enableVertexAttribArray(1)
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 40, 12)
      gl.enableVertexAttribArray(2)
      gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 40, 24)
      gl.enableVertexAttribArray(3)
      gl.vertexAttribPointer(3, 1, gl.FLOAT, false, 40, 36)

      // Buffer 1: UV [u, v] = 8 bytes
      const uvVBO = gl.createBuffer()!
      gl.bindBuffer(gl.ARRAY_BUFFER, uvVBO)
      gl.bufferData(gl.ARRAY_BUFFER, uvs, gl.STATIC_DRAW)
      gl.enableVertexAttribArray(4)
      gl.vertexAttribPointer(4, 2, gl.FLOAT, false, 8, 0)

      const ebo = gl.createBuffer()!
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo)
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW)

      gl.bindVertexArray(null)

      geometries.set(id, {
        vao,
        indexCount: indices.length,
        indexType: indices instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT,
        skinned: false,
        textured: true,
      })
    },

    registerTexture(id, data, width, height) {
      const tex = gl.createTexture()!
      gl.bindTexture(gl.TEXTURE_2D, tex)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, data)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
      glTextures.set(id, tex)
    },

    updateCamera(view, projection) {
      const data = new Float32Array(32)
      data.set(view, 0)
      data.set(projection, 16)
      gl.bindBuffer(gl.UNIFORM_BUFFER, cameraUBO)
      gl.bufferSubData(gl.UNIFORM_BUFFER, 0, data)
    },

    updateLighting(dir, dirColor, ambient) {
      const data = new Float32Array(12)
      data.set(dir.subarray(0, 3), 0)
      data.set(dirColor.subarray(0, 3), 4)
      data.set(ambient.subarray(0, 3), 8)
      gl.bindBuffer(gl.UNIFORM_BUFFER, lightUBO)
      gl.bufferSubData(gl.UNIFORM_BUFFER, 0, data)
    },

    draw(entities: DrawEntity[], count: number) {
      if (!bloomEnabled) {
        // ── Original non-bloom path ──
        gl.bindFramebuffer(gl.FRAMEBUFFER, null)
        gl.viewport(0, 0, currentWidth, currentHeight)
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
        drawScene(entities, count, false)
        return
      }

      // ── Bloom MRT path ──

      // 1. Render to MRT FBO
      gl.bindFramebuffer(gl.FRAMEBUFFER, mrtFBO)
      gl.viewport(0, 0, currentWidth, currentHeight)
      // Clear each attachment separately: scene to dark gray, bloom to black
      gl.clearBufferfv(gl.COLOR, 0, [0.1, 0.1, 0.1, 1.0])
      gl.clearBufferfv(gl.COLOR, 1, [0.0, 0.0, 0.0, 0.0])
      gl.clear(gl.DEPTH_BUFFER_BIT)
      drawScene(entities, count, true)

      // 2. Resolve MSAA → textures via blitFramebuffer
      // Resolve scene (attachment 0)
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, mrtFBO)
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, resolveSceneFBO)
      gl.readBuffer(gl.COLOR_ATTACHMENT0)
      gl.blitFramebuffer(
        0,
        0,
        currentWidth,
        currentHeight,
        0,
        0,
        currentWidth,
        currentHeight,
        gl.COLOR_BUFFER_BIT,
        gl.NEAREST,
      )

      // Resolve bloom (attachment 1)
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, mrtFBO)
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, resolveBloomFBO)
      gl.readBuffer(gl.COLOR_ATTACHMENT1)
      gl.blitFramebuffer(
        0,
        0,
        currentWidth,
        currentHeight,
        0,
        0,
        currentWidth,
        currentHeight,
        gl.COLOR_BUFFER_BIT,
        gl.NEAREST,
      )

      // Disable depth for post-processing
      gl.disable(gl.DEPTH_TEST)
      gl.disable(gl.CULL_FACE)

      // 3. Downsample bloom through mip chain
      gl.useProgram(downsampleProgram)
      gl.bindVertexArray(fullscreenVAO)

      let srcW = currentWidth
      let srcH = currentHeight
      for (let i = 0; i < BLOOM_MIP_LEVELS; i++) {
        const dstW = Math.max(1, srcW >> 1)
        const dstH = Math.max(1, srcH >> 1)
        gl.bindFramebuffer(gl.FRAMEBUFFER, bloomMipFBOs[i]!)
        gl.viewport(0, 0, dstW, dstH)

        gl.activeTexture(gl.TEXTURE0)
        gl.bindTexture(gl.TEXTURE_2D, i === 0 ? resolvedBloomTexture : bloomMipTextures[i - 1]!)
        gl.uniform1i(uDownsampleInput, 0)

        gl.drawArrays(gl.TRIANGLES, 0, 3)
        srcW = dstW
        srcH = dstH
      }

      // 4. Upsample bloom back up (additive blend)
      gl.useProgram(upsampleProgram)
      gl.enable(gl.BLEND)
      gl.blendFunc(gl.ONE, gl.ONE)

      for (let i = BLOOM_MIP_LEVELS - 1; i > 0; i--) {
        const targetIdx = i - 1
        const targetW = bloomMipTextures[targetIdx] ? getMipSize(currentWidth, targetIdx) : currentWidth
        const targetH = bloomMipTextures[targetIdx] ? getMipSize(currentHeight, targetIdx) : currentHeight

        gl.bindFramebuffer(gl.FRAMEBUFFER, bloomMipFBOs[targetIdx]!)
        gl.viewport(0, 0, targetW, targetH)

        gl.activeTexture(gl.TEXTURE0)
        gl.bindTexture(gl.TEXTURE_2D, bloomMipTextures[i]!)
        gl.uniform1i(uUpsampleInput, 0)
        gl.uniform1f(uUpsampleRadius, bloomRadius)

        gl.drawArrays(gl.TRIANGLES, 0, 3)
      }

      gl.disable(gl.BLEND)

      // 5. Composite scene + bloom → default framebuffer
      gl.useProgram(compositeProgram)
      gl.bindFramebuffer(gl.FRAMEBUFFER, null)
      gl.viewport(0, 0, currentWidth, currentHeight)

      gl.activeTexture(gl.TEXTURE0)
      gl.bindTexture(gl.TEXTURE_2D, resolvedSceneTexture)
      gl.uniform1i(uCompositeScene, 0)

      gl.activeTexture(gl.TEXTURE1)
      gl.bindTexture(gl.TEXTURE_2D, bloomMipTextures[0]!)
      gl.uniform1i(uCompositeBloom, 1)

      gl.uniform1f(uCompositeIntensity, bloomIntensity)

      gl.drawArrays(gl.TRIANGLES, 0, 3)

      gl.bindVertexArray(null)

      // Restore state
      gl.enable(gl.DEPTH_TEST)
      gl.enable(gl.CULL_FACE)
    },

    resize(width, height) {
      if (width === 0 || height === 0) return
      currentWidth = width
      currentHeight = height
      gl.viewport(0, 0, width, height)
      if (bloomEnabled) {
        createBloomFBOs()
      }
    },

    setBloom(config: BloomConfig) {
      bloomEnabled = config.enabled
      bloomIntensity = config.intensity ?? 1.0
      bloomRadius = config.radius ?? 1.0
      if (bloomEnabled) {
        ensureBloomPrograms()
        createBloomFBOs()
      }
    },

    destroy() {
      gl.deleteBuffer(cameraUBO)
      gl.deleteBuffer(modelUBO)
      gl.deleteBuffer(lightUBO)
      gl.deleteBuffer(jointUBO)
      gl.deleteProgram(program)
      gl.deleteProgram(skinnedProgram)
      gl.deleteProgram(texturedProgram)
      gl.deleteVertexArray(fullscreenVAO)
      if (mrtProgram) gl.deleteProgram(mrtProgram)
      if (mrtSkinnedProgram) gl.deleteProgram(mrtSkinnedProgram)
      if (mrtTexturedProgram) gl.deleteProgram(mrtTexturedProgram)
      if (downsampleProgram) gl.deleteProgram(downsampleProgram)
      if (upsampleProgram) gl.deleteProgram(upsampleProgram)
      if (compositeProgram) gl.deleteProgram(compositeProgram)
      if (mrtFBO) gl.deleteFramebuffer(mrtFBO)
      if (mrtColorRB) gl.deleteRenderbuffer(mrtColorRB)
      if (mrtBloomRB) gl.deleteRenderbuffer(mrtBloomRB)
      if (mrtDepthRB) gl.deleteRenderbuffer(mrtDepthRB)
      if (resolveSceneFBO) gl.deleteFramebuffer(resolveSceneFBO)
      if (resolveBloomFBO) gl.deleteFramebuffer(resolveBloomFBO)
      if (resolvedSceneTexture) gl.deleteTexture(resolvedSceneTexture)
      if (resolvedBloomTexture) gl.deleteTexture(resolvedBloomTexture)
      for (const fbo of bloomMipFBOs) gl.deleteFramebuffer(fbo)
      for (const tex of bloomMipTextures) gl.deleteTexture(tex)
      for (const geo of geometries.values()) {
        gl.deleteVertexArray(geo.vao)
      }
      for (const tex of glTextures.values()) {
        gl.deleteTexture(tex)
      }
    },
  }

  return renderer
}

function createTexture2D(gl: WebGL2RenderingContext, width: number, height: number): WebGLTexture {
  const tex = gl.createTexture()!
  gl.bindTexture(gl.TEXTURE_2D, tex)
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
  return tex
}

function getMipSize(baseSize: number, mipLevel: number): number {
  return Math.max(1, baseSize >> (mipLevel + 1))
}
