import { vertexShaderGLSL, fragmentShaderGLSL, skinnedVertexShaderGLSL } from './webgl-shaders.ts'

import type { DrawEntity, Renderer } from './renderer.ts'

interface GLGeometry {
  vao: WebGLVertexArrayObject
  indexCount: number
  indexType: GLenum
  skinned: boolean
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

  const renderer: Renderer = {
    backend: 'webgl',

    registerGeometry(id, vertices, indices) {
      const vao = gl.createVertexArray()!
      gl.bindVertexArray(vao)

      const vbo = gl.createBuffer()!
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo)
      gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW)

      // position
      gl.enableVertexAttribArray(0)
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 36, 0)
      // normal
      gl.enableVertexAttribArray(1)
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 36, 12)
      // vertex color
      gl.enableVertexAttribArray(2)
      gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 36, 24)

      const ebo = gl.createBuffer()!
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo)
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW)

      gl.bindVertexArray(null)

      geometries.set(id, {
        vao,
        indexCount: indices.length,
        indexType: indices instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT,
        skinned: false,
      })
    },

    registerSkinnedGeometry(id, vertices, indices, joints, weights) {
      const vao = gl.createVertexArray()!
      gl.bindVertexArray(vao)

      // Buffer 0: geometry [pos, normal, vertColor] = 36 bytes
      const vbo = gl.createBuffer()!
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo)
      gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW)

      gl.enableVertexAttribArray(0)
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 36, 0)
      gl.enableVertexAttribArray(1)
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 36, 12)
      gl.enableVertexAttribArray(2)
      gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 36, 24)

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
      gl.enableVertexAttribArray(3)
      gl.vertexAttribIPointer(3, 4, gl.UNSIGNED_BYTE, 20, 0)
      // weights as float attribute (vec4)
      gl.enableVertexAttribArray(4)
      gl.vertexAttribPointer(4, 4, gl.FLOAT, false, 20, 4)

      const ebo = gl.createBuffer()!
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo)
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW)

      gl.bindVertexArray(null)

      geometries.set(id, {
        vao,
        indexCount: indices.length,
        indexType: indices instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT,
        skinned: true,
      })
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
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

      gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, cameraUBO)
      gl.bindBufferBase(gl.UNIFORM_BUFFER, 2, lightUBO)

      // Draw non-skinned entities
      gl.useProgram(program)
      for (let i = 0; i < count; i++) {
        const entity = entities[i]!
        const geo = geometries.get(entity.geometryId)
        if (!geo || geo.skinned) continue

        modelData.set(entity.worldMatrix, 0)
        modelData.set(entity.color, 16)
        modelData[20] = entity.unlit ? 1.0 : 0.0
        gl.bindBuffer(gl.UNIFORM_BUFFER, modelUBO)
        gl.bufferSubData(gl.UNIFORM_BUFFER, 0, modelData)
        gl.bindBufferBase(gl.UNIFORM_BUFFER, 1, modelUBO)

        gl.bindVertexArray(geo.vao)
        gl.drawElements(gl.TRIANGLES, geo.indexCount, geo.indexType, 0)
      }

      // Draw skinned entities
      gl.useProgram(skinnedProgram)
      gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, cameraUBO)
      gl.bindBufferBase(gl.UNIFORM_BUFFER, 2, lightUBO)

      for (let i = 0; i < count; i++) {
        const entity = entities[i]!
        const geo = geometries.get(entity.geometryId)
        if (!geo?.skinned || !entity.jointMatrices) continue

        modelData.set(entity.worldMatrix, 0)
        modelData.set(entity.color, 16)
        modelData[20] = entity.unlit ? 1.0 : 0.0
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

      gl.bindVertexArray(null)
    },

    resize(width, height) {
      if (width === 0 || height === 0) return
      gl.viewport(0, 0, width, height)
    },

    destroy() {
      gl.deleteBuffer(cameraUBO)
      gl.deleteBuffer(modelUBO)
      gl.deleteBuffer(lightUBO)
      gl.deleteBuffer(jointUBO)
      gl.deleteProgram(program)
      gl.deleteProgram(skinnedProgram)
      for (const geo of geometries.values()) {
        gl.deleteVertexArray(geo.vao)
      }
    },
  }

  return renderer
}
