import { vertexShaderGLSL, fragmentShaderGLSL } from './webgl-shaders.ts'

import type { DrawEntity, Renderer } from './renderer.ts'

interface GLGeometry {
  vao: WebGLVertexArrayObject
  indexCount: number
  indexType: GLenum
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

export function createWebGLRenderer(canvas: HTMLCanvasElement): Renderer {
  const gl = canvas.getContext('webgl2', { antialias: true })!
  if (!gl) throw new Error('WebGL2 not supported')

  gl.enable(gl.DEPTH_TEST)
  gl.depthFunc(gl.LESS)
  gl.enable(gl.CULL_FACE)
  gl.cullFace(gl.BACK)
  gl.clearColor(0.1, 0.1, 0.1, 1.0)

  const program = createProgram(gl, vertexShaderGLSL, fragmentShaderGLSL)

  // Bind UBO block indices
  const cameraBlockIdx = gl.getUniformBlockIndex(program, 'CameraUniforms')
  const modelBlockIdx = gl.getUniformBlockIndex(program, 'ModelUniforms')
  const lightBlockIdx = gl.getUniformBlockIndex(program, 'LightUniforms')
  gl.uniformBlockBinding(program, cameraBlockIdx, 0)
  gl.uniformBlockBinding(program, modelBlockIdx, 1)
  gl.uniformBlockBinding(program, lightBlockIdx, 2)

  // Camera UBO (2 × mat4 = 128 bytes)
  const cameraUBO = gl.createBuffer()!
  gl.bindBuffer(gl.UNIFORM_BUFFER, cameraUBO)
  gl.bufferData(gl.UNIFORM_BUFFER, 128, gl.DYNAMIC_DRAW)

  // Model UBO (mat4 + vec4 + vec4 = 96 bytes, pad to 112 for std140)
  // std140: mat4(64) + vec4(16) + vec4(16) = 96 bytes
  const MODEL_UBO_SIZE = 96
  const modelUBO = gl.createBuffer()!
  gl.bindBuffer(gl.UNIFORM_BUFFER, modelUBO)
  gl.bufferData(gl.UNIFORM_BUFFER, MODEL_UBO_SIZE, gl.DYNAMIC_DRAW)

  // Light UBO (3 × vec4 = 48 bytes)
  const lightUBO = gl.createBuffer()!
  gl.bindBuffer(gl.UNIFORM_BUFFER, lightUBO)
  gl.bufferData(gl.UNIFORM_BUFFER, 48, gl.DYNAMIC_DRAW)

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
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0)
      // normal
      gl.enableVertexAttribArray(1)
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12)

      const ebo = gl.createBuffer()!
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo)
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW)

      gl.bindVertexArray(null)

      geometries.set(id, {
        vao,
        indexCount: indices.length,
        indexType: indices instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT,
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
      gl.useProgram(program)

      gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, cameraUBO)
      gl.bindBufferBase(gl.UNIFORM_BUFFER, 2, lightUBO)

      for (let i = 0; i < count; i++) {
        const entity = entities[i]!
        const geo = geometries.get(entity.geometryId)
        if (!geo) continue

        // Update model UBO
        modelData.set(entity.worldMatrix, 0) // mat4
        modelData.set(entity.color, 16) // vec4
        modelData[20] = entity.unlit ? 1.0 : 0.0 // flags.x
        gl.bindBuffer(gl.UNIFORM_BUFFER, modelUBO)
        gl.bufferSubData(gl.UNIFORM_BUFFER, 0, modelData)
        gl.bindBufferBase(gl.UNIFORM_BUFFER, 1, modelUBO)

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
      gl.deleteProgram(program)
      for (const geo of geometries.values()) {
        gl.deleteVertexArray(geo.vao)
      }
    },
  }

  return renderer
}
