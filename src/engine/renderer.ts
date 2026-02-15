import { shaderSource } from './shaders.ts'

import type { Backend } from './gpu.ts'

export interface DrawEntity {
  worldMatrix: Float32Array
  color: Float32Array
  geometryId: number
  unlit: boolean
}

export interface Renderer {
  backend: Backend
  registerGeometry(id: number, vertices: Float32Array, indices: Uint16Array | Uint32Array): void
  updateCamera(view: Float32Array, projection: Float32Array): void
  updateLighting(dir: Float32Array, dirColor: Float32Array, ambient: Float32Array): void
  draw(entities: DrawEntity[], count: number): void
  resize(width: number, height: number): void
  destroy(): void
}

// 256-byte aligned model uniform size (mat4 + vec4 + vec4 = 96 bytes, aligned to 256)
const MODEL_UNIFORM_SIZE = 256
const MAX_ENTITIES = 4096

interface GeometryBuffers {
  vertexBuffer: GPUBuffer
  indexBuffer: GPUBuffer
  indexCount: number
  indexFormat: GPUIndexFormat
}

export async function createWebGPURenderer(canvas: HTMLCanvasElement): Promise<Renderer> {
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error('No WebGPU adapter')
  const device = await adapter.requestDevice()

  const context = canvas.getContext('webgpu')!
  const format = navigator.gpu.getPreferredCanvasFormat()
  context.configure({ device, format, alphaMode: 'premultiplied' })

  // Shader module
  const shaderModule = device.createShaderModule({
    code: shaderSource,
  })

  // Bind group layouts
  const cameraBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'uniform' },
      },
    ],
  })

  const modelBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'uniform', hasDynamicOffset: true },
      },
    ],
  })

  const lightBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      },
    ],
  })

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [cameraBindGroupLayout, modelBindGroupLayout, lightBindGroupLayout],
  })

  // Depth texture
  let depthTexture = device.createTexture({
    size: [canvas.width || 1, canvas.height || 1],
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
    sampleCount: 4,
  })

  // MSAA texture
  let msaaTexture = device.createTexture({
    size: [canvas.width || 1, canvas.height || 1],
    format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
    sampleCount: 4,
  })

  // Pipeline
  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: 'vs_main',
      buffers: [
        {
          arrayStride: 24, // 6 floats × 4 bytes
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' }, // position
            { shaderLocation: 1, offset: 12, format: 'float32x3' }, // normal
          ],
        },
      ],
    },
    fragment: {
      module: shaderModule,
      entryPoint: 'fs_main',
      targets: [{ format }],
    },
    primitive: {
      topology: 'triangle-list',
      cullMode: 'back',
    },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: 'less',
      format: 'depth24plus',
    },
    multisample: { count: 4 },
  })

  // Camera uniform buffer (2 × mat4 = 128 bytes)
  const cameraBuffer = device.createBuffer({
    size: 128,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  const cameraBindGroup = device.createBindGroup({
    layout: cameraBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: cameraBuffer } }],
  })

  // Model uniform buffer (dynamic offsets)
  const modelBuffer = device.createBuffer({
    size: MODEL_UNIFORM_SIZE * MAX_ENTITIES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  const modelBindGroup = device.createBindGroup({
    layout: modelBindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: { buffer: modelBuffer, size: MODEL_UNIFORM_SIZE },
      },
    ],
  })

  // Light uniform buffer (3 × vec4 = 48 bytes)
  const lightBuffer = device.createBuffer({
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  const lightBindGroup = device.createBindGroup({
    layout: lightBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: lightBuffer } }],
  })

  // Temp buffer for model uniforms
  const modelData = new Float32Array((MODEL_UNIFORM_SIZE / 4) * MAX_ENTITIES)

  const geometries = new Map<number, GeometryBuffers>()

  const renderer: Renderer = {
    backend: 'webgpu',

    registerGeometry(id, vertices, indices) {
      const vertexBuffer = device.createBuffer({
        size: vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(vertexBuffer, 0, new Float32Array(vertices))

      const indexBuffer = device.createBuffer({
        size: indices.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(
        indexBuffer,
        0,
        indices instanceof Uint16Array ? new Uint16Array(indices) : new Uint32Array(indices),
      )

      geometries.set(id, {
        vertexBuffer,
        indexBuffer,
        indexCount: indices.length,
        indexFormat: indices instanceof Uint32Array ? 'uint32' : 'uint16',
      })
    },

    updateCamera(view, projection) {
      device.queue.writeBuffer(cameraBuffer, 0, new Float32Array(view))
      device.queue.writeBuffer(cameraBuffer, 64, new Float32Array(projection))
    },

    updateLighting(dir, dirColor, ambient) {
      // Pad to vec4 (16 bytes each)
      const data = new Float32Array(12)
      data.set(dir.subarray(0, 3), 0)
      data.set(dirColor.subarray(0, 3), 4)
      data.set(ambient.subarray(0, 3), 8)
      device.queue.writeBuffer(lightBuffer, 0, data)
    },

    draw(entities, count) {
      const commandEncoder = device.createCommandEncoder()

      const colorView = msaaTexture.createView()
      const resolveTarget = context.getCurrentTexture().createView()

      const renderPass = commandEncoder.beginRenderPass({
        colorAttachments: [
          {
            view: colorView,
            resolveTarget,
            clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1 },
            loadOp: 'clear',
            storeOp: 'discard',
          },
        ],
        depthStencilAttachment: {
          view: depthTexture.createView(),
          depthClearValue: 1.0,
          depthLoadOp: 'clear',
          depthStoreOp: 'discard',
        },
      })

      renderPass.setPipeline(pipeline)
      renderPass.setBindGroup(0, cameraBindGroup)
      renderPass.setBindGroup(2, lightBindGroup)

      // Pack model uniforms
      for (let i = 0; i < count; i++) {
        const entity = entities[i]!
        const base = (MODEL_UNIFORM_SIZE / 4) * i
        modelData.set(entity.worldMatrix, base) // mat4 at offset 0
        modelData.set(entity.color, base + 16) // vec4 at offset 64
        modelData[base + 20] = entity.unlit ? 1.0 : 0.0 // flags.x at offset 80
      }
      device.queue.writeBuffer(modelBuffer, 0, modelData.buffer, 0, MODEL_UNIFORM_SIZE * count)

      for (let i = 0; i < count; i++) {
        const entity = entities[i]!
        const geo = geometries.get(entity.geometryId)
        if (!geo) continue

        renderPass.setBindGroup(1, modelBindGroup, [MODEL_UNIFORM_SIZE * i])
        renderPass.setVertexBuffer(0, geo.vertexBuffer)
        renderPass.setIndexBuffer(geo.indexBuffer, geo.indexFormat)
        renderPass.drawIndexed(geo.indexCount)
      }

      renderPass.end()
      device.queue.submit([commandEncoder.finish()])
    },

    resize(width, height) {
      if (width === 0 || height === 0) return
      depthTexture.destroy()
      msaaTexture.destroy()
      depthTexture = device.createTexture({
        size: [width, height],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
        sampleCount: 4,
      })
      msaaTexture = device.createTexture({
        size: [width, height],
        format,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
        sampleCount: 4,
      })
    },

    destroy() {
      cameraBuffer.destroy()
      modelBuffer.destroy()
      lightBuffer.destroy()
      depthTexture.destroy()
      msaaTexture.destroy()
      for (const geo of geometries.values()) {
        geo.vertexBuffer.destroy()
        geo.indexBuffer.destroy()
      }
      device.destroy()
    },
  }

  return renderer
}
