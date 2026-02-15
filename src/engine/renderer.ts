import { shaderSource, skinnedShaderSource } from './shaders.ts'

import type { Backend } from './gpu.ts'

export interface DrawEntity {
  worldMatrix: Float32Array
  color: Float32Array
  geometryId: number
  unlit: boolean
  jointMatrices?: Float32Array
}

export interface Renderer {
  backend: Backend
  registerGeometry(id: number, vertices: Float32Array, indices: Uint16Array | Uint32Array): void
  registerSkinnedGeometry(
    id: number,
    vertices: Float32Array,
    indices: Uint16Array | Uint32Array,
    joints: Uint8Array,
    weights: Float32Array,
  ): void
  updateCamera(view: Float32Array, projection: Float32Array): void
  updateLighting(dir: Float32Array, dirColor: Float32Array, ambient: Float32Array): void
  draw(entities: DrawEntity[], count: number): void
  resize(width: number, height: number): void
  destroy(): void
}

// 256-byte aligned model uniform size (mat4 + vec4 + vec4 = 96 bytes, aligned to 256)
const MODEL_UNIFORM_SIZE = 256
const MAX_ENTITIES = 4096

const MAX_JOINTS = 128
const JOINT_SLOT_BYTES = MAX_JOINTS * 16 * 4 // 128 joints × 16 floats × 4 bytes = 8192
const MAX_SKINNED_ENTITIES = 1024

interface GeometryBuffers {
  vertexBuffer: GPUBuffer
  indexBuffer: GPUBuffer
  indexCount: number
  indexFormat: GPUIndexFormat
  skinned: boolean
  skinBuffer?: GPUBuffer
}

export async function createWebGPURenderer(canvas: HTMLCanvasElement): Promise<Renderer> {
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error('No WebGPU adapter')
  const device = await adapter.requestDevice()

  const context = canvas.getContext('webgpu')!
  const format = navigator.gpu.getPreferredCanvasFormat()
  context.configure({ device, format, alphaMode: 'premultiplied' })

  // Shader modules
  const shaderModule = device.createShaderModule({ code: shaderSource })
  const skinnedShaderModule = device.createShaderModule({ code: skinnedShaderSource })

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

  const jointBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'read-only-storage', hasDynamicOffset: true },
      },
    ],
  })

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [cameraBindGroupLayout, modelBindGroupLayout, lightBindGroupLayout],
  })

  const skinnedPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [cameraBindGroupLayout, modelBindGroupLayout, lightBindGroupLayout, jointBindGroupLayout],
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

  // Non-skinned pipeline
  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: 'vs_main',
      buffers: [
        {
          arrayStride: 36, // 9 floats × 4 bytes
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32x3' },
            { shaderLocation: 2, offset: 24, format: 'float32x3' },
          ],
        },
      ],
    },
    fragment: {
      module: shaderModule,
      entryPoint: 'fs_main',
      targets: [{ format }],
    },
    primitive: { topology: 'triangle-list', cullMode: 'back' },
    depthStencil: { depthWriteEnabled: true, depthCompare: 'less', format: 'depth24plus' },
    multisample: { count: 4 },
  })

  // Skinned pipeline (two vertex buffers + joint storage)
  const skinnedPipeline = device.createRenderPipeline({
    layout: skinnedPipelineLayout,
    vertex: {
      module: skinnedShaderModule,
      entryPoint: 'vs_main',
      buffers: [
        {
          // Buffer 0: same geometry layout
          arrayStride: 36,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32x3' },
            { shaderLocation: 2, offset: 24, format: 'float32x3' },
          ],
        },
        {
          // Buffer 1: skin data [joints u8x4, weights f32x4] = 20 bytes
          arrayStride: 20,
          attributes: [
            { shaderLocation: 3, offset: 0, format: 'uint8x4' },
            { shaderLocation: 4, offset: 4, format: 'float32x4' },
          ],
        },
      ],
    },
    fragment: {
      module: skinnedShaderModule,
      entryPoint: 'fs_main',
      targets: [{ format }],
    },
    primitive: { topology: 'triangle-list', cullMode: 'back' },
    depthStencil: { depthWriteEnabled: true, depthCompare: 'less', format: 'depth24plus' },
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
    entries: [{ binding: 0, resource: { buffer: modelBuffer, size: MODEL_UNIFORM_SIZE } }],
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

  // Joint matrices storage buffer (for skinned meshes)
  const jointBuffer = device.createBuffer({
    size: JOINT_SLOT_BYTES * MAX_SKINNED_ENTITIES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })

  const jointBindGroup = device.createBindGroup({
    layout: jointBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: jointBuffer, size: JOINT_SLOT_BYTES } }],
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
        skinned: false,
      })
    },

    registerSkinnedGeometry(id, vertices, indices, joints, weights) {
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

      // Interleave skin data: [joints u8x4, weights f32x4] = 20 bytes per vertex
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

      const skinBuffer = device.createBuffer({
        size: skinBuf.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(skinBuffer, 0, skinBuf)

      geometries.set(id, {
        vertexBuffer,
        indexBuffer,
        indexCount: indices.length,
        indexFormat: indices instanceof Uint32Array ? 'uint32' : 'uint16',
        skinned: true,
        skinBuffer,
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
      // Pack model uniforms for all entities
      for (let i = 0; i < count; i++) {
        const entity = entities[i]!
        const base = (MODEL_UNIFORM_SIZE / 4) * i
        modelData.set(entity.worldMatrix, base)
        modelData.set(entity.color, base + 16)
        modelData[base + 20] = entity.unlit ? 1.0 : 0.0
      }
      device.queue.writeBuffer(modelBuffer, 0, modelData.buffer, 0, MODEL_UNIFORM_SIZE * count)

      // Upload joint matrices for skinned entities
      let skinSlot = 0
      const skinSlotMap = new Map<number, number>()
      for (let i = 0; i < count; i++) {
        const entity = entities[i]!
        if (!entity.jointMatrices) continue
        const geo = geometries.get(entity.geometryId)
        if (!geo?.skinned) continue
        device.queue.writeBuffer(
          jointBuffer,
          skinSlot * JOINT_SLOT_BYTES,
          entity.jointMatrices.buffer as ArrayBuffer,
          entity.jointMatrices.byteOffset,
          entity.jointMatrices.byteLength,
        )
        skinSlotMap.set(i, skinSlot)
        skinSlot++
      }

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

      // Draw non-skinned entities
      renderPass.setPipeline(pipeline)
      renderPass.setBindGroup(0, cameraBindGroup)
      renderPass.setBindGroup(2, lightBindGroup)

      for (let i = 0; i < count; i++) {
        const entity = entities[i]!
        const geo = geometries.get(entity.geometryId)
        if (!geo || geo.skinned) continue

        renderPass.setBindGroup(1, modelBindGroup, [MODEL_UNIFORM_SIZE * i])
        renderPass.setVertexBuffer(0, geo.vertexBuffer)
        renderPass.setIndexBuffer(geo.indexBuffer, geo.indexFormat)
        renderPass.drawIndexed(geo.indexCount)
      }

      // Draw skinned entities
      if (skinSlot > 0) {
        renderPass.setPipeline(skinnedPipeline)
        renderPass.setBindGroup(0, cameraBindGroup)
        renderPass.setBindGroup(2, lightBindGroup)

        for (let i = 0; i < count; i++) {
          const slot = skinSlotMap.get(i)
          if (slot === undefined) continue
          const entity = entities[i]!
          const geo = geometries.get(entity.geometryId)!

          renderPass.setBindGroup(1, modelBindGroup, [MODEL_UNIFORM_SIZE * i])
          renderPass.setBindGroup(3, jointBindGroup, [slot * JOINT_SLOT_BYTES])
          renderPass.setVertexBuffer(0, geo.vertexBuffer)
          renderPass.setVertexBuffer(1, geo.skinBuffer!)
          renderPass.setIndexBuffer(geo.indexBuffer, geo.indexFormat)
          renderPass.drawIndexed(geo.indexCount)
        }
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
      jointBuffer.destroy()
      depthTexture.destroy()
      msaaTexture.destroy()
      for (const geo of geometries.values()) {
        geo.vertexBuffer.destroy()
        geo.indexBuffer.destroy()
        geo.skinBuffer?.destroy()
      }
      device.destroy()
    },
  }

  return renderer
}
