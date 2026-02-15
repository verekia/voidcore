import {
  shaderSource,
  skinnedShaderSource,
  mrtShaderSource,
  mrtSkinnedShaderSource,
  texturedShaderSource,
  texturedMrtShaderSource,
  fullscreenVertexSource,
  downsampleSource,
  upsampleSource,
  compositeSource,
} from './shaders.ts'

import type { Backend } from './gpu.ts'

export interface BloomConfig {
  enabled: boolean
  intensity?: number
  radius?: number
}

export interface DrawEntity {
  worldMatrix: Float32Array
  color: Float32Array
  geometryId: number
  unlit: boolean
  jointMatrices?: Float32Array
  textureId?: number
  isTextured?: boolean
  aoIntensity?: number
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
  registerTexturedGeometry(
    id: number,
    vertices: Float32Array,
    indices: Uint16Array | Uint32Array,
    uvs: Float32Array,
  ): void
  registerTexture(id: number, data: Uint8Array, width: number, height: number): void
  updateCamera(view: Float32Array, projection: Float32Array): void
  updateLighting(dir: Float32Array, dirColor: Float32Array, ambient: Float32Array): void
  draw(entities: DrawEntity[], count: number): void
  resize(width: number, height: number): void
  setBloom(config: BloomConfig): void
  destroy(): void
}

// 256-byte aligned model uniform size (mat4 + vec4 + vec4 = 96 bytes, aligned to 256)
const MODEL_UNIFORM_SIZE = 256
const MAX_ENTITIES = 4096

const MAX_JOINTS = 128
const JOINT_SLOT_BYTES = MAX_JOINTS * 16 * 4 // 128 joints × 16 floats × 4 bytes = 8192
const MAX_SKINNED_ENTITIES = 1024

const BLOOM_MIP_LEVELS = 5

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
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
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

  const textureBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [cameraBindGroupLayout, modelBindGroupLayout, lightBindGroupLayout],
  })

  const skinnedPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [cameraBindGroupLayout, modelBindGroupLayout, lightBindGroupLayout, jointBindGroupLayout],
  })

  const texturedPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [cameraBindGroupLayout, modelBindGroupLayout, lightBindGroupLayout, textureBGL],
  })

  // Vertex buffer layouts
  const staticVertexBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 40, // 10 floats × 4 bytes
    attributes: [
      { shaderLocation: 0, offset: 0, format: 'float32x3' }, // position
      { shaderLocation: 1, offset: 12, format: 'float32x3' }, // normal
      { shaderLocation: 2, offset: 24, format: 'float32x3' }, // vertColor
      { shaderLocation: 3, offset: 36, format: 'float32' }, // bloom
    ],
  }

  const skinBufferLayout: GPUVertexBufferLayout = {
    // Buffer 1: skin data [joints u8x4, weights f32x4] = 20 bytes
    arrayStride: 20,
    attributes: [
      { shaderLocation: 4, offset: 0, format: 'uint8x4' }, // joints
      { shaderLocation: 5, offset: 4, format: 'float32x4' }, // weights
    ],
  }

  const uvBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 8, // 2 floats × 4 bytes
    attributes: [
      { shaderLocation: 4, offset: 0, format: 'float32x2' }, // uv
    ],
  }

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
      buffers: [staticVertexBufferLayout],
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
      buffers: [staticVertexBufferLayout, skinBufferLayout],
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

  // Textured pipeline (two vertex buffers: static + UV, texture bind group at slot 3)
  const texturedShaderModule = device.createShaderModule({ code: texturedShaderSource })
  const texturedPipeline = device.createRenderPipeline({
    layout: texturedPipelineLayout,
    vertex: {
      module: texturedShaderModule,
      entryPoint: 'vs_main',
      buffers: [staticVertexBufferLayout, uvBufferLayout],
    },
    fragment: {
      module: texturedShaderModule,
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

  // Textured geometry storage (includes UV buffer)
  interface TexturedGeometryGPU {
    vertexBuffer: GPUBuffer
    uvBuffer: GPUBuffer
    indexBuffer: GPUBuffer
    indexCount: number
    indexFormat: GPUIndexFormat
  }
  const texturedGeometries = new Map<number, TexturedGeometryGPU>()

  // Texture storage
  const textures = new Map<number, { bindGroup: GPUBindGroup; gpuTexture: GPUTexture }>()

  const textureSampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  })

  // ── Bloom state ──────────────────────────────────────────────────
  let bloomEnabled = false
  let bloomIntensity = 1.0
  let bloomRadius = 1.0

  // MRT pipelines (created lazily)
  let mrtPipeline: GPURenderPipeline | null = null
  let mrtSkinnedPipeline: GPURenderPipeline | null = null
  let mrtTexturedPipeline: GPURenderPipeline | null = null

  // Post-processing pipelines
  let downsamplePipeline: GPURenderPipeline | null = null
  let upsamplePipeline: GPURenderPipeline | null = null
  let compositePipeline: GPURenderPipeline | null = null

  // Bloom textures
  let msaaBloomTexture: GPUTexture | null = null
  let resolvedSceneTexture: GPUTexture | null = null
  let resolvedBloomTexture: GPUTexture | null = null
  let bloomMipTextures: GPUTexture[] = []
  let bloomSampler: GPUSampler | null = null

  // Post-processing bind group layouts
  let ppTextureBindGroupLayout: GPUBindGroupLayout | null = null
  let upsampleBindGroupLayout: GPUBindGroupLayout | null = null
  let compositeBindGroupLayout: GPUBindGroupLayout | null = null
  let fullscreenShaderModule: GPUShaderModule | null = null

  // Bloom uniform buffers
  let bloomRadiusBuffer: GPUBuffer | null = null
  let bloomIntensityBuffer: GPUBuffer | null = null

  // Pre-cached bind groups (recreated on resize)
  let downsampleBindGroups: GPUBindGroup[] = []
  let upsampleBindGroups: GPUBindGroup[] = []
  let compositeBindGroup: GPUBindGroup | null = null

  function createBloomResources() {
    const w = canvas.width || 1
    const h = canvas.height || 1

    // Sampler
    if (!bloomSampler) {
      bloomSampler = device.createSampler({
        magFilter: 'linear',
        minFilter: 'linear',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
      })
    }

    // Uniform buffers
    if (!bloomRadiusBuffer) {
      bloomRadiusBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
    }
    if (!bloomIntensityBuffer) {
      bloomIntensityBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
    }

    // Destroy old textures
    msaaBloomTexture?.destroy()
    resolvedSceneTexture?.destroy()
    resolvedBloomTexture?.destroy()
    for (const t of bloomMipTextures) t.destroy()

    // MSAA bloom texture (second MRT target)
    msaaBloomTexture = device.createTexture({
      size: [w, h],
      format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
      sampleCount: 4,
    })

    // Resolved textures (1x)
    resolvedSceneTexture = device.createTexture({
      size: [w, h],
      format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    resolvedBloomTexture = device.createTexture({
      size: [w, h],
      format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })

    // Mip chain textures for bloom blur
    bloomMipTextures = []
    let mw = Math.max(1, w >> 1)
    let mh = Math.max(1, h >> 1)
    for (let i = 0; i < BLOOM_MIP_LEVELS; i++) {
      bloomMipTextures.push(
        device.createTexture({
          size: [mw, mh],
          format,
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        }),
      )
      mw = Math.max(1, mw >> 1)
      mh = Math.max(1, mh >> 1)
    }

    rebuildBloomBindGroups()
  }

  function rebuildBloomBindGroups() {
    if (!ppTextureBindGroupLayout || !upsampleBindGroupLayout || !compositeBindGroupLayout) return
    if (!bloomSampler || !resolvedSceneTexture || !resolvedBloomTexture) return

    // Downsample bind groups: mip[0] reads from resolvedBloomTexture, mip[i] reads from mip[i-1]
    downsampleBindGroups = []
    for (let i = 0; i < BLOOM_MIP_LEVELS; i++) {
      const srcTexture = i === 0 ? resolvedBloomTexture : bloomMipTextures[i - 1]!
      downsampleBindGroups.push(
        device.createBindGroup({
          layout: ppTextureBindGroupLayout,
          entries: [
            { binding: 0, resource: srcTexture.createView() },
            { binding: 1, resource: bloomSampler },
          ],
        }),
      )
    }

    // Upsample bind groups: going from smallest mip back up
    upsampleBindGroups = []
    for (let i = BLOOM_MIP_LEVELS - 1; i >= 0; i--) {
      upsampleBindGroups.push(
        device.createBindGroup({
          layout: upsampleBindGroupLayout,
          entries: [
            { binding: 0, resource: bloomMipTextures[i]!.createView() },
            { binding: 1, resource: bloomSampler },
            { binding: 2, resource: { buffer: bloomRadiusBuffer! } },
          ],
        }),
      )
    }

    // Composite bind group
    // The upsample chain writes progressively back up. The final result is in mip[0].
    compositeBindGroup = device.createBindGroup({
      layout: compositeBindGroupLayout,
      entries: [
        { binding: 0, resource: resolvedSceneTexture.createView() },
        { binding: 1, resource: bloomMipTextures[0]!.createView() },
        { binding: 2, resource: bloomSampler },
        { binding: 3, resource: { buffer: bloomIntensityBuffer! } },
      ],
    })
  }

  function ensureBloomPipelines() {
    if (mrtPipeline) return

    // MRT shader modules
    const mrtModule = device.createShaderModule({ code: mrtShaderSource })
    const mrtSkinnedModule = device.createShaderModule({ code: mrtSkinnedShaderSource })
    fullscreenShaderModule = device.createShaderModule({ code: fullscreenVertexSource })

    // MRT pipelines (2 color targets)
    const mrtTargets: GPUColorTargetState[] = [{ format }, { format }]

    mrtPipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: mrtModule,
        entryPoint: 'vs_main',
        buffers: [staticVertexBufferLayout],
      },
      fragment: {
        module: mrtModule,
        entryPoint: 'fs_main',
        targets: mrtTargets,
      },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { depthWriteEnabled: true, depthCompare: 'less', format: 'depth24plus' },
      multisample: { count: 4 },
    })

    mrtSkinnedPipeline = device.createRenderPipeline({
      layout: skinnedPipelineLayout,
      vertex: {
        module: mrtSkinnedModule,
        entryPoint: 'vs_main',
        buffers: [staticVertexBufferLayout, skinBufferLayout],
      },
      fragment: {
        module: mrtSkinnedModule,
        entryPoint: 'fs_main',
        targets: mrtTargets,
      },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { depthWriteEnabled: true, depthCompare: 'less', format: 'depth24plus' },
      multisample: { count: 4 },
    })

    const mrtTexturedModule = device.createShaderModule({ code: texturedMrtShaderSource })
    mrtTexturedPipeline = device.createRenderPipeline({
      layout: texturedPipelineLayout,
      vertex: {
        module: mrtTexturedModule,
        entryPoint: 'vs_main',
        buffers: [staticVertexBufferLayout, uvBufferLayout],
      },
      fragment: {
        module: mrtTexturedModule,
        entryPoint: 'fs_main',
        targets: mrtTargets,
      },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { depthWriteEnabled: true, depthCompare: 'less', format: 'depth24plus' },
      multisample: { count: 4 },
    })

    // Post-processing bind group layouts
    ppTextureBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    })

    upsampleBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    })

    compositeBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    })

    // Downsample pipeline
    const downsampleModule = device.createShaderModule({ code: downsampleSource })
    const downsampleLayout = device.createPipelineLayout({ bindGroupLayouts: [ppTextureBindGroupLayout] })
    downsamplePipeline = device.createRenderPipeline({
      layout: downsampleLayout,
      vertex: { module: fullscreenShaderModule, entryPoint: 'vs_main' },
      fragment: { module: downsampleModule, entryPoint: 'fs_main', targets: [{ format }] },
    })

    // Upsample pipeline (additive blending)
    const upsampleModule = device.createShaderModule({ code: upsampleSource })
    const upsampleLayout = device.createPipelineLayout({ bindGroupLayouts: [upsampleBindGroupLayout] })
    upsamplePipeline = device.createRenderPipeline({
      layout: upsampleLayout,
      vertex: { module: fullscreenShaderModule, entryPoint: 'vs_main' },
      fragment: {
        module: upsampleModule,
        entryPoint: 'fs_main',
        targets: [
          {
            format,
            blend: {
              color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            },
          },
        ],
      },
    })

    // Composite pipeline
    const compositeModule = device.createShaderModule({ code: compositeSource })
    const compositeLayout = device.createPipelineLayout({ bindGroupLayouts: [compositeBindGroupLayout] })
    compositePipeline = device.createRenderPipeline({
      layout: compositeLayout,
      vertex: { module: fullscreenShaderModule, entryPoint: 'vs_main' },
      fragment: { module: compositeModule, entryPoint: 'fs_main', targets: [{ format }] },
    })
  }

  function uploadModelUniforms(entities: DrawEntity[], count: number) {
    for (let i = 0; i < count; i++) {
      const entity = entities[i]!
      const base = (MODEL_UNIFORM_SIZE / 4) * i
      modelData.set(entity.worldMatrix, base)
      modelData.set(entity.color, base + 16)
      modelData[base + 20] = entity.unlit ? 1.0 : 0.0
      modelData[base + 21] = entity.aoIntensity ?? 0.0
    }
    device.queue.writeBuffer(modelBuffer, 0, modelData.buffer, 0, MODEL_UNIFORM_SIZE * count)
  }

  function uploadJointMatrices(entities: DrawEntity[], count: number): Map<number, number> {
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
    return skinSlotMap
  }

  function drawEntitiesOnPass(
    renderPass: GPURenderPassEncoder,
    entities: DrawEntity[],
    count: number,
    skinSlotMap: Map<number, number>,
    useMrt: boolean,
  ) {
    const staticPipe = useMrt ? mrtPipeline! : pipeline
    const skinnedPipe = useMrt ? mrtSkinnedPipeline! : skinnedPipeline
    const texturedPipe = useMrt ? mrtTexturedPipeline! : texturedPipeline

    // Draw non-skinned, non-textured entities
    renderPass.setPipeline(staticPipe)
    renderPass.setBindGroup(0, cameraBindGroup)
    renderPass.setBindGroup(2, lightBindGroup)

    for (let i = 0; i < count; i++) {
      const entity = entities[i]!
      if (entity.isTextured) continue
      const geo = geometries.get(entity.geometryId)
      if (!geo || geo.skinned) continue

      renderPass.setBindGroup(1, modelBindGroup, [MODEL_UNIFORM_SIZE * i])
      renderPass.setVertexBuffer(0, geo.vertexBuffer)
      renderPass.setIndexBuffer(geo.indexBuffer, geo.indexFormat)
      renderPass.drawIndexed(geo.indexCount)
    }

    // Draw skinned entities
    if (skinSlotMap.size > 0) {
      renderPass.setPipeline(skinnedPipe)
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

    // Draw textured entities
    let texturedPipelineBound = false
    for (let i = 0; i < count; i++) {
      const entity = entities[i]!
      if (!entity.isTextured || entity.textureId === undefined) continue
      const geo = texturedGeometries.get(entity.geometryId)
      const tex = textures.get(entity.textureId)
      if (!geo || !tex) continue

      if (!texturedPipelineBound) {
        renderPass.setPipeline(texturedPipe)
        renderPass.setBindGroup(0, cameraBindGroup)
        renderPass.setBindGroup(2, lightBindGroup)
        texturedPipelineBound = true
      }

      renderPass.setBindGroup(1, modelBindGroup, [MODEL_UNIFORM_SIZE * i])
      renderPass.setBindGroup(3, tex.bindGroup)
      renderPass.setVertexBuffer(0, geo.vertexBuffer)
      renderPass.setVertexBuffer(1, geo.uvBuffer)
      renderPass.setIndexBuffer(geo.indexBuffer, geo.indexFormat)
      renderPass.drawIndexed(geo.indexCount)
    }
  }

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

    registerTexturedGeometry(id, vertices, indices, uvs) {
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

      const uvBuffer = device.createBuffer({
        size: uvs.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(uvBuffer, 0, new Float32Array(uvs))

      texturedGeometries.set(id, {
        vertexBuffer,
        uvBuffer,
        indexBuffer,
        indexCount: indices.length,
        indexFormat: indices instanceof Uint32Array ? 'uint32' : 'uint16',
      })
    },

    registerTexture(id, data, width, height) {
      const gpuTexture = device.createTexture({
        size: [width, height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      })
      device.queue.writeTexture(
        { texture: gpuTexture },
        data.buffer as ArrayBuffer,
        { offset: data.byteOffset, bytesPerRow: width * 4, rowsPerImage: height },
        [width, height],
      )

      const bindGroup = device.createBindGroup({
        layout: textureBGL,
        entries: [
          { binding: 0, resource: gpuTexture.createView() },
          { binding: 1, resource: textureSampler },
        ],
      })

      textures.set(id, { bindGroup, gpuTexture })
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
      uploadModelUniforms(entities, count)
      const skinSlotMap = uploadJointMatrices(entities, count)

      if (!bloomEnabled) {
        // ── Original non-bloom path ──
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

        drawEntitiesOnPass(renderPass, entities, count, skinSlotMap, false)
        renderPass.end()
        device.queue.submit([commandEncoder.finish()])
        return
      }

      // ── Bloom MRT path ──
      // Update bloom uniforms
      device.queue.writeBuffer(bloomRadiusBuffer!, 0, new Float32Array([bloomRadius]))
      device.queue.writeBuffer(bloomIntensityBuffer!, 0, new Float32Array([bloomIntensity]))

      const commandEncoder = device.createCommandEncoder()

      // 1. MRT render pass → MSAA scene + MSAA bloom → resolve to 1x textures
      const mrtPass = commandEncoder.beginRenderPass({
        colorAttachments: [
          {
            view: msaaTexture.createView(),
            resolveTarget: resolvedSceneTexture!.createView(),
            clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1 },
            loadOp: 'clear',
            storeOp: 'discard',
          },
          {
            view: msaaBloomTexture!.createView(),
            resolveTarget: resolvedBloomTexture!.createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
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

      drawEntitiesOnPass(mrtPass, entities, count, skinSlotMap, true)
      mrtPass.end()

      // 2. Downsample bloom through mip chain
      for (let i = 0; i < BLOOM_MIP_LEVELS; i++) {
        const dsPass = commandEncoder.beginRenderPass({
          colorAttachments: [
            {
              view: bloomMipTextures[i]!.createView(),
              clearValue: { r: 0, g: 0, b: 0, a: 0 },
              loadOp: 'clear',
              storeOp: 'store',
            },
          ],
        })
        dsPass.setPipeline(downsamplePipeline!)
        dsPass.setBindGroup(0, downsampleBindGroups[i]!)
        dsPass.draw(3)
        dsPass.end()
      }

      // 3. Upsample bloom back up the mip chain (additive blend)
      // Go from smallest mip (BLOOM_MIP_LEVELS-1) upward to mip 0
      // upsampleBindGroups[0] reads from mip[BLOOM_MIP_LEVELS-1], renders to mip[BLOOM_MIP_LEVELS-2]
      for (let i = 0; i < BLOOM_MIP_LEVELS - 1; i++) {
        const targetMipIdx = BLOOM_MIP_LEVELS - 2 - i
        const usPass = commandEncoder.beginRenderPass({
          colorAttachments: [
            {
              view: bloomMipTextures[targetMipIdx]!.createView(),
              loadOp: 'load',
              storeOp: 'store',
            },
          ],
        })
        usPass.setPipeline(upsamplePipeline!)
        usPass.setBindGroup(0, upsampleBindGroups[i]!)
        usPass.draw(3)
        usPass.end()
      }

      // 4. Composite scene + bloom → canvas
      const canvasView = context.getCurrentTexture().createView()
      const compPass = commandEncoder.beginRenderPass({
        colorAttachments: [
          {
            view: canvasView,
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
          },
        ],
      })
      compPass.setPipeline(compositePipeline!)
      compPass.setBindGroup(0, compositeBindGroup!)
      compPass.draw(3)
      compPass.end()

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
      if (bloomEnabled) {
        createBloomResources()
      }
    },

    setBloom(config: BloomConfig) {
      bloomEnabled = config.enabled
      bloomIntensity = config.intensity ?? 1.0
      bloomRadius = config.radius ?? 1.0
      if (bloomEnabled) {
        ensureBloomPipelines()
        createBloomResources()
      }
    },

    destroy() {
      cameraBuffer.destroy()
      modelBuffer.destroy()
      lightBuffer.destroy()
      jointBuffer.destroy()
      depthTexture.destroy()
      msaaTexture.destroy()
      msaaBloomTexture?.destroy()
      resolvedSceneTexture?.destroy()
      resolvedBloomTexture?.destroy()
      for (const t of bloomMipTextures) t.destroy()
      bloomRadiusBuffer?.destroy()
      bloomIntensityBuffer?.destroy()
      for (const geo of geometries.values()) {
        geo.vertexBuffer.destroy()
        geo.indexBuffer.destroy()
        geo.skinBuffer?.destroy()
      }
      for (const geo of texturedGeometries.values()) {
        geo.vertexBuffer.destroy()
        geo.uvBuffer.destroy()
        geo.indexBuffer.destroy()
      }
      for (const tex of textures.values()) {
        tex.gpuTexture.destroy()
      }
      device.destroy()
    },
  }

  return renderer
}
