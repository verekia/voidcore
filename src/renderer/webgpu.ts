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
  LAMBERT_WGSL,
  BASIC_WGSL,
  LAMBERT_SKINNED_WGSL,
  BASIC_SKINNED_WGSL,
  BLOOM_DOWN_WGSL,
  BLOOM_UP_WGSL,
  BLIT_WGSL,
} from './shaders-wgsl.ts'

import type { Geometry } from '../geometry/geometry.ts'
import type { PaletteEntry } from '../materials/material.ts'
import type { Material } from '../materials/material.ts'
import type { AABB, Mat4 } from '../math/index.ts'
import type { PerspectiveCamera } from '../scene/camera.ts'
import type { DirectionalLight } from '../scene/light.ts'
import type { Scene } from '../scene/scene.ts'
import type { Renderer, RendererConfig, FrameStats } from './renderer.ts'

// ─── Types ───────────────────────────────────────────────────────────

interface GeoBufs {
  position: GPUBuffer
  normal: GPUBuffer
  uv: GPUBuffer
  materialIndex: GPUBuffer
  index: GPUBuffer
  indexFormat: GPUIndexFormat
  indexCount: number
  joints?: GPUBuffer
  weights?: GPUBuffer
}

interface MatCache {
  buffer: GPUBuffer
  bindGroup: GPUBindGroup
}

interface MeshCache {
  buffer: GPUBuffer
  bindGroup: GPUBindGroup
}

interface RenderTargets {
  msaaColor: GPUTexture
  msaaColorView: GPUTextureView
  msaaEmissive: GPUTexture
  msaaEmissiveView: GPUTextureView
  msaaDepth: GPUTexture
  msaaDepthView: GPUTextureView
  resolvedColor: GPUTexture
  resolvedColorView: GPUTextureView
  resolvedEmissive: GPUTexture
  resolvedEmissiveView: GPUTextureView
  bloomTextures: GPUTexture[]
  bloomViews: GPUTextureView[]
  bloomWidths: number[]
  bloomHeights: number[]
  width: number
  height: number
}

// ─── Uniform buffer sizes ────────────────────────────────────────────

// FrameUniforms: mat4(64) + vec3+f32(16) + vec3+f32(16) + vec3+pad(16) = 112 bytes
const FRAME_UB_SIZE = 112
// ObjectUniforms: mat4(64) + mat4(64) = 128 bytes
const OBJECT_UB_SIZE = 128
// MaterialUniforms: vec3+f32(16) + f32+pad(16) + 32*PaletteEntry(32) = 1056 bytes
const MATERIAL_UB_SIZE = 1056
// Bloom down params: vec2+f32+pad(16)
const BLOOM_DOWN_UB_SIZE = 16
// Bloom up params: vec2+pad(16)
const BLOOM_UP_UB_SIZE = 16
// Blit params: f32+pad(16)
const BLIT_UB_SIZE = 16

// SkinnedObjectUniforms: mat4(64) + mat4(64) + 32*mat4(2048) = 2176 bytes
const SKINNED_OBJECT_UB_SIZE = 2176

// ─── Vertex buffer layout (shared by lambert + basic) ────────────────

const VERTEX_BUFFER_LAYOUT: GPUVertexBufferLayout[] = [
  { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' as GPUVertexFormat }] },
  { arrayStride: 12, attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' as GPUVertexFormat }] },
  { arrayStride: 8, attributes: [{ shaderLocation: 2, offset: 0, format: 'float32x2' as GPUVertexFormat }] },
  { arrayStride: 4, attributes: [{ shaderLocation: 3, offset: 0, format: 'float32' as GPUVertexFormat }] },
]

const SKINNED_VERTEX_BUFFER_LAYOUT: GPUVertexBufferLayout[] = [
  ...VERTEX_BUFFER_LAYOUT,
  { arrayStride: 4, attributes: [{ shaderLocation: 4, offset: 0, format: 'uint8x4' as GPUVertexFormat }] },
  { arrayStride: 16, attributes: [{ shaderLocation: 5, offset: 0, format: 'float32x4' as GPUVertexFormat }] },
]

// ─── Renderer ─────────────────────────────────────────────────────────

export class WebGPURenderer implements Renderer {
  readonly backend = 'webgpu' as const

  private device: GPUDevice
  private context: GPUCanvasContext
  private format: GPUTextureFormat
  private canvas: HTMLCanvasElement

  // Pipelines
  private lambertPipeline: GPURenderPipeline
  private basicPipeline: GPURenderPipeline
  private lambertSkinnedPipeline: GPURenderPipeline
  private basicSkinnedPipeline: GPURenderPipeline
  private bloomDownPipeline: GPURenderPipeline
  private bloomUpPipeline: GPURenderPipeline
  private blitPipeline: GPURenderPipeline

  // Bind group layouts
  private frameBGL: GPUBindGroupLayout
  private materialBGL: GPUBindGroupLayout
  private objectBGL: GPUBindGroupLayout
  private skinnedObjectBGL: GPUBindGroupLayout
  private postProcessBGL: GPUBindGroupLayout
  private blitBGL: GPUBindGroupLayout

  // Per-frame resources
  private frameUB: GPUBuffer
  private frameBG!: GPUBindGroup
  private linearSampler: GPUSampler

  // Config
  private samples: number
  private bloomLevels: number
  private bloomIntensity: number
  private bloomEnabled: boolean

  // Render targets
  private renderTargets: RenderTargets | null = null
  // Bloom / blit bind groups (recreated on resize)
  private bloomDownBGs: GPUBindGroup[] = []
  private bloomUpBGs: GPUBindGroup[] = []
  private blitBG: GPUBindGroup | null = null
  // Bloom uniform buffers
  private bloomDownUBs: GPUBuffer[] = []
  private bloomUpUBs: GPUBuffer[] = []
  private blitUB: GPUBuffer
  // 1x1 black texture for blit when bloom is disabled
  private dummyTexture: GPUTexture
  private dummyTextureView: GPUTextureView

  // Cached GPU resources
  private _geoCache = new WeakMap<Geometry, GeoBufs>()
  private _matCache = new WeakMap<Material, MatCache>()
  private _meshCache = new WeakMap<Mesh, MeshCache>()
  private _skinnedMeshCache = new WeakMap<Mesh, MeshCache>()

  // Scratch matrices
  private _vpMatrix: Mat4 = mat4Create()
  private _invWorldMatrix: Mat4 = mat4Create()
  private _normalMatrix: Mat4 = mat4Create()
  private _frustumPlanes = new Float32Array(24)
  private _worldAABB: AABB = new Float32Array(6)
  // Reusable typed arrays for uniform writes
  private _frameData = new Float32Array(FRAME_UB_SIZE / 4)
  private _objectData = new Float32Array(OBJECT_UB_SIZE / 4)
  private _skinnedObjectData = new Float32Array(SKINNED_OBJECT_UB_SIZE / 4)
  private _materialData = new Float32Array(MATERIAL_UB_SIZE / 4)

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

  private constructor(
    device: GPUDevice,
    context: GPUCanvasContext,
    format: GPUTextureFormat,
    canvas: HTMLCanvasElement,
    lambertPipeline: GPURenderPipeline,
    basicPipeline: GPURenderPipeline,
    lambertSkinnedPipeline: GPURenderPipeline,
    basicSkinnedPipeline: GPURenderPipeline,
    bloomDownPipeline: GPURenderPipeline,
    bloomUpPipeline: GPURenderPipeline,
    blitPipeline: GPURenderPipeline,
    frameBGL: GPUBindGroupLayout,
    materialBGL: GPUBindGroupLayout,
    objectBGL: GPUBindGroupLayout,
    skinnedObjectBGL: GPUBindGroupLayout,
    postProcessBGL: GPUBindGroupLayout,
    blitBGL: GPUBindGroupLayout,
    frameUB: GPUBuffer,
    linearSampler: GPUSampler,
    blitUB: GPUBuffer,
    dummyTexture: GPUTexture,
    samples: number,
    bloomEnabled: boolean,
    bloomIntensity: number,
    bloomLevels: number,
  ) {
    this.device = device
    this.context = context
    this.format = format
    this.canvas = canvas
    this.lambertPipeline = lambertPipeline
    this.basicPipeline = basicPipeline
    this.lambertSkinnedPipeline = lambertSkinnedPipeline
    this.basicSkinnedPipeline = basicSkinnedPipeline
    this.bloomDownPipeline = bloomDownPipeline
    this.bloomUpPipeline = bloomUpPipeline
    this.blitPipeline = blitPipeline
    this.frameBGL = frameBGL
    this.materialBGL = materialBGL
    this.objectBGL = objectBGL
    this.skinnedObjectBGL = skinnedObjectBGL
    this.postProcessBGL = postProcessBGL
    this.blitBGL = blitBGL
    this.frameUB = frameUB
    this.linearSampler = linearSampler
    this.blitUB = blitUB
    this.dummyTexture = dummyTexture
    this.dummyTextureView = dummyTexture.createView()
    this.samples = samples
    this.bloomEnabled = bloomEnabled
    this.bloomIntensity = bloomIntensity
    this.bloomLevels = bloomLevels

    // Create bloom uniform buffers
    for (let i = 0; i < bloomLevels; i++) {
      this.bloomDownUBs.push(
        device.createBuffer({ size: BLOOM_DOWN_UB_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      )
    }
    for (let i = 0; i < Math.max(0, bloomLevels - 1); i++) {
      this.bloomUpUBs.push(
        device.createBuffer({ size: BLOOM_UP_UB_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      )
    }

    // Create frame bind group (just needs the UB, no textures)
    this.frameBG = device.createBindGroup({
      layout: frameBGL,
      entries: [{ binding: 0, resource: { buffer: frameUB } }],
    })
  }

  static async create(canvas: HTMLCanvasElement, config: RendererConfig = {}): Promise<WebGPURenderer> {
    if (!navigator.gpu) throw new Error('WebGPU not supported')

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
    if (!adapter) throw new Error('No WebGPU adapter found')

    const device = await adapter.requestDevice()
    const context = canvas.getContext('webgpu')
    if (!context) throw new Error('WebGPU canvas context not available')

    const format = navigator.gpu.getPreferredCanvasFormat()
    context.configure({ device, format, alphaMode: 'opaque' })

    const samples = config.antialias !== false ? 4 : 1
    let bloomEnabled: boolean
    let bloomIntensity: number
    let bloomLevels: number

    const bloomConfig = config.bloom
    if (bloomConfig === false) {
      bloomEnabled = false
      bloomIntensity = 0
      bloomLevels = 0
    } else if (typeof bloomConfig === 'object') {
      bloomEnabled = true
      bloomIntensity = bloomConfig.intensity ?? 0.5
      bloomLevels = bloomConfig.levels ?? 5
    } else {
      bloomEnabled = true
      bloomIntensity = 0.5
      bloomLevels = 5
    }

    // ─── Shader modules ────────────────────────────────────────────
    const lambertModule = device.createShaderModule({ code: LAMBERT_WGSL })
    const basicModule = device.createShaderModule({ code: BASIC_WGSL })
    const lambertSkinnedModule = device.createShaderModule({ code: LAMBERT_SKINNED_WGSL })
    const basicSkinnedModule = device.createShaderModule({ code: BASIC_SKINNED_WGSL })
    const bloomDownModule = device.createShaderModule({ code: BLOOM_DOWN_WGSL })
    const bloomUpModule = device.createShaderModule({ code: BLOOM_UP_WGSL })
    const blitModule = device.createShaderModule({ code: BLIT_WGSL })

    // ─── Bind group layouts ────────────────────────────────────────
    const frameBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    })

    const materialBGL = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
    })

    const objectBGL = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
    })

    const skinnedObjectBGL = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: 'uniform', minBindingSize: SKINNED_OBJECT_UB_SIZE },
        },
      ],
    })

    // Shared by bloom down + bloom up
    const postProcessBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    })

    const blitBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    })

    // ─── Pipeline layouts ──────────────────────────────────────────
    const opaquePipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [frameBGL, materialBGL, objectBGL],
    })

    const skinnedOpaquePipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [frameBGL, materialBGL, skinnedObjectBGL],
    })

    const postProcessPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [postProcessBGL],
    })

    const blitPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [blitBGL],
    })

    // ─── MRT fragment targets (color + emissive) ───────────────────
    const opaqueTargets: GPUColorTargetState[] = [{ format: 'rgba8unorm' }, { format: 'rgba16float' }]

    // ─── Render pipelines ──────────────────────────────────────────
    const lambertPipeline = device.createRenderPipeline({
      layout: opaquePipelineLayout,
      vertex: { module: lambertModule, entryPoint: 'vs_main', buffers: VERTEX_BUFFER_LAYOUT },
      fragment: { module: lambertModule, entryPoint: 'fs_main', targets: opaqueTargets },
      primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less-equal' },
      multisample: { count: samples },
    })

    const basicPipeline = device.createRenderPipeline({
      layout: opaquePipelineLayout,
      vertex: { module: basicModule, entryPoint: 'vs_main', buffers: VERTEX_BUFFER_LAYOUT },
      fragment: { module: basicModule, entryPoint: 'fs_main', targets: opaqueTargets },
      primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less-equal' },
      multisample: { count: samples },
    })

    const lambertSkinnedPipeline = device.createRenderPipeline({
      layout: skinnedOpaquePipelineLayout,
      vertex: { module: lambertSkinnedModule, entryPoint: 'vs_main', buffers: SKINNED_VERTEX_BUFFER_LAYOUT },
      fragment: { module: lambertSkinnedModule, entryPoint: 'fs_main', targets: opaqueTargets },
      primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less-equal' },
      multisample: { count: samples },
    })

    const basicSkinnedPipeline = device.createRenderPipeline({
      layout: skinnedOpaquePipelineLayout,
      vertex: { module: basicSkinnedModule, entryPoint: 'vs_main', buffers: SKINNED_VERTEX_BUFFER_LAYOUT },
      fragment: { module: basicSkinnedModule, entryPoint: 'fs_main', targets: opaqueTargets },
      primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less-equal' },
      multisample: { count: samples },
    })

    const bloomDownPipeline = device.createRenderPipeline({
      layout: postProcessPipelineLayout,
      vertex: { module: bloomDownModule, entryPoint: 'vs_main' },
      fragment: { module: bloomDownModule, entryPoint: 'fs_main', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list' },
    })

    const bloomUpPipeline = device.createRenderPipeline({
      layout: postProcessPipelineLayout,
      vertex: { module: bloomUpModule, entryPoint: 'vs_main' },
      fragment: {
        module: bloomUpModule,
        entryPoint: 'fs_main',
        targets: [
          {
            format: 'rgba16float',
            blend: {
              color: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
              alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list' },
    })

    const blitPipeline = device.createRenderPipeline({
      layout: blitPipelineLayout,
      vertex: { module: blitModule, entryPoint: 'vs_main' },
      fragment: { module: blitModule, entryPoint: 'fs_main', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    })

    // ─── Shared resources ──────────────────────────────────────────
    const frameUB = device.createBuffer({
      size: FRAME_UB_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    const linearSampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    })

    const blitUB = device.createBuffer({
      size: BLIT_UB_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    // 1x1 black texture for when bloom is disabled
    const dummyTexture = device.createTexture({
      size: [1, 1],
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    device.queue.writeTexture(
      { texture: dummyTexture },
      new ArrayBuffer(8), // 4 * f16 zeros = black
      { bytesPerRow: 8 },
      [1, 1],
    )

    return new WebGPURenderer(
      device,
      context,
      format,
      canvas,
      lambertPipeline,
      basicPipeline,
      lambertSkinnedPipeline,
      basicSkinnedPipeline,
      bloomDownPipeline,
      bloomUpPipeline,
      blitPipeline,
      frameBGL,
      materialBGL,
      objectBGL,
      skinnedObjectBGL,
      postProcessBGL,
      blitBGL,
      frameUB,
      linearSampler,
      blitUB,
      dummyTexture,
      samples,
      bloomEnabled,
      bloomIntensity,
      bloomLevels,
    )
  }

  // ─── Render targets ──────────────────────────────────────────────

  private createRenderTargets(w: number, h: number): RenderTargets {
    const d = this.device

    const msaaColor = d.createTexture({
      size: [w, h],
      format: 'rgba8unorm',
      sampleCount: this.samples,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    })
    const msaaEmissive = d.createTexture({
      size: [w, h],
      format: 'rgba16float',
      sampleCount: this.samples,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    })
    const msaaDepth = d.createTexture({
      size: [w, h],
      format: 'depth24plus',
      sampleCount: this.samples,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    })

    const resolvedColor = d.createTexture({
      size: [w, h],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    const resolvedEmissive = d.createTexture({
      size: [w, h],
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })

    // Bloom chain
    const bloomTextures: GPUTexture[] = []
    const bloomViews: GPUTextureView[] = []
    const bloomWidths: number[] = []
    const bloomHeights: number[] = []

    let bw = Math.max(1, Math.floor(w / 2))
    let bh = Math.max(1, Math.floor(h / 2))

    for (let i = 0; i < this.bloomLevels; i++) {
      const tex = d.createTexture({
        size: [bw, bh],
        format: 'rgba16float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      })
      bloomTextures.push(tex)
      bloomViews.push(tex.createView())
      bloomWidths.push(bw)
      bloomHeights.push(bh)
      bw = Math.max(1, Math.floor(bw / 2))
      bh = Math.max(1, Math.floor(bh / 2))
    }

    return {
      msaaColor,
      msaaColorView: msaaColor.createView(),
      msaaEmissive,
      msaaEmissiveView: msaaEmissive.createView(),
      msaaDepth,
      msaaDepthView: msaaDepth.createView(),
      resolvedColor,
      resolvedColorView: resolvedColor.createView(),
      resolvedEmissive,
      resolvedEmissiveView: resolvedEmissive.createView(),
      bloomTextures,
      bloomViews,
      bloomWidths,
      bloomHeights,
      width: w,
      height: h,
    }
  }

  private destroyRenderTargets() {
    if (!this.renderTargets) return
    const rt = this.renderTargets
    rt.msaaColor.destroy()
    rt.msaaEmissive.destroy()
    rt.msaaDepth.destroy()
    rt.resolvedColor.destroy()
    rt.resolvedEmissive.destroy()
    for (const tex of rt.bloomTextures) tex.destroy()
    this.renderTargets = null
  }

  private ensureRenderTargets() {
    const w = this.canvas.width
    const h = this.canvas.height
    if (this.renderTargets && this.renderTargets.width === w && this.renderTargets.height === h) return
    if (this.renderTargets) this.destroyRenderTargets()
    this.renderTargets = this.createRenderTargets(w, h)
    this.rebuildPostProcessBindGroups()
  }

  private rebuildPostProcessBindGroups() {
    const rt = this.renderTargets!
    const d = this.device

    // Bloom downsample bind groups
    this.bloomDownBGs = []
    let srcView = rt.resolvedEmissiveView
    let srcW = rt.width
    let srcH = rt.height

    for (let i = 0; i < this.bloomLevels; i++) {
      // Write texel size + karis flag
      const data = new Float32Array([1 / srcW, 1 / srcH, i === 0 ? 1.0 : 0.0, 0])
      d.queue.writeBuffer(this.bloomDownUBs[i]!, 0, data.buffer, data.byteOffset, data.byteLength)

      this.bloomDownBGs.push(
        d.createBindGroup({
          layout: this.postProcessBGL,
          entries: [
            { binding: 0, resource: this.linearSampler },
            { binding: 1, resource: srcView },
            { binding: 2, resource: { buffer: this.bloomDownUBs[i]! } },
          ],
        }),
      )

      srcView = rt.bloomViews[i]!
      srcW = rt.bloomWidths[i]!
      srcH = rt.bloomHeights[i]!
    }

    // Bloom upsample bind groups
    this.bloomUpBGs = []
    for (let i = this.bloomLevels - 1; i > 0; i--) {
      const data = new Float32Array([1 / rt.bloomWidths[i]!, 1 / rt.bloomHeights[i]!, 0, 0])
      const ubIdx = this.bloomLevels - 1 - i
      d.queue.writeBuffer(this.bloomUpUBs[ubIdx]!, 0, data.buffer, data.byteOffset, data.byteLength)

      this.bloomUpBGs.push(
        d.createBindGroup({
          layout: this.postProcessBGL,
          entries: [
            { binding: 0, resource: this.linearSampler },
            { binding: 1, resource: rt.bloomViews[i]! },
            { binding: 2, resource: { buffer: this.bloomUpUBs[ubIdx]! } },
          ],
        }),
      )
    }

    // Blit bind group
    const bloomView = this.bloomEnabled && this.bloomLevels > 0 ? rt.bloomViews[0]! : this.dummyTextureView

    const blitData = new Float32Array([this.bloomIntensity, 0, 0, 0])
    d.queue.writeBuffer(this.blitUB, 0, blitData.buffer, blitData.byteOffset, blitData.byteLength)

    this.blitBG = d.createBindGroup({
      layout: this.blitBGL,
      entries: [
        { binding: 0, resource: this.linearSampler },
        { binding: 1, resource: rt.resolvedColorView },
        { binding: 2, resource: bloomView },
        { binding: 3, resource: { buffer: this.blitUB } },
      ],
    })
  }

  // ─── GPU buffer management ──────────────────────────────────────

  private ensureGeometryBuffers(geometry: Geometry): GeoBufs {
    const cached = this._geoCache.get(geometry)
    if (cached && !geometry.needsUpdate) return cached

    const d = this.device

    const positionBuf = d.createBuffer({
      size: geometry.positions.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    d.queue.writeBuffer(
      positionBuf,
      0,
      geometry.positions.buffer,
      geometry.positions.byteOffset,
      geometry.positions.byteLength,
    )

    const normalBuf = d.createBuffer({
      size: geometry.normals.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    d.queue.writeBuffer(normalBuf, 0, geometry.normals.buffer, geometry.normals.byteOffset, geometry.normals.byteLength)

    // UVs: use geometry UVs or zero-filled
    let uvData: Float32Array
    if (geometry.uvs) {
      uvData = geometry.uvs
    } else {
      uvData = new Float32Array(geometry.vertexCount * 2)
    }
    const uvBuf = d.createBuffer({
      size: uvData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    d.queue.writeBuffer(uvBuf, 0, uvData.buffer, uvData.byteOffset, uvData.byteLength)

    // Material indices: convert Uint8 to Float32 or zero-filled
    let matIdxData: Float32Array
    if (geometry.materialIndices) {
      matIdxData = new Float32Array(geometry.materialIndices.length)
      for (let i = 0; i < geometry.materialIndices.length; i++) {
        matIdxData[i] = geometry.materialIndices[i]!
      }
    } else {
      matIdxData = new Float32Array(geometry.vertexCount)
    }
    const matIdxBuf = d.createBuffer({
      size: matIdxData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    d.queue.writeBuffer(matIdxBuf, 0, matIdxData.buffer, matIdxData.byteOffset, matIdxData.byteLength)

    // Index buffer
    const indexFormat: GPUIndexFormat = geometry.indices instanceof Uint32Array ? 'uint32' : 'uint16'
    const indexBuf = d.createBuffer({
      size: geometry.indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    })
    d.queue.writeBuffer(indexBuf, 0, geometry.indices.buffer, geometry.indices.byteOffset, geometry.indices.byteLength)

    // Joints buffer (uint8x4 raw — WGSL reads as vec4<u32>)
    let jointsBuf: GPUBuffer | undefined
    if (geometry.joints) {
      // Need Uint8Array for uint8x4 vertex format
      const jointsU8 = geometry.joints instanceof Uint8Array ? geometry.joints : new Uint8Array(geometry.joints)
      jointsBuf = d.createBuffer({
        size: Math.max(jointsU8.byteLength, 4),
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      })
      d.queue.writeBuffer(jointsBuf, 0, jointsU8.buffer, jointsU8.byteOffset, jointsU8.byteLength)
    }

    // Weights buffer (float32x4)
    let weightsBuf: GPUBuffer | undefined
    if (geometry.weights) {
      weightsBuf = d.createBuffer({
        size: geometry.weights.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      })
      d.queue.writeBuffer(
        weightsBuf,
        0,
        geometry.weights.buffer,
        geometry.weights.byteOffset,
        geometry.weights.byteLength,
      )
    }

    // Destroy old buffers
    if (cached) {
      cached.position.destroy()
      cached.normal.destroy()
      cached.uv.destroy()
      cached.materialIndex.destroy()
      cached.index.destroy()
      if (cached.joints) cached.joints.destroy()
      if (cached.weights) cached.weights.destroy()
    }

    const bufs: GeoBufs = {
      position: positionBuf,
      normal: normalBuf,
      uv: uvBuf,
      materialIndex: matIdxBuf,
      index: indexBuf,
      indexFormat,
      indexCount: geometry.indexCount,
      joints: jointsBuf,
      weights: weightsBuf,
    }
    this._geoCache.set(geometry, bufs)
    geometry.needsUpdate = false
    return bufs
  }

  private ensureMaterialCache(material: Material): MatCache {
    const cached = this._matCache.get(material)
    if (cached) return cached

    const d = this.device
    const buffer = d.createBuffer({
      size: MATERIAL_UB_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const bindGroup = d.createBindGroup({
      layout: this.materialBGL,
      entries: [{ binding: 0, resource: { buffer } }],
    })

    const cache: MatCache = { buffer, bindGroup }
    this._matCache.set(material, cache)
    return cache
  }

  private writeMaterialBuffer(material: Material, cache: MatCache) {
    const data = this._materialData
    data.fill(0)

    data[0] = material.color[0]
    data[1] = material.color[1]
    data[2] = material.color[2]
    data[3] = material.opacity

    const hasPalette = !!material.palette
    data[4] = hasPalette ? 1.0 : 0.0

    if (hasPalette && material.palette) {
      for (let i = 0; i < 32; i++) {
        const entry: PaletteEntry = material.palette[i] ?? { color: [1, 1, 1] }
        const offset = 8 + i * 8
        data[offset] = entry.color[0]
        data[offset + 1] = entry.color[1]
        data[offset + 2] = entry.color[2]
        data[offset + 3] = entry.opacity ?? 1.0
        data[offset + 4] = entry.emissive?.[0] ?? 0
        data[offset + 5] = entry.emissive?.[1] ?? 0
        data[offset + 6] = entry.emissive?.[2] ?? 0
        data[offset + 7] = entry.emissiveIntensity ?? 0
      }
    }

    this.device.queue.writeBuffer(cache.buffer, 0, data.buffer, data.byteOffset, data.byteLength)
  }

  private ensureMeshCache(mesh: Mesh): MeshCache {
    const cached = this._meshCache.get(mesh)
    if (cached) return cached

    const d = this.device
    const buffer = d.createBuffer({
      size: OBJECT_UB_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const bindGroup = d.createBindGroup({
      layout: this.objectBGL,
      entries: [{ binding: 0, resource: { buffer } }],
    })

    const cache: MeshCache = { buffer, bindGroup }
    this._meshCache.set(mesh, cache)
    return cache
  }

  private writeObjectBuffer(mesh: Mesh, cache: MeshCache) {
    const data = this._objectData
    // World matrix (16 floats)
    data.set(mesh._worldMatrix, 0)
    // Normal matrix = transpose(inverse(worldMatrix)) (16 floats)
    if (mat4Invert(this._invWorldMatrix, mesh._worldMatrix)) {
      mat4Transpose(this._normalMatrix, this._invWorldMatrix)
    }
    data.set(this._normalMatrix, 16)

    this.device.queue.writeBuffer(cache.buffer, 0, data.buffer, data.byteOffset, data.byteLength)
  }

  private ensureSkinnedMeshCache(mesh: Mesh): MeshCache {
    const cached = this._skinnedMeshCache.get(mesh)
    if (cached) return cached

    const d = this.device
    const buffer = d.createBuffer({
      size: SKINNED_OBJECT_UB_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const bindGroup = d.createBindGroup({
      layout: this.skinnedObjectBGL,
      entries: [{ binding: 0, resource: { buffer } }],
    })

    const cache: MeshCache = { buffer, bindGroup }
    this._skinnedMeshCache.set(mesh, cache)
    return cache
  }

  private writeSkinnedObjectBuffer(mesh: Mesh, cache: MeshCache) {
    const data = this._skinnedObjectData
    // World matrix (16 floats)
    data.set(mesh._worldMatrix, 0)
    // Normal matrix = transpose(inverse(worldMatrix)) (16 floats)
    if (mat4Invert(this._invWorldMatrix, mesh._worldMatrix)) {
      mat4Transpose(this._normalMatrix, this._invWorldMatrix)
    }
    data.set(this._normalMatrix, 16)
    // Bone matrices (32 * 16 = 512 floats starting at offset 32)
    data.set(mesh.skeleton!.boneMatrices, 32)

    this.device.queue.writeBuffer(cache.buffer, 0, data.buffer, data.byteOffset, data.byteLength)
  }

  // ─── Render ──────────────────────────────────────────────────────

  render(scene: Scene, camera: PerspectiveCamera) {
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
    camera.updateProjection('zero-to-one') // WebGPU clip space

    // Update scene graph
    scene.updateGraph()

    // View-projection matrix
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

    // Compute light direction
    const lightDir = vec3Create()
    if (dirLight) {
      const lp = (dirLight as DirectionalLight)._worldMatrix
      const lx = lp[12]!,
        ly = lp[13]!,
        lz = lp[14]!
      vec3Normalize(lightDir, new Float32Array([lx, ly, lz]))
    }

    // ─── Update per-frame uniform buffer ────────────────────────
    const fd = this._frameData
    fd.set(this._vpMatrix, 0)
    fd[16] = lightDir[0]!
    fd[17] = lightDir[1]!
    fd[18] = lightDir[2]!
    fd[19] = dirLight ? (dirLight as DirectionalLight).intensity : 0
    fd[20] = dirLight ? (dirLight as DirectionalLight).color[0] : 0
    fd[21] = dirLight ? (dirLight as DirectionalLight).color[1] : 0
    fd[22] = dirLight ? (dirLight as DirectionalLight).color[2] : 0
    fd[23] = scene.ambientLight.intensity
    fd[24] = scene.ambientLight.color[0]
    fd[25] = scene.ambientLight.color[1]
    fd[26] = scene.ambientLight.color[2]
    this.device.queue.writeBuffer(this.frameUB, 0, fd.buffer, fd.byteOffset, fd.byteLength)

    // ─── Ensure render targets ──────────────────────────────────
    this.ensureRenderTargets()
    const rt = this.renderTargets!

    const encoder = this.device.createCommandEncoder()

    // ─── Opaque pass (MSAA MRT) ─────────────────────────────────
    const opaquePass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: rt.msaaColorView,
          resolveTarget: rt.resolvedColorView,
          loadOp: 'clear',
          storeOp: 'discard',
          clearValue: { r: 0.05, g: 0.05, b: 0.08, a: 1.0 },
        },
        {
          view: rt.msaaEmissiveView,
          resolveTarget: rt.resolvedEmissiveView,
          loadOp: 'clear',
          storeOp: 'discard',
          clearValue: { r: 0, g: 0, b: 0, a: 1.0 },
        },
      ],
      depthStencilAttachment: {
        view: rt.msaaDepthView,
        depthLoadOp: 'clear',
        depthClearValue: 1.0,
        depthStoreOp: 'discard',
      },
    })

    let drawCalls = 0
    let triangles = 0

    for (const mesh of meshes) {
      const isSkinned = !!mesh.skeleton && !!mesh.geometry.joints && !!mesh.geometry.weights
      let pipeline: GPURenderPipeline
      if (isSkinned) {
        pipeline = mesh.material.type === 'lambert' ? this.lambertSkinnedPipeline : this.basicSkinnedPipeline
      } else {
        pipeline = mesh.material.type === 'lambert' ? this.lambertPipeline : this.basicPipeline
      }
      opaquePass.setPipeline(pipeline)

      const geoBufs = this.ensureGeometryBuffers(mesh.geometry)
      opaquePass.setVertexBuffer(0, geoBufs.position)
      opaquePass.setVertexBuffer(1, geoBufs.normal)
      opaquePass.setVertexBuffer(2, geoBufs.uv)
      opaquePass.setVertexBuffer(3, geoBufs.materialIndex)
      if (isSkinned && geoBufs.joints && geoBufs.weights) {
        opaquePass.setVertexBuffer(4, geoBufs.joints)
        opaquePass.setVertexBuffer(5, geoBufs.weights)
      }
      opaquePass.setIndexBuffer(geoBufs.index, geoBufs.indexFormat)

      const matCache = this.ensureMaterialCache(mesh.material)
      this.writeMaterialBuffer(mesh.material, matCache)

      if (isSkinned) {
        mesh.skeleton!.update()
        const skinnedCache = this.ensureSkinnedMeshCache(mesh)
        this.writeSkinnedObjectBuffer(mesh, skinnedCache)
        opaquePass.setBindGroup(2, skinnedCache.bindGroup)
      } else {
        const meshCache = this.ensureMeshCache(mesh)
        this.writeObjectBuffer(mesh, meshCache)
        opaquePass.setBindGroup(2, meshCache.bindGroup)
      }

      opaquePass.setBindGroup(0, this.frameBG)
      opaquePass.setBindGroup(1, matCache.bindGroup)

      opaquePass.drawIndexed(geoBufs.indexCount)
      drawCalls++
      triangles += geoBufs.indexCount / 3
    }

    opaquePass.end()

    // ─── Bloom ──────────────────────────────────────────────────
    if (this.bloomEnabled && this.bloomLevels > 0) {
      // Downsample chain
      for (let i = 0; i < this.bloomLevels; i++) {
        const pass = encoder.beginRenderPass({
          colorAttachments: [
            {
              view: rt.bloomViews[i]!,
              loadOp: 'clear',
              storeOp: 'store',
              clearValue: { r: 0, g: 0, b: 0, a: 1 },
            },
          ],
        })
        pass.setPipeline(this.bloomDownPipeline)
        pass.setBindGroup(0, this.bloomDownBGs[i]!)
        pass.draw(3)
        pass.end()
      }

      // Upsample chain (additive)
      for (let i = 0; i < this.bloomUpBGs.length; i++) {
        const targetIdx = this.bloomLevels - 2 - i
        const pass = encoder.beginRenderPass({
          colorAttachments: [
            {
              view: rt.bloomViews[targetIdx]!,
              loadOp: 'load',
              storeOp: 'store',
            },
          ],
        })
        pass.setPipeline(this.bloomUpPipeline)
        pass.setBindGroup(0, this.bloomUpBGs[i]!)
        pass.draw(3)
        pass.end()
      }
    }

    // ─── Final blit ─────────────────────────────────────────────
    const surfaceTexture = this.context.getCurrentTexture()
    const blitPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: surfaceTexture.createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
    })
    blitPass.setPipeline(this.blitPipeline)
    blitPass.setBindGroup(0, this.blitBG!)
    blitPass.draw(3)
    blitPass.end()

    this.device.queue.submit([encoder.finish()])

    // Update stats
    this.stats.fps = this._currentFps
    this.stats.frameTime = dt
    this.stats.drawCalls = drawCalls
    this.stats.triangles = triangles
    this.stats.visibleObjects = meshes.length
    this.stats.culledObjects = culledCount
  }

  // ─── Dispose ──────────────────────────────────────────────────

  dispose() {
    this.destroyRenderTargets()
    this.frameUB.destroy()
    this.blitUB.destroy()
    this.dummyTexture.destroy()
    for (const ub of this.bloomDownUBs) ub.destroy()
    for (const ub of this.bloomUpUBs) ub.destroy()
    this.device.destroy()
  }
}
