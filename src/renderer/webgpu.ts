// WebGPU Renderer – Renders the scene using the WebGPU API.
//
// This is the high-performance rendering backend. Each frame it:
//   1. Resizes the canvas to match display resolution (accounting for device pixel ratio)
//   2. Updates the camera's projection matrix
//   3. Walks the scene graph to collect visible meshes (with frustum culling)
//   4. Sorts meshes by material/pipeline to minimize GPU state changes
//   5. Uploads per-frame uniforms (view-projection matrix, light data) to the GPU
//   6. Batches all object transforms into dynamic uniform buffers (1-2 uploads vs N)
//   7. Renders shadow maps (3-cascade CSM with comparison sampling)
//   8. Draws meshes in an MSAA render pass with two render targets (color + emissive)
//   9. MSAA resolve
//  10. Runs a bloom post-processing chain (downsample → upsample with tent filter)
//  11. Blits the final image to screen with bloom compositing and gamma correction
//
// Key concepts:
//   Pipeline      – A compiled GPU program (vertex + fragment shader + render state).
//   Bind group    – A set of GPU resources (buffers, textures) bound to shader slots.
//   Uniform buffer – CPU-to-GPU data (matrices, colors) uploaded once and read by shaders.
//   MSAA          – Multi-sample anti-aliasing (4x) to smooth jagged edges.
//   Premul alpha  – Transparent pipelines use premultiplied alpha blend (src=one, dst=
//                    one-minus-src-alpha). This avoids the 'src-alpha' blend factor which
//                    triggers VK_ERROR_UNKNOWN on some Android Vulkan drivers when combined
//                    with comparison texture sampling (textureSampleCompareLevel).
//   MRT           – Multiple render targets: color and emissive are written simultaneously.
//   Vertex packing – Normals use snorm8x4, UVs use float16x2, bone weights use unorm8x4
//                    to reduce vertex buffer size (~40% smaller than all-float32).
//
// WebGPURenderer.create()  – Async factory that initializes the GPU device and pipelines.
// WebGPURenderer.render()  – Draws one frame.
// WebGPURenderer.dispose() – Releases all GPU resources.

import {
  frustumFromViewProjection,
  mat4Create,
  mat4Invert,
  mat4Multiply,
  mat4OrthoZO,
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
  LAMBERT_WGSL,
  BASIC_WGSL,
  LAMBERT_SKINNED_WGSL,
  BASIC_SKINNED_WGSL,
  SHADOW_DEPTH_WGSL,
  SHADOW_DEPTH_SKINNED_WGSL,
  BLOOM_DOWN_WGSL,
  BLOOM_UP_WGSL,
  BLIT_WGSL,
} from './webgpu-shaders.ts'

import type { Geometry } from '../geometry/geometry.ts'
import type { PaletteEntry } from '../materials/material.ts'
import type { Material } from '../materials/material.ts'
import type { AABB, Mat4, Vec3 } from '../math/index.ts'
import type { PerspectiveCamera } from '../scene/camera.ts'
import type { DirectionalLight } from '../scene/light.ts'
import type { Scene } from '../scene/scene.ts'
import type { Renderer, RendererConfig, FrameStats } from './renderer.ts'
import type { SortState } from './sort.ts'

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

// FrameUniforms: mat4(64) + light(48) + 3×cascadeVP(192) + splits+bias(32) = 336 bytes
const FRAME_UB_SIZE = 336
// ShadowUniforms: mat4(64)
const SHADOW_UB_SIZE = 64
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
  { arrayStride: 4, attributes: [{ shaderLocation: 1, offset: 0, format: 'snorm8x4' as GPUVertexFormat }] },
  { arrayStride: 4, attributes: [{ shaderLocation: 2, offset: 0, format: 'float16x2' as GPUVertexFormat }] },
  { arrayStride: 4, attributes: [{ shaderLocation: 3, offset: 0, format: 'float32' as GPUVertexFormat }] },
]

const SKINNED_VERTEX_BUFFER_LAYOUT: GPUVertexBufferLayout[] = [
  ...VERTEX_BUFFER_LAYOUT,
  { arrayStride: 4, attributes: [{ shaderLocation: 4, offset: 0, format: 'uint8x4' as GPUVertexFormat }] },
  { arrayStride: 4, attributes: [{ shaderLocation: 5, offset: 0, format: 'unorm8x4' as GPUVertexFormat }] },
]

// Shadow depth pass: position only
const SHADOW_VERTEX_BUFFER_LAYOUT: GPUVertexBufferLayout[] = [
  { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' as GPUVertexFormat }] },
]

// Shadow depth pass: position + joints + weights (skinned)
const SHADOW_SKINNED_VERTEX_BUFFER_LAYOUT: GPUVertexBufferLayout[] = [
  { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' as GPUVertexFormat }] },
  { arrayStride: 4, attributes: [{ shaderLocation: 1, offset: 0, format: 'uint8x4' as GPUVertexFormat }] },
  { arrayStride: 4, attributes: [{ shaderLocation: 2, offset: 0, format: 'unorm8x4' as GPUVertexFormat }] },
]

// ─── Renderer ─────────────────────────────────────────────────────────

export class WebGPURenderer implements Renderer {
  readonly backend = 'webgpu' as const

  // DPR limiting
  private _maxDpr: number = 1.5

  get maxDpr(): number {
    return this._maxDpr
  }

  set maxDpr(value: number) {
    this._maxDpr = value
  }

  private device: GPUDevice
  private context: GPUCanvasContext
  private format: GPUTextureFormat
  private canvas: HTMLCanvasElement

  // Pipelines
  private lambertPipeline: GPURenderPipeline
  private basicPipeline: GPURenderPipeline
  private lambertSkinnedPipeline: GPURenderPipeline
  private basicSkinnedPipeline: GPURenderPipeline
  // Transparent variants (blend + no depth write + no cull)
  private lambertTransparentPipeline: GPURenderPipeline
  private basicTransparentPipeline: GPURenderPipeline
  private lambertSkinnedTransparentPipeline: GPURenderPipeline
  private basicSkinnedTransparentPipeline: GPURenderPipeline
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

  // Shadow config
  private shadowEnabled: boolean
  private shadowResolution: number
  private shadowLambda: number
  private shadowBackExtend: number
  private shadowConstantBias: number
  private shadowSlopeBias: number
  private shadowBlendRange: number

  // Shadow GPU resources
  private shadowTexture: GPUTexture | null = null
  private shadowTextureView: GPUTextureView | null = null
  private shadowCascadeViews: GPUTextureView[] = []
  private shadowSampler!: GPUSampler
  private shadowUB!: GPUBuffer
  private shadowBGL!: GPUBindGroupLayout
  private shadowSkinnedBGL!: GPUBindGroupLayout
  private shadowPipeline!: GPURenderPipeline
  private shadowSkinnedPipeline!: GPURenderPipeline
  private shadowBG!: GPUBindGroup
  private dummyShadowTexture!: GPUTexture
  private dummyShadowTextureView!: GPUTextureView
  private _alignedShadowSize = 0

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
  private _lastMaterial: Material | null = null

  // Dynamic uniform buffers
  private _alignment: number
  private _alignedObjectSize: number
  private _alignedSkinnedSize: number
  private _objectDynBuf!: GPUBuffer
  private _skinnedDynBuf!: GPUBuffer
  private _objectDynBG!: GPUBindGroup
  private _skinnedDynBG!: GPUBindGroup
  private _objectBatchData!: Float32Array
  private _skinnedBatchData!: Float32Array
  private _objectCapacity: number
  private _skinnedCapacity: number

  // Traversal
  private _traversalStack: Node[] = []

  // Scratch matrices
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
  // Reusable typed arrays for uniform writes
  private _frameData = new Float32Array(FRAME_UB_SIZE / 4)
  private _materialData = new Float32Array(MATERIAL_UB_SIZE / 4)

  // Shadow scratch data
  private _shadowMeshes: Mesh[] = []
  private _cascadeVPs: Mat4[] = [mat4Create(), mat4Create(), mat4Create()]
  private _cascadeSplits = new Float32Array(NUM_CASCADES)
  private _shadowUBData!: Float32Array
  private _shadowLightView: Mat4 = mat4Create()
  private _shadowLightProj: Mat4 = mat4Create()
  private _frameNum = 0

  // Pre-allocated render pass descriptors (avoid per-frame heap allocations)
  private _shadowPassDescs: GPURenderPassDescriptor[] = []
  private _opaquePassCA0: GPURenderPassColorAttachment = null!
  private _opaquePassCA1: GPURenderPassColorAttachment = null!
  private _opaquePassDA: GPURenderPassDepthStencilAttachment = null!
  private _opaquePassDesc: GPURenderPassDescriptor = null!
  private _bloomDownPassCAs: GPURenderPassColorAttachment[] = []
  private _bloomDownPassDescs: GPURenderPassDescriptor[] = []
  private _bloomUpPassCAs: GPURenderPassColorAttachment[] = []
  private _bloomUpPassDescs: GPURenderPassDescriptor[] = []
  private _blitPassCA: GPURenderPassColorAttachment = null!
  private _blitPassDesc: GPURenderPassDescriptor = null!
  private _submitArr: GPUCommandBuffer[] = [null as unknown as GPUCommandBuffer]

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

  private constructor(
    device: GPUDevice,
    context: GPUCanvasContext,
    format: GPUTextureFormat,
    canvas: HTMLCanvasElement,
    lambertPipeline: GPURenderPipeline,
    basicPipeline: GPURenderPipeline,
    lambertSkinnedPipeline: GPURenderPipeline,
    basicSkinnedPipeline: GPURenderPipeline,
    lambertTransparentPipeline: GPURenderPipeline,
    basicTransparentPipeline: GPURenderPipeline,
    lambertSkinnedTransparentPipeline: GPURenderPipeline,
    basicSkinnedTransparentPipeline: GPURenderPipeline,
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
    shadowEnabled: boolean,
    shadowResolution: number,
    shadowLambda: number,
    shadowBackExtend: number,
    shadowConstantBias: number,
    shadowSlopeBias: number,
    shadowBlendRange: number,
    shadowBGL: GPUBindGroupLayout,
    shadowSkinnedBGL: GPUBindGroupLayout,
    shadowPipeline: GPURenderPipeline,
    shadowSkinnedPipeline: GPURenderPipeline,
    shadowUB: GPUBuffer,
    shadowSampler: GPUSampler,
    dummyShadowTexture: GPUTexture,
    shadowTexture: GPUTexture | null,
  ) {
    this.device = device
    this.context = context
    this.format = format
    this.canvas = canvas
    this.lambertPipeline = lambertPipeline
    this.basicPipeline = basicPipeline
    this.lambertSkinnedPipeline = lambertSkinnedPipeline
    this.basicSkinnedPipeline = basicSkinnedPipeline
    this.lambertTransparentPipeline = lambertTransparentPipeline
    this.basicTransparentPipeline = basicTransparentPipeline
    this.lambertSkinnedTransparentPipeline = lambertSkinnedTransparentPipeline
    this.basicSkinnedTransparentPipeline = basicSkinnedTransparentPipeline
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

    // Shadow config
    this.shadowEnabled = shadowEnabled
    this.shadowResolution = shadowResolution
    this.shadowLambda = shadowLambda
    this.shadowBackExtend = shadowBackExtend
    this.shadowConstantBias = shadowConstantBias
    this.shadowSlopeBias = shadowSlopeBias
    this.shadowBlendRange = shadowBlendRange
    this.shadowBGL = shadowBGL
    this.shadowSkinnedBGL = shadowSkinnedBGL
    this.shadowPipeline = shadowPipeline
    this.shadowSkinnedPipeline = shadowSkinnedPipeline
    this.shadowUB = shadowUB
    this.shadowSampler = shadowSampler
    this.dummyShadowTexture = dummyShadowTexture
    this.dummyShadowTextureView = dummyShadowTexture.createView()

    // Shadow textures
    if (shadowEnabled) {
      this.shadowTexture = shadowTexture
      this.shadowTextureView = shadowTexture!.createView({ dimension: '2d-array' })
      this.shadowCascadeViews = []
      for (let i = 0; i < NUM_CASCADES; i++) {
        this.shadowCascadeViews.push(
          shadowTexture!.createView({ dimension: '2d', baseArrayLayer: i, arrayLayerCount: 1 }),
        )
      }
      for (let i = 0; i < NUM_CASCADES; i++) {
        this._shadowPassDescs.push({
          colorAttachments: [],
          depthStencilAttachment: {
            view: this.shadowCascadeViews[i]!,
            depthLoadOp: 'clear',
            depthClearValue: 1.0,
            depthStoreOp: 'store',
          },
        })
      }
    }

    // Dynamic uniform buffer setup
    this._alignment = device.limits.minUniformBufferOffsetAlignment
    this._alignedObjectSize = Math.ceil(OBJECT_UB_SIZE / this._alignment) * this._alignment
    this._alignedSkinnedSize = Math.ceil(SKINNED_OBJECT_UB_SIZE / this._alignment) * this._alignment
    this._alignedShadowSize = Math.ceil(SHADOW_UB_SIZE / this._alignment) * this._alignment
    this._shadowUBData = new Float32Array((this._alignedShadowSize * NUM_CASCADES) / 4)
    this._objectCapacity = 2048
    this._skinnedCapacity = 2048
    this.createDynamicBuffers()

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

    // Pre-allocate render pass descriptors (views updated in rebuildPostProcessBindGroups)
    this._opaquePassCA0 = {
      view: null!,
      resolveTarget: null!,
      loadOp: 'clear',
      storeOp: 'discard',
      clearValue: { r: 0, g: 0, b: 0, a: 1.0 },
    }
    this._opaquePassCA1 = {
      view: null!,
      resolveTarget: null!,
      loadOp: 'clear',
      storeOp: 'discard',
      clearValue: { r: 0, g: 0, b: 0, a: 1.0 },
    }
    this._opaquePassDA = { view: null!, depthLoadOp: 'clear', depthClearValue: 1.0, depthStoreOp: 'store' }
    this._opaquePassDesc = {
      colorAttachments: [this._opaquePassCA0, this._opaquePassCA1],
      depthStencilAttachment: this._opaquePassDA,
    }

    for (let i = 0; i < this.bloomLevels; i++) {
      const ca: GPURenderPassColorAttachment = {
        view: null!,
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }
      this._bloomDownPassCAs.push(ca)
      this._bloomDownPassDescs.push({ colorAttachments: [ca] })
    }
    for (let i = 0; i < Math.max(0, this.bloomLevels - 1); i++) {
      const ca: GPURenderPassColorAttachment = { view: null!, loadOp: 'load', storeOp: 'store' }
      this._bloomUpPassCAs.push(ca)
      this._bloomUpPassDescs.push({ colorAttachments: [ca] })
    }
    this._blitPassCA = { view: null!, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }
    this._blitPassDesc = { colorAttachments: [this._blitPassCA] }

    // Create frame bind group with shadow texture + sampler
    const shadowTexView = this.shadowTextureView ?? this.dummyShadowTextureView
    this.frameBG = device.createBindGroup({
      layout: frameBGL,
      entries: [
        { binding: 0, resource: { buffer: frameUB } },
        { binding: 1, resource: shadowTexView },
        { binding: 2, resource: shadowSampler },
      ],
    })

    // Shadow bind group (dynamic offset per cascade)
    this.shadowBG = device.createBindGroup({
      layout: shadowBGL,
      entries: [{ binding: 0, resource: { buffer: shadowUB, size: SHADOW_UB_SIZE } }],
    })

    // Cache canvas dimensions
    this._displayW = canvas.clientWidth
    this._displayH = canvas.clientHeight
    this._resizeObserver = new ResizeObserver(() => {
      this._displayW = this.canvas.clientWidth
      this._displayH = this.canvas.clientHeight
    })
    this._resizeObserver.observe(canvas)
  }

  static async create(canvas: HTMLCanvasElement, config: RendererConfig = {}): Promise<WebGPURenderer> {
    if (!navigator.gpu) throw new Error('WebGPU not supported')

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
    if (!adapter) throw new Error('No WebGPU adapter found')

    const device = await adapter.requestDevice()

    // Surface hidden GPU errors (critical for Android debugging)
    device.onuncapturederror = event => {
      console.error('WebGPU uncaptured error:', event.error.message || event.error)
    }

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

    // ─── Shadow config ────────────────────────────────────────────
    let shadowEnabled: boolean
    let shadowResolution: number
    let shadowLambda: number
    let shadowBackExtend: number
    let shadowConstantBias: number
    let shadowSlopeBias: number
    let shadowBlendRange: number

    const shadowConfig = config.shadows
    if (!shadowConfig) {
      shadowEnabled = false
      shadowResolution = 1024
      shadowLambda = 0.7
      shadowBackExtend = 75
      shadowConstantBias = 0.001
      shadowSlopeBias = 0.005
      shadowBlendRange = 0.1
    } else if (typeof shadowConfig === 'object') {
      shadowEnabled = shadowConfig.enabled !== false
      shadowResolution = shadowConfig.resolution ?? 1024
      shadowLambda = shadowConfig.lambda ?? 0.7
      shadowBackExtend = shadowConfig.backExtend ?? 75
      shadowConstantBias = shadowConfig.constantBias ?? 0.001
      shadowSlopeBias = shadowConfig.slopeBias ?? 0.005
      shadowBlendRange = shadowConfig.blendRange ?? 0.1
    } else {
      shadowEnabled = true
      shadowResolution = 1024
      shadowLambda = 0.7
      shadowBackExtend = 75
      shadowConstantBias = 0.001
      shadowSlopeBias = 0.005
      shadowBlendRange = 0.1
    }

    // ─── Shader modules ────────────────────────────────────────────
    const lambertModule = device.createShaderModule({ code: LAMBERT_WGSL })
    const basicModule = device.createShaderModule({ code: BASIC_WGSL })
    const lambertSkinnedModule = device.createShaderModule({ code: LAMBERT_SKINNED_WGSL })
    const basicSkinnedModule = device.createShaderModule({ code: BASIC_SKINNED_WGSL })
    const shadowDepthModule = device.createShaderModule({ code: SHADOW_DEPTH_WGSL })
    const shadowDepthSkinnedModule = device.createShaderModule({ code: SHADOW_DEPTH_SKINNED_WGSL })
    const bloomDownModule = device.createShaderModule({ code: BLOOM_DOWN_WGSL })
    const bloomUpModule = device.createShaderModule({ code: BLOOM_UP_WGSL })
    const blitModule = device.createShaderModule({ code: BLIT_WGSL })

    // ─── Bind group layouts ────────────────────────────────────────
    const frameBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'depth', viewDimension: '2d-array' },
        },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
      ],
    })

    const materialBGL = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
    })

    const objectBGL = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform', hasDynamicOffset: true } }],
    })

    const skinnedObjectBGL = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: SKINNED_OBJECT_UB_SIZE },
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

    // ─── Shadow bind group layouts + pipeline layouts ───────────────
    const shadowBGL = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: 'uniform', hasDynamicOffset: true },
        },
      ],
    })

    const shadowPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [shadowBGL, objectBGL],
    })

    const shadowSkinnedBGL = shadowBGL
    const shadowSkinnedPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [shadowBGL, skinnedObjectBGL],
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

    // ─── Transparent pipeline variants (premultiplied alpha blend + no depth write + no cull) ──
    // Premultiplied alpha blend – avoids the 'src-alpha' blend factor which triggers
    // VK_ERROR_UNKNOWN on some Android Vulkan drivers when combined with comparison sampling.
    // Shaders output premultiplied color (rgb * alpha) so 'one' is correct for src factor.
    const transparentBlend: GPUBlendState = {
      color: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
      alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
    }
    const transparentTargets: GPUColorTargetState[] = [
      { format: 'rgba8unorm', blend: transparentBlend },
      { format: 'rgba16float', blend: transparentBlend },
    ]

    const lambertTransparentPipeline = device.createRenderPipeline({
      layout: opaquePipelineLayout,
      vertex: { module: lambertModule, entryPoint: 'vs_main', buffers: VERTEX_BUFFER_LAYOUT },
      fragment: { module: lambertModule, entryPoint: 'fs_main', targets: transparentTargets },
      primitive: { topology: 'triangle-list', cullMode: 'none', frontFace: 'ccw' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less-equal' },
      multisample: { count: samples },
    })

    const basicTransparentPipeline = device.createRenderPipeline({
      layout: opaquePipelineLayout,
      vertex: { module: basicModule, entryPoint: 'vs_main', buffers: VERTEX_BUFFER_LAYOUT },
      fragment: { module: basicModule, entryPoint: 'fs_main', targets: transparentTargets },
      primitive: { topology: 'triangle-list', cullMode: 'none', frontFace: 'ccw' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less-equal' },
      multisample: { count: samples },
    })

    const lambertSkinnedTransparentPipeline = device.createRenderPipeline({
      layout: skinnedOpaquePipelineLayout,
      vertex: { module: lambertSkinnedModule, entryPoint: 'vs_main', buffers: SKINNED_VERTEX_BUFFER_LAYOUT },
      fragment: { module: lambertSkinnedModule, entryPoint: 'fs_main', targets: transparentTargets },
      primitive: { topology: 'triangle-list', cullMode: 'none', frontFace: 'ccw' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less-equal' },
      multisample: { count: samples },
    })

    const basicSkinnedTransparentPipeline = device.createRenderPipeline({
      layout: skinnedOpaquePipelineLayout,
      vertex: { module: basicSkinnedModule, entryPoint: 'vs_main', buffers: SKINNED_VERTEX_BUFFER_LAYOUT },
      fragment: { module: basicSkinnedModule, entryPoint: 'fs_main', targets: transparentTargets },
      primitive: { topology: 'triangle-list', cullMode: 'none', frontFace: 'ccw' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less-equal' },
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

    // ─── Shadow pipelines (depth-only, front-face culling) ─────────
    const shadowPipeline = device.createRenderPipeline({
      layout: shadowPipelineLayout,
      vertex: { module: shadowDepthModule, entryPoint: 'vs_main', buffers: SHADOW_VERTEX_BUFFER_LAYOUT },
      primitive: { topology: 'triangle-list', cullMode: 'front', frontFace: 'ccw' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less-equal' },
    })

    const shadowSkinnedPipeline = device.createRenderPipeline({
      layout: shadowSkinnedPipelineLayout,
      vertex: {
        module: shadowDepthSkinnedModule,
        entryPoint: 'vs_main',
        buffers: SHADOW_SKINNED_VERTEX_BUFFER_LAYOUT,
      },
      primitive: { topology: 'triangle-list', cullMode: 'front', frontFace: 'ccw' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less-equal' },
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

    // ─── Shadow resources ──────────────────────────────────────────
    const alignment = device.limits.minUniformBufferOffsetAlignment
    const alignedShadowSize = Math.ceil(SHADOW_UB_SIZE / alignment) * alignment
    const shadowUB = device.createBuffer({
      size: alignedShadowSize * NUM_CASCADES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    const shadowSampler = device.createSampler({
      compare: 'less',
      magFilter: 'linear',
      minFilter: 'linear',
    })

    // Dummy 1×1×3 depth texture for when shadows are disabled
    const dummyShadowTexture = device.createTexture({
      size: [1, 1, NUM_CASCADES],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    // Clear the dummy shadow texture layers to depth 1.0
    for (let i = 0; i < NUM_CASCADES; i++) {
      const view = dummyShadowTexture.createView({ dimension: '2d', baseArrayLayer: i, arrayLayerCount: 1 })
      const enc = device.createCommandEncoder()
      enc
        .beginRenderPass({
          colorAttachments: [],
          depthStencilAttachment: {
            view,
            depthLoadOp: 'clear',
            depthClearValue: 1.0,
            depthStoreOp: 'store',
          },
        })
        .end()
      device.queue.submit([enc.finish()])
    }

    // Real shadow texture (only when shadows enabled)
    let shadowTexture: GPUTexture | null = null
    if (shadowEnabled) {
      shadowTexture = device.createTexture({
        size: [shadowResolution, shadowResolution, NUM_CASCADES],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      })
    }

    const renderer = new WebGPURenderer(
      device,
      context,
      format,
      canvas,
      lambertPipeline,
      basicPipeline,
      lambertSkinnedPipeline,
      basicSkinnedPipeline,
      lambertTransparentPipeline,
      basicTransparentPipeline,
      lambertSkinnedTransparentPipeline,
      basicSkinnedTransparentPipeline,
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
      shadowEnabled,
      shadowResolution,
      shadowLambda,
      shadowBackExtend,
      shadowConstantBias,
      shadowSlopeBias,
      shadowBlendRange,
      shadowBGL,
      shadowSkinnedBGL,
      shadowPipeline,
      shadowSkinnedPipeline,
      shadowUB,
      shadowSampler,
      dummyShadowTexture,
      shadowTexture,
    )
    renderer.maxDpr = config.maxDpr === false ? Infinity : (config.maxDpr ?? defaultMaxDpr())
    return renderer
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

    // Update pre-allocated render pass descriptor views
    this._opaquePassCA0.view = rt.msaaColorView
    this._opaquePassCA0.resolveTarget = rt.resolvedColorView
    this._opaquePassCA1.view = rt.msaaEmissiveView
    this._opaquePassCA1.resolveTarget = rt.resolvedEmissiveView
    this._opaquePassDA.view = rt.msaaDepthView

    for (let i = 0; i < this.bloomLevels; i++) {
      this._bloomDownPassCAs[i]!.view = rt.bloomViews[i]!
    }
    for (let i = 0; i < this.bloomUpBGs.length; i++) {
      const targetIdx = this.bloomLevels - 2 - i
      this._bloomUpPassCAs[i]!.view = rt.bloomViews[targetIdx]!
    }
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

    // Pack normals: float32x3 → snorm8x4 (12 → 4 bytes/vertex)
    const packedNormals = packNormalsSnorm8(geometry.normals, geometry.vertexCount)
    const normalBuf = d.createBuffer({
      size: packedNormals.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    d.queue.writeBuffer(normalBuf, 0, packedNormals.buffer, packedNormals.byteOffset, packedNormals.byteLength)

    // Pack UVs: float32x2 → float16x2 (8 → 4 bytes/vertex)
    const packedUVs = geometry.uvs ? packUVsFloat16(geometry.uvs) : new Uint16Array(geometry.vertexCount * 2)
    const uvBuf = d.createBuffer({
      size: packedUVs.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    d.queue.writeBuffer(uvBuf, 0, packedUVs.buffer, packedUVs.byteOffset, packedUVs.byteLength)

    // Material indices: convert Uint8 to Float32 (WebGPU has no uint8 scalar vertex format,
    // so we can't match WebGL2's raw Uint8 upload — the smallest integer format is uint8x2)
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

    // Pack weights: float32x4 → unorm8x4 (16 → 4 bytes/vertex)
    let weightsBuf: GPUBuffer | undefined
    if (geometry.weights) {
      const packedWeights = packWeightsUnorm8(geometry.weights)
      weightsBuf = d.createBuffer({
        size: Math.max(packedWeights.byteLength, 4),
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      })
      d.queue.writeBuffer(weightsBuf, 0, packedWeights.buffer, packedWeights.byteOffset, packedWeights.byteLength)
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
    data[5] = material.receiveShadow ? 1.0 : 0.0

    if (hasPalette && material.palette) {
      for (let i = 0; i < 32; i++) {
        const entry: PaletteEntry = material.palette[i] ?? { color: [1, 1, 1] }
        const offset = 8 + i * 8
        data[offset] = entry.color[0]
        data[offset + 1] = entry.color[1]
        data[offset + 2] = entry.color[2]
        data[offset + 3] = 1.0
        data[offset + 4] = entry.emissive?.[0] ?? 0
        data[offset + 5] = entry.emissive?.[1] ?? 0
        data[offset + 6] = entry.emissive?.[2] ?? 0
        data[offset + 7] = entry.emissiveIntensity ?? 0
      }
    }

    this.device.queue.writeBuffer(cache.buffer, 0, data.buffer, data.byteOffset, data.byteLength)
  }

  private createDynamicBuffers() {
    const d = this.device
    this._objectDynBuf = d.createBuffer({
      size: this._objectCapacity * this._alignedObjectSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this._objectBatchData = new Float32Array((this._objectCapacity * this._alignedObjectSize) / 4)
    this._objectDynBG = d.createBindGroup({
      layout: this.objectBGL,
      entries: [{ binding: 0, resource: { buffer: this._objectDynBuf, size: OBJECT_UB_SIZE } }],
    })

    this._skinnedDynBuf = d.createBuffer({
      size: this._skinnedCapacity * this._alignedSkinnedSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this._skinnedBatchData = new Float32Array((this._skinnedCapacity * this._alignedSkinnedSize) / 4)
    this._skinnedDynBG = d.createBindGroup({
      layout: this.skinnedObjectBGL,
      entries: [{ binding: 0, resource: { buffer: this._skinnedDynBuf, size: SKINNED_OBJECT_UB_SIZE } }],
    })
  }

  private ensureDynamicCapacity(count: number, type: 'object' | 'skinned') {
    if (type === 'object') {
      if (count <= this._objectCapacity) return
      this._objectDynBuf.destroy()
      this._objectCapacity = Math.max(count, this._objectCapacity * 2)
      this._objectDynBuf = this.device.createBuffer({
        size: this._objectCapacity * this._alignedObjectSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
      this._objectBatchData = new Float32Array((this._objectCapacity * this._alignedObjectSize) / 4)
      this._objectDynBG = this.device.createBindGroup({
        layout: this.objectBGL,
        entries: [{ binding: 0, resource: { buffer: this._objectDynBuf, size: OBJECT_UB_SIZE } }],
      })
    } else {
      if (count <= this._skinnedCapacity) return
      this._skinnedDynBuf.destroy()
      this._skinnedCapacity = Math.max(count, this._skinnedCapacity * 2)
      this._skinnedDynBuf = this.device.createBuffer({
        size: this._skinnedCapacity * this._alignedSkinnedSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
      this._skinnedBatchData = new Float32Array((this._skinnedCapacity * this._alignedSkinnedSize) / 4)
      this._skinnedDynBG = this.device.createBindGroup({
        layout: this.skinnedObjectBGL,
        entries: [{ binding: 0, resource: { buffer: this._skinnedDynBuf, size: SKINNED_OBJECT_UB_SIZE } }],
      })
    }
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
      mat4OrthoZO,
    )
  }

  // ─── Render ──────────────────────────────────────────────────────

  render(scene: Scene, camera: PerspectiveCamera) {
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
    camera.updateProjection('zero-to-one') // WebGPU clip space

    // Update scene graph
    scene.updateGraph()

    // View-projection matrix
    mat4Multiply(this._vpMatrix, camera._projectionMatrix, camera._viewMatrix)

    // Camera frustum
    frustumFromViewProjection(this._frustumPlanes, this._vpMatrix)

    // Find directional light (quick early-exit traversal)
    const dirLight = findDirectionalLight(scene, this._traversalStack)

    // Compute light direction
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
      frustumFromViewProjection(this._shadowFrustumPlanes, this._cascadeVPs[NUM_CASCADES - 1]!)
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
    let objIdx = 0
    let skinnedIdx = 0
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

    this.ensureDynamicCapacity(objCount, 'object')
    this.ensureDynamicCapacity(skinnedCount, 'skinned')

    const objBatch = this._objectBatchData
    const skinnedBatch = this._skinnedBatchData

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

    // Write dynamic buffers (camera-visible + shadow-only)
    if (objIdx > 0) {
      const byteLen = objIdx * this._alignedObjectSize
      this.device.queue.writeBuffer(this._objectDynBuf, 0, objBatch.buffer, objBatch.byteOffset, byteLen)
    }
    if (skinnedIdx > 0) {
      const byteLen = skinnedIdx * this._alignedSkinnedSize
      this.device.queue.writeBuffer(this._skinnedDynBuf, 0, skinnedBatch.buffer, skinnedBatch.byteOffset, byteLen)
    }

    // ─── Write shadow UBO (all cascades at once) ─────────────────
    if (shadowActive) {
      const alignedFloats = this._alignedShadowSize / 4
      const sud = this._shadowUBData
      sud.fill(0)
      for (let c = 0; c < NUM_CASCADES; c++) {
        sud.set(this._cascadeVPs[c]!, c * alignedFloats)
      }
      this.device.queue.writeBuffer(
        this.shadowUB,
        0,
        sud.buffer,
        sud.byteOffset,
        this._alignedShadowSize * NUM_CASCADES,
      )
    }

    // ─── Update per-frame uniform buffer (336 bytes) ─────────────
    const fd = this._frameData
    fd.fill(0)
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
    fd[27] = shadowActive ? 1.0 : 0.0
    if (shadowActive) {
      fd.set(this._cascadeVPs[0]!, 28)
      fd.set(this._cascadeVPs[1]!, 44)
      fd.set(this._cascadeVPs[2]!, 60)
      fd[76] = this._cascadeSplits[0]!
      fd[77] = this._cascadeSplits[1]!
      fd[78] = this._cascadeSplits[2]!
      fd[80] = this.shadowConstantBias
      fd[81] = this.shadowSlopeBias
      fd[82] = 1.0 / this.shadowResolution
      fd[83] = this.shadowBlendRange
    }
    this.device.queue.writeBuffer(this.frameUB, 0, fd.buffer, fd.byteOffset, fd.byteLength)

    // ─── Ensure render targets ──────────────────────────────────
    this.ensureRenderTargets()

    const encoder = this.device.createCommandEncoder()

    // ─── Shadow render pass (3 cascades, depth-only) ─────────────
    if (shadowActive) {
      for (let c = 0; c < NUM_CASCADES; c++) {
        const shadowPass = encoder.beginRenderPass(this._shadowPassDescs[c]!)

        // Per-cascade light frustum culling: project mesh center, skip if outside
        const cvp = this._cascadeVPs[c]!
        const PAD = 0.3

        // Draw camera-visible shadow casters
        for (let si = 0; si < meshes.length; si++) {
          const mesh = meshes[sortedIndices[si]!]!
          if (!mesh.castShadow) continue

          // Light-space frustum cull using mesh world center
          const wm = mesh._worldMatrix
          const wx = wm[12]!,
            wy = wm[13]!,
            wz = wm[14]!
          const lx = cvp[0]! * wx + cvp[4]! * wy + cvp[8]! * wz + cvp[12]!
          const ly = cvp[1]! * wx + cvp[5]! * wy + cvp[9]! * wz + cvp[13]!
          const lz = cvp[2]! * wx + cvp[6]! * wy + cvp[10]! * wz + cvp[14]!
          if (lx < -(1 + PAD) || lx > 1 + PAD || ly < -(1 + PAD) || ly > 1 + PAD || lz < -PAD || lz > 1 + PAD) continue

          const geoBufs = this.ensureGeometryBuffers(mesh.geometry)

          if (mesh._isSkinned) {
            shadowPass.setPipeline(this.shadowSkinnedPipeline)
            shadowPass.setVertexBuffer(0, geoBufs.position)
            shadowPass.setVertexBuffer(1, geoBufs.joints!)
            shadowPass.setVertexBuffer(2, geoBufs.weights!)
            shadowPass.setBindGroup(1, this._skinnedDynBG, [mesh._batchIndex * this._alignedSkinnedSize])
          } else {
            shadowPass.setPipeline(this.shadowPipeline)
            shadowPass.setVertexBuffer(0, geoBufs.position)
            shadowPass.setBindGroup(1, this._objectDynBG, [mesh._batchIndex * this._alignedObjectSize])
          }

          shadowPass.setBindGroup(0, this.shadowBG, [c * this._alignedShadowSize])
          shadowPass.setIndexBuffer(geoBufs.index, geoBufs.indexFormat)
          shadowPass.drawIndexed(geoBufs.indexCount)
          shadowDrawCalls++
        }

        // Draw shadow-only casters
        for (let i = 0; i < shadowMeshes.length; i++) {
          const mesh = shadowMeshes[i]!

          // Light-space frustum cull
          const wm = mesh._worldMatrix
          const wx = wm[12]!,
            wy = wm[13]!,
            wz = wm[14]!
          const lx = cvp[0]! * wx + cvp[4]! * wy + cvp[8]! * wz + cvp[12]!
          const ly = cvp[1]! * wx + cvp[5]! * wy + cvp[9]! * wz + cvp[13]!
          const lz = cvp[2]! * wx + cvp[6]! * wy + cvp[10]! * wz + cvp[14]!
          if (lx < -(1 + PAD) || lx > 1 + PAD || ly < -(1 + PAD) || ly > 1 + PAD || lz < -PAD || lz > 1 + PAD) continue

          const geoBufs = this.ensureGeometryBuffers(mesh.geometry)

          if (mesh._isSkinned) {
            shadowPass.setPipeline(this.shadowSkinnedPipeline)
            shadowPass.setVertexBuffer(0, geoBufs.position)
            shadowPass.setVertexBuffer(1, geoBufs.joints!)
            shadowPass.setVertexBuffer(2, geoBufs.weights!)
            shadowPass.setBindGroup(1, this._skinnedDynBG, [mesh._batchIndex * this._alignedSkinnedSize])
          } else {
            shadowPass.setPipeline(this.shadowPipeline)
            shadowPass.setVertexBuffer(0, geoBufs.position)
            shadowPass.setBindGroup(1, this._objectDynBG, [mesh._batchIndex * this._alignedObjectSize])
          }

          shadowPass.setBindGroup(0, this.shadowBG, [c * this._alignedShadowSize])
          shadowPass.setIndexBuffer(geoBufs.index, geoBufs.indexFormat)
          shadowPass.drawIndexed(geoBufs.indexCount)
          shadowDrawCalls++
        }

        shadowPass.end()
      }
    }

    // ─── Scene pass (MSAA MRT): opaque then transparent ─────────
    const scenePass = encoder.beginRenderPass(this._opaquePassDesc)

    // Find the split between opaque and transparent meshes
    const transparentStart = findTransparentStart(this._sortState, meshes.length)

    // ─── Draw helper (shared by opaque + transparent loops) ─────
    const drawMeshGPU = (pass: GPURenderPassEncoder, mesh: Mesh, pipeline: GPURenderPipeline) => {
      pass.setPipeline(pipeline)

      const geoBufs = this.ensureGeometryBuffers(mesh.geometry)
      pass.setVertexBuffer(0, geoBufs.position)
      pass.setVertexBuffer(1, geoBufs.normal)
      pass.setVertexBuffer(2, geoBufs.uv)
      pass.setVertexBuffer(3, geoBufs.materialIndex)
      if (mesh._isSkinned && geoBufs.joints && geoBufs.weights) {
        pass.setVertexBuffer(4, geoBufs.joints)
        pass.setVertexBuffer(5, geoBufs.weights)
      }
      pass.setIndexBuffer(geoBufs.index, geoBufs.indexFormat)

      const matCache = this.ensureMaterialCache(mesh.material)
      if (mesh.material !== this._lastMaterial) {
        this._lastMaterial = mesh.material
        this.writeMaterialBuffer(mesh.material, matCache)
      }

      if (mesh._isSkinned) {
        pass.setBindGroup(2, this._skinnedDynBG, [mesh._batchIndex * this._alignedSkinnedSize])
      } else {
        pass.setBindGroup(2, this._objectDynBG, [mesh._batchIndex * this._alignedObjectSize])
      }

      pass.setBindGroup(0, this.frameBG)
      pass.setBindGroup(1, matCache.bindGroup)

      pass.drawIndexed(geoBufs.indexCount)
      drawCalls++
      triangles += geoBufs.indexCount / 3
    }

    // ─── Opaque draw loop ────────────────────────────────────────
    this._lastMaterial = null
    for (let si = 0; si < transparentStart; si++) {
      const mesh = meshes[sortedIndices[si]!]!
      let pipeline: GPURenderPipeline
      if (mesh._isSkinned) {
        pipeline = mesh.material.type === 'lambert' ? this.lambertSkinnedPipeline : this.basicSkinnedPipeline
      } else {
        pipeline = mesh.material.type === 'lambert' ? this.lambertPipeline : this.basicPipeline
      }
      drawMeshGPU(scenePass, mesh, pipeline)
    }

    // ─── Transparent draw loop (back-to-front, blend + no depth write + no cull) ──
    for (let si = transparentStart; si < meshes.length; si++) {
      const mesh = meshes[sortedIndices[si]!]!
      let pipeline: GPURenderPipeline
      if (mesh._isSkinned) {
        pipeline =
          mesh.material.type === 'lambert'
            ? this.lambertSkinnedTransparentPipeline
            : this.basicSkinnedTransparentPipeline
      } else {
        pipeline = mesh.material.type === 'lambert' ? this.lambertTransparentPipeline : this.basicTransparentPipeline
      }
      drawMeshGPU(scenePass, mesh, pipeline)
    }

    scenePass.end()

    // ─── Bloom ──────────────────────────────────────────────────
    if (this.bloomEnabled && this.bloomLevels > 0) {
      // Downsample chain
      for (let i = 0; i < this.bloomLevels; i++) {
        const pass = encoder.beginRenderPass(this._bloomDownPassDescs[i]!)
        pass.setPipeline(this.bloomDownPipeline)
        pass.setBindGroup(0, this.bloomDownBGs[i]!)
        pass.draw(3)
        pass.end()
      }

      // Upsample chain (additive)
      for (let i = 0; i < this.bloomUpBGs.length; i++) {
        const pass = encoder.beginRenderPass(this._bloomUpPassDescs[i]!)
        pass.setPipeline(this.bloomUpPipeline)
        pass.setBindGroup(0, this.bloomUpBGs[i]!)
        pass.draw(3)
        pass.end()
      }
    }

    // ─── Final blit ─────────────────────────────────────────────
    const surfaceTexture = this.context.getCurrentTexture()
    this._blitPassCA.view = surfaceTexture.createView()
    const blitPass = encoder.beginRenderPass(this._blitPassDesc)
    blitPass.setPipeline(this.blitPipeline)
    blitPass.setBindGroup(0, this.blitBG!)
    blitPass.draw(3)
    blitPass.end()

    this._submitArr[0] = encoder.finish()
    this.device.queue.submit(this._submitArr)

    // Update stats
    this.stats.fps = this._currentFps
    this.stats.frameTime = dt
    this.stats.drawCalls = drawCalls
    this.stats.shadowDrawCalls = shadowDrawCalls
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
    this._objectDynBuf.destroy()
    this._skinnedDynBuf.destroy()
    for (const ub of this.bloomDownUBs) ub.destroy()
    for (const ub of this.bloomUpUBs) ub.destroy()
    this.shadowUB.destroy()
    this.shadowTexture?.destroy()
    this.dummyShadowTexture.destroy()
    this._resizeObserver?.disconnect()
    this._resizeObserver = null
    this.device.destroy()
  }
}
