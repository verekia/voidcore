// WebGPU Renderer – Renders the scene using the WebGPU API.
//
// This is the high-performance rendering backend. Each frame it:
//   1. Resizes the canvas to match display resolution (accounting for device pixel ratio)
//   2. Updates the camera's projection matrix
//   3. Walks the scene graph to collect visible meshes (with frustum culling)
//   4. Sorts meshes by material/pipeline to minimize GPU state changes
//   5. Uploads per-frame uniforms (view-projection matrix, light data) to the GPU
//   6. Batches all object transforms into dynamic uniform buffers (1-2 uploads vs N)
//   7. Renders a single shadow depth pass (with comparison sampling)
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
//   Outline       – Inverted hull outlines via combined geometry (doubled vertices with
//                    smooth normals) drawn in a single call with cullMode:none and
//                    front_facing discard in the fragment shader.
//   Side          – Per-material face culling control (front/back/double).
//   Premul alpha  – Transparent pipelines use premultiplied alpha blend (src=one, dst=
//                    one-minus-src-alpha). This avoids the 'src-alpha' blend factor which
//                    triggers VK_ERROR_UNKNOWN on some Android Vulkan drivers when combined
//                    with comparison texture sampling (textureSampleCompareLevel).
//   MRT           – Multiple render targets: color and emissive are written simultaneously.
//   Vertex colors – Meshes with baked vertex colors use separate VC pipelines.
//                    Color (unorm8x4) and emissive (float16x4) are vertex attributes.
//   Vertex packing – Normals use snorm8x4, UVs use float16x2, bone weights use unorm8x4,
//                    vertex colors use unorm8x4, emissive uses float16x4.
//
// WebGPURenderer.create()  – Async factory that initializes the GPU device and pipelines.
// WebGPURenderer.render()  – Draws one frame.
// WebGPURenderer.dispose() – Releases all GPU resources.

import { computeSmoothNormals } from '../geometry/geometry'
import {
  aabbCreate,
  frustumFromViewProjection,
  mat4Create,
  mat4Invert,
  mat4Multiply,
  mat4OrthoZO,
  mat4Transpose,
  vec3Create,
} from '../math/index'
import { Mesh } from '../scene/mesh'
import { Node } from '../scene/node'
import { packColorsUnorm8, packEmissiveFloat16, packNormalsSnorm8, packUVsFloat16, packWeightsUnorm8 } from './pack'
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
  LAMBERT_WGSL,
  LAMBERT_VC_WGSL,
  LAMBERT_SKINNED_VC_WGSL,
  LAMBERT_TEXTURED_WGSL,
  BASIC_WGSL,
  LAMBERT_SKINNED_WGSL,
  BASIC_SKINNED_WGSL,
  SHADOW_DEPTH_WGSL,
  SHADOW_DEPTH_SKINNED_WGSL,
  BLOOM_DOWN_WGSL,
  BLOOM_UP_WGSL,
  BLIT_WGSL,
} from './webgpu-shaders'

import type { Geometry } from '../geometry/geometry'
import type { Material } from '../materials/material'
import type { CompressedTextureFormat, Texture, TextureFormat } from '../materials/texture'
import type { AABB, Mat4, Vec3 } from '../math/index'
import type { PerspectiveCamera } from '../scene/camera'
import type { DirectionalLight } from '../scene/light'
import type { Scene } from '../scene/scene'
import type { Renderer, RendererConfig, FrameStats } from './renderer'
import type { SortState } from './sort'

// ─── Types ───────────────────────────────────────────────────────────

// Maps our TextureFormat enum to WebGPU GPUTextureFormat strings
const _toGPUTextureFormat = (fmt: TextureFormat): GPUTextureFormat => {
  switch (fmt) {
    case 'astc-4x4':
      return 'astc-4x4-unorm'
    case 'bc7':
      return 'bc7-rgba-unorm'
    case 'bc3':
      return 'bc3-rgba-unorm'
    case 'etc2-rgba8':
      return 'etc2-rgba8unorm'
    default:
      return 'rgba8unorm'
  }
}

interface GeoBufs {
  position: GPUBuffer
  normal: GPUBuffer
  uv: GPUBuffer
  index: GPUBuffer
  indexFormat: GPUIndexFormat
  indexCount: number
  baseIndexCount?: number
  color?: GPUBuffer
  emissive?: GPUBuffer
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

// FrameUniforms: mat4(64) + light(48) + pad(16) + shadowVP(64) + bias(16) = 208 bytes
// (WGSL layout is 192 bytes due to tighter packing, but we use 192 for the GPU buffer)
const FRAME_UB_SIZE = 192
// ShadowUniforms: mat4(64)
const SHADOW_UB_SIZE = 64
// ObjectUniforms: mat4(64) + mat4(64) + vec4(16) = 144 bytes
const OBJECT_UB_SIZE = 144
// MaterialUniforms: vec3+f32(16) + 4*f32(16) = 32 bytes
const MATERIAL_UB_SIZE = 32
// Bloom down params: vec2+f32+pad(16)
const BLOOM_DOWN_UB_SIZE = 16
// Bloom up params: vec2+pad(16)
const BLOOM_UP_UB_SIZE = 16
// Blit params: f32+pad(16)
const BLIT_UB_SIZE = 16

// SkinnedObjectUniforms: mat4(64) + mat4(64) + vec4(16) + 32*mat4(2048) = 2192 bytes
const SKINNED_OBJECT_UB_SIZE = 2192

// ─── Vertex buffer layout (shared by lambert + basic) ────────────────

const VERTEX_BUFFER_LAYOUT: GPUVertexBufferLayout[] = [
  { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' as GPUVertexFormat }] },
  { arrayStride: 4, attributes: [{ shaderLocation: 1, offset: 0, format: 'snorm8x4' as GPUVertexFormat }] },
  { arrayStride: 4, attributes: [{ shaderLocation: 2, offset: 0, format: 'float16x2' as GPUVertexFormat }] },
]

const SKINNED_VERTEX_BUFFER_LAYOUT: GPUVertexBufferLayout[] = [
  ...VERTEX_BUFFER_LAYOUT,
  { arrayStride: 4, attributes: [{ shaderLocation: 3, offset: 0, format: 'uint8x4' as GPUVertexFormat }] },
  { arrayStride: 4, attributes: [{ shaderLocation: 4, offset: 0, format: 'unorm8x4' as GPUVertexFormat }] },
]

// Vertex-color Lambert: pos + normal + uv + color + emissive
const VC_VERTEX_BUFFER_LAYOUT: GPUVertexBufferLayout[] = [
  ...VERTEX_BUFFER_LAYOUT,
  { arrayStride: 4, attributes: [{ shaderLocation: 3, offset: 0, format: 'unorm8x4' as GPUVertexFormat }] },
  { arrayStride: 8, attributes: [{ shaderLocation: 6, offset: 0, format: 'float16x4' as GPUVertexFormat }] },
]

// Vertex-color Lambert skinned: pos + normal + uv + color + joints + weights + emissive
const VC_SKINNED_VERTEX_BUFFER_LAYOUT: GPUVertexBufferLayout[] = [
  ...VERTEX_BUFFER_LAYOUT,
  { arrayStride: 4, attributes: [{ shaderLocation: 3, offset: 0, format: 'unorm8x4' as GPUVertexFormat }] },
  { arrayStride: 4, attributes: [{ shaderLocation: 4, offset: 0, format: 'uint8x4' as GPUVertexFormat }] },
  { arrayStride: 4, attributes: [{ shaderLocation: 5, offset: 0, format: 'unorm8x4' as GPUVertexFormat }] },
  { arrayStride: 8, attributes: [{ shaderLocation: 6, offset: 0, format: 'float16x4' as GPUVertexFormat }] },
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
  readonly compressedTextureFormats: readonly CompressedTextureFormat[]

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
  // Textured pipeline variants
  private lambertTexturedPipeline!: GPURenderPipeline
  private lambertTexturedTransparentPipeline!: GPURenderPipeline
  // Vertex-color pipeline variants
  private lambertVCPipeline!: GPURenderPipeline
  private lambertSkinnedVCPipeline!: GPURenderPipeline
  private lambertVCTransparentPipeline!: GPURenderPipeline
  private lambertSkinnedVCTransparentPipeline!: GPURenderPipeline

  // Pipeline cache for per-material side (cull mode) variants
  // Key: basePipelineId << 2 | cullModeId (0=back, 1=front, 2=none)
  private _pipelineCache = new Map<number, GPURenderPipeline>()

  // Bind group layouts
  private frameBGL: GPUBindGroupLayout
  private materialBGL: GPUBindGroupLayout
  private objectBGL: GPUBindGroupLayout
  private skinnedObjectBGL: GPUBindGroupLayout
  private postProcessBGL: GPUBindGroupLayout
  private blitBGL: GPUBindGroupLayout
  private textureBGL!: GPUBindGroupLayout

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
  shadowsBaked = false
  private _shadowIsBaked = false
  private _prevShadowsBaked = false

  // Shadow GPU resources
  private shadowTexture: GPUTexture | null = null
  private shadowTextureView: GPUTextureView | null = null
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
  private _outlineGeoCache = new WeakMap<Geometry, GeoBufs>()
  private _matCache = new WeakMap<Material, MatCache>()
  private _gpuTexCache = new WeakMap<Texture, { texture: GPUTexture; view: GPUTextureView }>()
  private _texBGCache = new WeakMap<Material, GPUBindGroup>()
  private _dummyWhiteTexView!: GPUTextureView
  private _linearSampler!: GPUSampler
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
  private _worldAABB: AABB = aabbCreate()
  private _lightDir = vec3Create()
  private _tempVec3 = vec3Create()
  private _meshes: Mesh[] = []
  private _sortState: SortState = createSortState(4096)
  // Reusable typed arrays for uniform writes
  private _frameData = new Float32Array(FRAME_UB_SIZE / 4)
  private _materialData = new Float32Array(MATERIAL_UB_SIZE / 4)

  // Shadow scratch data
  private _shadowMeshes: Mesh[] = []
  private _shadowVP: Mat4 = mat4Create()
  private _shadowUBData!: Float32Array
  private _shadowLightView: Mat4 = mat4Create()
  private _shadowLightProj: Mat4 = mat4Create()
  private _frameNum = 0

  // Pre-allocated render pass descriptors (avoid per-frame heap allocations)
  private _shadowPassDesc: GPURenderPassDescriptor = null!
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
    shadowBGL: GPUBindGroupLayout,
    shadowSkinnedBGL: GPUBindGroupLayout,
    shadowPipeline: GPURenderPipeline,
    shadowSkinnedPipeline: GPURenderPipeline,
    shadowUB: GPUBuffer,
    shadowSampler: GPUSampler,
    dummyShadowTexture: GPUTexture,
    shadowTexture: GPUTexture | null,
    compressedFormats: readonly CompressedTextureFormat[],
  ) {
    this.compressedTextureFormats = compressedFormats
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
      this.shadowTextureView = shadowTexture!.createView()
      this._shadowPassDesc = {
        colorAttachments: [],
        depthStencilAttachment: {
          view: this.shadowTextureView!,
          depthLoadOp: 'clear',
          depthClearValue: 1.0,
          depthStoreOp: 'store',
        },
      }
    }

    // Dynamic uniform buffer setup
    this._alignment = device.limits.minUniformBufferOffsetAlignment
    this._alignedObjectSize = Math.ceil(OBJECT_UB_SIZE / this._alignment) * this._alignment
    this._alignedSkinnedSize = Math.ceil(SKINNED_OBJECT_UB_SIZE / this._alignment) * this._alignment
    this._alignedShadowSize = Math.ceil(SHADOW_UB_SIZE / this._alignment) * this._alignment
    this._shadowUBData = new Float32Array(this._alignedShadowSize / 4)
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

    // Shadow bind group
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

    // Initialize textured pipeline resources
    this._initTexturedPipelines()
    // Initialize vertex-color pipeline variants
    this._initVCPipelines()
  }

  private _initTexturedPipelines() {
    const device = this.device

    // Bind group layout for texture maps (group 3)
    this.textureBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    })

    // Pipeline layout: frame + material + object + textures
    const texturedPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.frameBGL, this.materialBGL, this.objectBGL, this.textureBGL],
    })

    // Shader module
    const texturedModule = device.createShaderModule({ code: LAMBERT_TEXTURED_WGSL })

    // MRT targets (same as non-textured)
    const opaqueTargets: GPUColorTargetState[] = [{ format: 'rgba8unorm' }, { format: 'rgba16float' }]
    const transparentBlend: GPUBlendState = {
      color: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
      alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
    }
    const transparentTargets: GPUColorTargetState[] = [
      { format: 'rgba8unorm', blend: transparentBlend },
      { format: 'rgba16float', blend: transparentBlend },
    ]

    // Opaque textured Lambert pipeline
    this.lambertTexturedPipeline = device.createRenderPipeline({
      layout: texturedPipelineLayout,
      vertex: { module: texturedModule, entryPoint: 'vs_main', buffers: VERTEX_BUFFER_LAYOUT },
      fragment: { module: texturedModule, entryPoint: 'fs_main', targets: opaqueTargets },
      primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less-equal' },
      multisample: { count: this.samples },
    })

    // Transparent textured Lambert pipeline
    this.lambertTexturedTransparentPipeline = device.createRenderPipeline({
      layout: texturedPipelineLayout,
      vertex: { module: texturedModule, entryPoint: 'vs_main', buffers: VERTEX_BUFFER_LAYOUT },
      fragment: { module: texturedModule, entryPoint: 'fs_main', targets: transparentTargets },
      primitive: { topology: 'triangle-list', cullMode: 'none', frontFace: 'ccw' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less-equal' },
      multisample: { count: this.samples },
    })

    // 1x1 white dummy texture for missing maps (sampling white = identity)
    const dummyWhiteTex = device.createTexture({
      size: [1, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    device.queue.writeTexture(
      { texture: dummyWhiteTex },
      new Uint8Array([255, 255, 255, 255]),
      { bytesPerRow: 4 },
      [1, 1],
    )
    this._dummyWhiteTexView = dummyWhiteTex.createView()

    // Linear sampler for texture maps
    this._linearSampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    })
  }

  private _initVCPipelines() {
    const device = this.device

    const vcModule = device.createShaderModule({ code: LAMBERT_VC_WGSL })
    const vcSkinnedModule = device.createShaderModule({ code: LAMBERT_SKINNED_VC_WGSL })

    const vcPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.frameBGL, this.materialBGL, this.objectBGL, this.textureBGL],
    })
    const vcSkinnedPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.frameBGL, this.materialBGL, this.skinnedObjectBGL, this.textureBGL],
    })

    const opaqueTargets: GPUColorTargetState[] = [{ format: 'rgba8unorm' }, { format: 'rgba16float' }]
    const transparentBlend: GPUBlendState = {
      color: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
      alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
    }
    const transparentTargets: GPUColorTargetState[] = [
      { format: 'rgba8unorm', blend: transparentBlend },
      { format: 'rgba16float', blend: transparentBlend },
    ]

    this.lambertVCPipeline = device.createRenderPipeline({
      layout: vcPipelineLayout,
      vertex: { module: vcModule, entryPoint: 'vs_main', buffers: VC_VERTEX_BUFFER_LAYOUT },
      fragment: { module: vcModule, entryPoint: 'fs_main', targets: opaqueTargets },
      primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less-equal' },
      multisample: { count: this.samples },
    })

    this.lambertSkinnedVCPipeline = device.createRenderPipeline({
      layout: vcSkinnedPipelineLayout,
      vertex: { module: vcSkinnedModule, entryPoint: 'vs_main', buffers: VC_SKINNED_VERTEX_BUFFER_LAYOUT },
      fragment: { module: vcSkinnedModule, entryPoint: 'fs_main', targets: opaqueTargets },
      primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less-equal' },
      multisample: { count: this.samples },
    })

    this.lambertVCTransparentPipeline = device.createRenderPipeline({
      layout: vcPipelineLayout,
      vertex: { module: vcModule, entryPoint: 'vs_main', buffers: VC_VERTEX_BUFFER_LAYOUT },
      fragment: { module: vcModule, entryPoint: 'fs_main', targets: transparentTargets },
      primitive: { topology: 'triangle-list', cullMode: 'none', frontFace: 'ccw' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less-equal' },
      multisample: { count: this.samples },
    })

    this.lambertSkinnedVCTransparentPipeline = device.createRenderPipeline({
      layout: vcSkinnedPipelineLayout,
      vertex: { module: vcSkinnedModule, entryPoint: 'vs_main', buffers: VC_SKINNED_VERTEX_BUFFER_LAYOUT },
      fragment: { module: vcSkinnedModule, entryPoint: 'fs_main', targets: transparentTargets },
      primitive: { topology: 'triangle-list', cullMode: 'none', frontFace: 'ccw' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less-equal' },
      multisample: { count: this.samples },
    })
  }

  /**
   * Get or create a pipeline variant with a specific cull mode.
   * Pipeline types:
   *   0=lambert, 1=basic, 2=lambertSkinned, 3=basicSkinned, 4=lambertTextured,
   *   5=lambertTransparent, 6=basicTransparent, 7=lambertSkinnedTransparent,
   *   8=basicSkinnedTransparent, 9=lambertTexturedTransparent,
   *   10=lambertVC, 11=lambertSkinnedVC, 12=lambertVCTransparent, 13=lambertSkinnedVCTransparent
   * Cull modes: 0=back, 1=front, 2=none
   */
  private _getSidePipeline(pipelineType: number, cullMode: GPUCullMode): GPURenderPipeline {
    const cullId = cullMode === 'back' ? 0 : cullMode === 'front' ? 1 : 2
    const key = pipelineType * 3 + cullId
    let cached = this._pipelineCache.get(key)
    if (cached) return cached

    const device = this.device
    const isTransparent = pipelineType >= 5 && pipelineType <= 9 ? true : pipelineType === 12 || pipelineType === 13
    const isSkinned =
      pipelineType === 2 ||
      pipelineType === 3 ||
      pipelineType === 7 ||
      pipelineType === 8 ||
      pipelineType === 11 ||
      pipelineType === 13
    const isTextured = pipelineType === 4 || pipelineType === 9
    const isVC = pipelineType >= 10

    // Determine shader code
    let shaderCode: string
    if (isVC) {
      shaderCode = isSkinned ? LAMBERT_SKINNED_VC_WGSL : LAMBERT_VC_WGSL
    } else if (isTextured) {
      shaderCode = LAMBERT_TEXTURED_WGSL
    } else {
      const isLambert = pipelineType === 0 || pipelineType === 2 || pipelineType === 5 || pipelineType === 7
      if (isSkinned) {
        shaderCode = isLambert ? LAMBERT_SKINNED_WGSL : BASIC_SKINNED_WGSL
      } else {
        shaderCode = isLambert ? LAMBERT_WGSL : BASIC_WGSL
      }
    }

    // Pipeline layout
    const bindGroupLayouts =
      isTextured || isVC
        ? [this.frameBGL, this.materialBGL, isSkinned ? this.skinnedObjectBGL : this.objectBGL, this.textureBGL]
        : [this.frameBGL, this.materialBGL, isSkinned ? this.skinnedObjectBGL : this.objectBGL]
    const layout = device.createPipelineLayout({ bindGroupLayouts })

    // Vertex buffers
    let buffers: GPUVertexBufferLayout[]
    if (isVC) {
      buffers = isSkinned ? VC_SKINNED_VERTEX_BUFFER_LAYOUT : VC_VERTEX_BUFFER_LAYOUT
    } else {
      buffers = isSkinned ? SKINNED_VERTEX_BUFFER_LAYOUT : VERTEX_BUFFER_LAYOUT
    }

    // Fragment targets
    let targets: GPUColorTargetState[]
    if (isTransparent) {
      const blend: GPUBlendState = {
        color: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
        alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
      }
      targets = [
        { format: 'rgba8unorm', blend },
        { format: 'rgba16float', blend },
      ]
    } else {
      targets = [{ format: 'rgba8unorm' }, { format: 'rgba16float' }]
    }

    const module = device.createShaderModule({ code: shaderCode })
    cached = device.createRenderPipeline({
      layout,
      vertex: { module, entryPoint: 'vs_main', buffers },
      fragment: { module, entryPoint: 'fs_main', targets },
      primitive: { topology: 'triangle-list', cullMode, frontFace: 'ccw' },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: !isTransparent,
        depthCompare: 'less-equal',
      },
      multisample: { count: this.samples },
    })

    this._pipelineCache.set(key, cached)
    return cached
  }

  private _ensureGPUTexture(tex: Texture): { texture: GPUTexture; view: GPUTextureView } {
    const cached = this._gpuTexCache.get(tex)
    if (cached) return cached

    const gpuFormat = _toGPUTextureFormat(tex.format)
    const isCompressed = tex.format !== 'rgba8'

    const gpuTex = this.device.createTexture({
      size: [tex.width, tex.height],
      format: gpuFormat,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })

    if (isCompressed) {
      // Compressed: 4×4 blocks, 16 bytes per block for all supported formats
      const blocksX = Math.ceil(tex.width / 4)
      this.device.queue.writeTexture(
        { texture: gpuTex },
        tex.data.buffer as ArrayBuffer,
        { bytesPerRow: blocksX * 16, rowsPerImage: tex.height },
        [tex.width, tex.height],
      )
    } else {
      this.device.queue.writeTexture(
        { texture: gpuTex },
        tex.data.buffer as ArrayBuffer,
        { bytesPerRow: tex.width * 4, rowsPerImage: tex.height },
        [tex.width, tex.height],
      )
    }

    const entry = { texture: gpuTex, view: gpuTex.createView() }
    this._gpuTexCache.set(tex, entry)
    return entry
  }

  private _ensureTextureBindGroup(material: Material): GPUBindGroup {
    const cached = this._texBGCache.get(material)
    if (cached) return cached

    const colorView = material.colorMap ? this._ensureGPUTexture(material.colorMap).view : this._dummyWhiteTexView
    const aoView = material.aoMap ? this._ensureGPUTexture(material.aoMap).view : this._dummyWhiteTexView

    const bg = this.device.createBindGroup({
      layout: this.textureBGL,
      entries: [
        { binding: 0, resource: colorView },
        { binding: 1, resource: aoView },
        { binding: 2, resource: this._linearSampler },
      ],
    })
    this._texBGCache.set(material, bg)
    return bg
  }

  static async create(canvas: HTMLCanvasElement, config: RendererConfig = {}): Promise<WebGPURenderer> {
    if (!navigator.gpu) throw new Error('WebGPU not supported')

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
    if (!adapter) throw new Error('No WebGPU adapter found')

    // Request compressed texture features if available on this device
    const requiredFeatures: GPUFeatureName[] = []
    if (adapter.features.has('texture-compression-astc')) requiredFeatures.push('texture-compression-astc')
    if (adapter.features.has('texture-compression-bc')) requiredFeatures.push('texture-compression-bc')
    if (adapter.features.has('texture-compression-etc2')) requiredFeatures.push('texture-compression-etc2')

    const device = await adapter.requestDevice({ requiredFeatures })

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

    const shadowConfig = config.shadows
    if (!shadowConfig) {
      shadowEnabled = false
      shadowResolution = 2048
    } else if (typeof shadowConfig === 'object') {
      shadowEnabled = shadowConfig.enabled !== false
      shadowResolution = shadowConfig.resolution ?? 2048
    } else {
      shadowEnabled = true
      shadowResolution = 2048
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
          texture: { sampleType: 'depth', viewDimension: '2d' },
        },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
      ],
    })

    const materialBGL = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
    })

    const objectBGL = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform', hasDynamicOffset: true },
        },
      ],
    })

    const skinnedObjectBGL = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
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
    const shadowUB = device.createBuffer({
      size: SHADOW_UB_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    const shadowSampler = device.createSampler({
      compare: 'less',
      magFilter: 'linear',
      minFilter: 'linear',
    })

    // Dummy 1×1 depth texture for when shadows are disabled
    const dummyShadowTexture = device.createTexture({
      size: [1, 1],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    // Clear the dummy shadow texture to depth 1.0
    {
      const enc = device.createCommandEncoder()
      enc
        .beginRenderPass({
          colorAttachments: [],
          depthStencilAttachment: {
            view: dummyShadowTexture.createView(),
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
        size: [shadowResolution, shadowResolution],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      })
    }

    // Build supported compressed texture format list (priority order for loader selection)
    const compressedFormats: CompressedTextureFormat[] = []
    if (device.features.has('texture-compression-astc')) compressedFormats.push('astc-4x4')
    if (device.features.has('texture-compression-bc')) compressedFormats.push('bc7')
    if (device.features.has('texture-compression-etc2')) compressedFormats.push('etc2-rgba8')
    if (device.features.has('texture-compression-bc')) compressedFormats.push('bc3')

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
      shadowBGL,
      shadowSkinnedBGL,
      shadowPipeline,
      shadowSkinnedPipeline,
      shadowUB,
      shadowSampler,
      dummyShadowTexture,
      shadowTexture,
      compressedFormats,
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

    // Vertex colors (unorm8x4) and emissive (float16x4) — only for baked-palette meshes
    let colorBuf: GPUBuffer | undefined
    let emissiveBuf: GPUBuffer | undefined
    if (geometry.colors) {
      const packedColors = packColorsUnorm8(geometry.colors, geometry.vertexCount)
      colorBuf = d.createBuffer({
        size: Math.max(packedColors.byteLength, 4),
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      })
      d.queue.writeBuffer(colorBuf, 0, packedColors.buffer, packedColors.byteOffset, packedColors.byteLength)

      const packedEmissive = packEmissiveFloat16(
        geometry.emissiveColors ?? new Float32Array(geometry.vertexCount * 4),
        geometry.vertexCount,
      )
      emissiveBuf = d.createBuffer({
        size: Math.max(packedEmissive.byteLength, 4),
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      })
      d.queue.writeBuffer(emissiveBuf, 0, packedEmissive.buffer, packedEmissive.byteOffset, packedEmissive.byteLength)
    }

    // Index buffer (size must be 4-byte aligned for WebGPU writeBuffer)
    const indexFormat: GPUIndexFormat = geometry.indices instanceof Uint32Array ? 'uint32' : 'uint16'
    const indexByteLength = geometry.indices.byteLength
    const indexAlignedSize = (indexByteLength + 3) & ~3
    const indexBuf = d.createBuffer({
      size: indexAlignedSize,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    })
    if (indexByteLength === indexAlignedSize) {
      d.queue.writeBuffer(indexBuf, 0, geometry.indices.buffer, geometry.indices.byteOffset, indexByteLength)
    } else {
      // Copy to a 4-byte-aligned buffer for Uint16Array with odd element count
      const padded = new Uint8Array(indexAlignedSize)
      padded.set(new Uint8Array(geometry.indices.buffer, geometry.indices.byteOffset, indexByteLength))
      d.queue.writeBuffer(indexBuf, 0, padded.buffer, 0, indexAlignedSize)
    }

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
      cached.index.destroy()
      if (cached.color) cached.color.destroy()
      if (cached.emissive) cached.emissive.destroy()
      if (cached.joints) cached.joints.destroy()
      if (cached.weights) cached.weights.destroy()
    }

    const bufs: GeoBufs = {
      position: positionBuf,
      normal: normalBuf,
      uv: uvBuf,
      index: indexBuf,
      indexFormat,
      indexCount: geometry.indexCount,
      color: colorBuf,
      emissive: emissiveBuf,
      joints: jointsBuf,
      weights: weightsBuf,
    }
    this._geoCache.set(geometry, bufs)
    geometry.needsUpdate = false
    return bufs
  }

  private ensureOutlineGeometryBuffers(geometry: Geometry): GeoBufs {
    const cached = this._outlineGeoCache.get(geometry)
    if (cached && !geometry.needsUpdate) return cached

    // Ensure base geometry buffers exist
    this.ensureGeometryBuffers(geometry)

    const d = this.device
    const vc = geometry.vertexCount
    const ic = geometry.indexCount

    // Compute smooth normals for gap-free outline inflation
    const smoothNormals = geometry._smoothNormals ?? computeSmoothNormals(geometry)

    // Combined positions: [original, duplicated for outline]
    const combinedPos = new Float32Array(vc * 2 * 3)
    combinedPos.set(geometry.positions, 0)
    combinedPos.set(geometry.positions, vc * 3)
    const posBuf = d.createBuffer({
      size: combinedPos.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    d.queue.writeBuffer(posBuf, 0, combinedPos.buffer, combinedPos.byteOffset, combinedPos.byteLength)

    // Combined normals: [packed w=0, packed smooth w=127]
    const baseNormals = packNormalsSnorm8(geometry.normals, vc, 0)
    const outlineNormals = packNormalsSnorm8(smoothNormals, vc, 127)
    const combinedNorm = new Int8Array(vc * 2 * 4)
    combinedNorm.set(baseNormals, 0)
    combinedNorm.set(outlineNormals, vc * 4)
    const normBuf = d.createBuffer({
      size: combinedNorm.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    d.queue.writeBuffer(normBuf, 0, combinedNorm.buffer, combinedNorm.byteOffset, combinedNorm.byteLength)

    // Combined UVs: [original, duplicated]
    const baseUVs = geometry.uvs ? packUVsFloat16(geometry.uvs) : new Uint16Array(vc * 2)
    const combinedUV = new Uint16Array(vc * 2 * 2)
    combinedUV.set(baseUVs, 0)
    combinedUV.set(baseUVs, vc * 2)
    const uvBuf = d.createBuffer({
      size: combinedUV.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    d.queue.writeBuffer(uvBuf, 0, combinedUV.buffer, combinedUV.byteOffset, combinedUV.byteLength)

    // Combined vertex colors and emissive: [original, duplicated]
    let colorBuf: GPUBuffer | undefined
    let emissiveBuf: GPUBuffer | undefined
    if (geometry.colors) {
      const baseColors = packColorsUnorm8(geometry.colors, vc)
      const combinedColors = new Uint8Array(vc * 2 * 4)
      combinedColors.set(baseColors, 0)
      combinedColors.set(baseColors, vc * 4)
      colorBuf = d.createBuffer({
        size: combinedColors.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      })
      d.queue.writeBuffer(colorBuf, 0, combinedColors.buffer, combinedColors.byteOffset, combinedColors.byteLength)

      const baseEmissive = packEmissiveFloat16(geometry.emissiveColors ?? new Float32Array(vc * 4), vc)
      const combinedEmissive = new Uint16Array(vc * 2 * 4)
      combinedEmissive.set(baseEmissive, 0)
      combinedEmissive.set(baseEmissive, vc * 4)
      emissiveBuf = d.createBuffer({
        size: combinedEmissive.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      })
      d.queue.writeBuffer(
        emissiveBuf,
        0,
        combinedEmissive.buffer,
        combinedEmissive.byteOffset,
        combinedEmissive.byteLength,
      )
    }

    // Combined index buffer: [original CCW, reversed CW + vc offset]
    const use32 = vc * 2 > 65535 || geometry.indices instanceof Uint32Array
    const IndexArray = use32 ? Uint32Array : Uint16Array
    const combinedIdx = new IndexArray(ic * 2)
    // Original indices (CCW)
    for (let i = 0; i < ic; i++) combinedIdx[i] = geometry.indices[i]!
    // Same winding (CCW) + vertex offset for outline (front_facing discard handles silhouette)
    for (let i = 0; i < ic; i++) combinedIdx[ic + i] = geometry.indices[i]! + vc
    const indexByteLength = combinedIdx.byteLength
    const indexAlignedSize = (indexByteLength + 3) & ~3
    const idxBuf = d.createBuffer({ size: indexAlignedSize, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST })
    if (indexByteLength === indexAlignedSize) {
      d.queue.writeBuffer(idxBuf, 0, combinedIdx.buffer, combinedIdx.byteOffset, indexByteLength)
    } else {
      const padded = new Uint8Array(indexAlignedSize)
      padded.set(new Uint8Array(combinedIdx.buffer, combinedIdx.byteOffset, indexByteLength))
      d.queue.writeBuffer(idxBuf, 0, padded.buffer, 0, indexAlignedSize)
    }

    // Combined joints + weights for skinned meshes
    let jointsBuf: GPUBuffer | undefined
    if (geometry.joints) {
      const jointsU8 = geometry.joints instanceof Uint8Array ? geometry.joints : new Uint8Array(geometry.joints)
      const combinedJoints = new Uint8Array(vc * 2 * 4)
      combinedJoints.set(jointsU8, 0)
      combinedJoints.set(jointsU8, vc * 4)
      jointsBuf = d.createBuffer({
        size: combinedJoints.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      })
      d.queue.writeBuffer(jointsBuf, 0, combinedJoints.buffer, combinedJoints.byteOffset, combinedJoints.byteLength)
    }

    let weightsBuf: GPUBuffer | undefined
    if (geometry.weights) {
      const baseWeights = packWeightsUnorm8(geometry.weights)
      const combinedWeights = new Uint8Array(vc * 2 * 4)
      combinedWeights.set(baseWeights, 0)
      combinedWeights.set(baseWeights, vc * 4)
      weightsBuf = d.createBuffer({
        size: combinedWeights.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      })
      d.queue.writeBuffer(weightsBuf, 0, combinedWeights.buffer, combinedWeights.byteOffset, combinedWeights.byteLength)
    }

    // Destroy old combined buffers
    if (cached) {
      cached.position.destroy()
      cached.normal.destroy()
      cached.uv.destroy()
      cached.index.destroy()
      if (cached.color) cached.color.destroy()
      if (cached.emissive) cached.emissive.destroy()
      if (cached.joints) cached.joints.destroy()
      if (cached.weights) cached.weights.destroy()
    }

    const bufs: GeoBufs = {
      position: posBuf,
      normal: normBuf,
      uv: uvBuf,
      index: idxBuf,
      indexFormat: use32 ? 'uint32' : 'uint16',
      indexCount: ic * 2,
      baseIndexCount: ic,
      color: colorBuf,
      emissive: emissiveBuf,
      joints: jointsBuf,
      weights: weightsBuf,
    }
    this._outlineGeoCache.set(geometry, bufs)
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
    data[4] = material.receiveShadow ? 1.0 : 0.0
    data[5] = material.aoIntensity
    data[6] = material.emissiveBrightness

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

    // Find lights (quick early-exit traversals)
    const dirLight = findDirectionalLight(scene, this._traversalStack)
    const ambLight = findAmbientLight(scene, this._traversalStack)

    // Compute light direction
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
      camera.position,
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
    const camPos = camera.position
    for (let si = 0; si < meshes.length; si++) {
      const mesh = meshes[sortedIndices[si]!]!

      // Compute effective outline thickness (0 if beyond outlineMaxDistance)
      let thickness = mesh._outlineThickness
      if (thickness > 0) {
        const maxDist = mesh._outlineMaxDistance
        if (maxDist > 0) {
          const dx = mesh._worldMatrix[12]! - camPos[0]!
          const dy = mesh._worldMatrix[13]! - camPos[1]!
          const dz = mesh._worldMatrix[14]! - camPos[2]!
          if (dx * dx + dy * dy + dz * dz > maxDist * maxDist) thickness = 0
        }
      }

      if (mesh._isSkinned) {
        mesh.skeleton!.update()
        const off = skinnedIdx * alignedSkinnedFloats
        skinnedBatch.set(mesh._worldMatrix, off)
        if (mat4Invert(this._invWorldMatrix, mesh._worldMatrix)) {
          mat4Transpose(this._normalMatrix, this._invWorldMatrix)
        }
        skinnedBatch.set(this._normalMatrix, off + 16)
        const oc = mesh._outlineColor
        skinnedBatch[off + 32] = oc[0]
        skinnedBatch[off + 33] = oc[1]
        skinnedBatch[off + 34] = oc[2]
        skinnedBatch[off + 35] = thickness
        skinnedBatch.set(mesh.skeleton!.boneMatrices, off + 36)
        mesh._batchIndex = skinnedIdx++
      } else {
        const off = objIdx * alignedObjFloats
        objBatch.set(mesh._worldMatrix, off)
        if (mat4Invert(this._invWorldMatrix, mesh._worldMatrix)) {
          mat4Transpose(this._normalMatrix, this._invWorldMatrix)
        }
        objBatch.set(this._normalMatrix, off + 16)
        const oc = mesh._outlineColor
        objBatch[off + 32] = oc[0]
        objBatch[off + 33] = oc[1]
        objBatch[off + 34] = oc[2]
        objBatch[off + 35] = thickness
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
        skinnedBatch[off + 32] = 0
        skinnedBatch[off + 33] = 0
        skinnedBatch[off + 34] = 0
        skinnedBatch[off + 35] = 0
        skinnedBatch.set(mesh.skeleton!.boneMatrices, off + 36)
        mesh._batchIndex = skinnedIdx++
      } else {
        const off = objIdx * alignedObjFloats
        objBatch.set(mesh._worldMatrix, off)
        if (mat4Invert(this._invWorldMatrix, mesh._worldMatrix)) {
          mat4Transpose(this._normalMatrix, this._invWorldMatrix)
        }
        objBatch.set(this._normalMatrix, off + 16)
        objBatch[off + 32] = 0
        objBatch[off + 33] = 0
        objBatch[off + 34] = 0
        objBatch[off + 35] = 0
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

    // ─── Write shadow UBO ─────────────────────────────────────────
    if (shadowActive) {
      const sud = this._shadowUBData
      sud.fill(0)
      sud.set(this._shadowVP, 0)
      this.device.queue.writeBuffer(this.shadowUB, 0, sud.buffer, sud.byteOffset, SHADOW_UB_SIZE)
    }

    // ─── Update per-frame uniform buffer ──────────────────────────
    // WGSL layout (192 bytes / 48 floats):
    //   viewProjection: mat4x4     (floats 0-15)
    //   lightDir: vec3             (floats 16-18)
    //   lightIntensity: f32        (float 19)
    //   lightColor: vec3           (floats 20-22)
    //   ambientIntensity: f32      (float 23)
    //   ambientColor: vec3         (floats 24-26)
    //   shadowEnabled: f32         (float 27)
    //   shadowVP: mat4x4           (floats 28-43, offset 112 due to 16-byte alignment)
    //   constantBias: f32          (float 44)
    //   slopeBias: f32             (float 45)
    //   invMapSize: f32            (float 46)
    //   _pad0: f32                 (float 47)
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
    fd[23] = ambLight ? ambLight.intensity : 0
    fd[24] = ambLight ? ambLight.color[0] : 0
    fd[25] = ambLight ? ambLight.color[1] : 0
    fd[26] = ambLight ? ambLight.color[2] : 0
    fd[27] = shadowActive ? 1.0 : 0.0
    if (shadowActive) {
      fd.set(this._shadowVP, 28)
      fd[44] = dirLight!.shadowBias
      fd[45] = dirLight!.shadowSlopeBias
      fd[46] = 1.0 / this.shadowResolution
    }
    this.device.queue.writeBuffer(this.frameUB, 0, fd.buffer, fd.byteOffset, fd.byteLength)

    // ─── Ensure render targets ──────────────────────────────────
    this.ensureRenderTargets()

    const encoder = this.device.createCommandEncoder()

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
      const shadowPass = encoder.beginRenderPass(this._shadowPassDesc)

      // Light-space frustum culling: project mesh center, skip if outside
      const svp = this._shadowVP
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
        const lx = svp[0]! * wx + svp[4]! * wy + svp[8]! * wz + svp[12]!
        const ly = svp[1]! * wx + svp[5]! * wy + svp[9]! * wz + svp[13]!
        const lz = svp[2]! * wx + svp[6]! * wy + svp[10]! * wz + svp[14]!
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

        shadowPass.setBindGroup(0, this.shadowBG, [0])
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
        const lx = svp[0]! * wx + svp[4]! * wy + svp[8]! * wz + svp[12]!
        const ly = svp[1]! * wx + svp[5]! * wy + svp[9]! * wz + svp[13]!
        const lz = svp[2]! * wx + svp[6]! * wy + svp[10]! * wz + svp[14]!
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

        shadowPass.setBindGroup(0, this.shadowBG, [0])
        shadowPass.setIndexBuffer(geoBufs.index, geoBufs.indexFormat)
        shadowPass.drawIndexed(geoBufs.indexCount)
        shadowDrawCalls++
      }

      shadowPass.end()
      this._shadowIsBaked = true
    }

    // ─── Scene pass (MSAA MRT): opaque then transparent ─────────
    const scenePass = encoder.beginRenderPass(this._opaquePassDesc)

    // Find the split between opaque and transparent meshes
    const transparentStart = findTransparentStart(this._sortState, meshes.length)

    // ─── Draw helper (shared by opaque + transparent loops) ─────
    const drawMeshGPU = (pass: GPURenderPassEncoder, mesh: Mesh, pipeline: GPURenderPipeline) => {
      pass.setPipeline(pipeline)

      // Use combined outline buffers for outlined Lambert meshes
      const isOutlined = mesh._outlineThickness > 0 && mesh.material.type === 'lambert'
      const geoBufs = isOutlined
        ? this.ensureOutlineGeometryBuffers(mesh.geometry)
        : this.ensureGeometryBuffers(mesh.geometry)
      pass.setVertexBuffer(0, geoBufs.position)
      pass.setVertexBuffer(1, geoBufs.normal)
      pass.setVertexBuffer(2, geoBufs.uv)
      if (geoBufs.color) {
        pass.setVertexBuffer(3, geoBufs.color)
        if (mesh._isSkinned && geoBufs.joints && geoBufs.weights) {
          pass.setVertexBuffer(4, geoBufs.joints)
          pass.setVertexBuffer(5, geoBufs.weights)
          pass.setVertexBuffer(6, geoBufs.emissive!)
        } else {
          pass.setVertexBuffer(4, geoBufs.emissive!)
        }
      } else if (mesh._isSkinned && geoBufs.joints && geoBufs.weights) {
        pass.setVertexBuffer(3, geoBufs.joints)
        pass.setVertexBuffer(4, geoBufs.weights)
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

      // Determine cull mode from material side
      const side = mesh.material.side
      let cullMode: GPUCullMode = side === 'back' ? 'front' : side === 'double' ? 'none' : 'back'

      // Outlined Lambert meshes use cullMode:none (shader handles front-face discard)
      if (mesh._outlineThickness > 0 && mesh.material.type === 'lambert') cullMode = 'none'

      const hasVC = !!mesh.geometry.colors && mesh.material.type === 'lambert'

      let pipeline: GPURenderPipeline
      if (cullMode !== 'back') {
        // Non-default cull mode: use lazy pipeline cache
        let pipelineType: number
        if (hasVC) {
          pipelineType = mesh._isSkinned ? 11 : 10
        } else if (mesh._isSkinned) {
          pipelineType = mesh.material.type === 'lambert' ? 2 : 3
        } else if (mesh.material._hasTextures && mesh.material.type === 'lambert') {
          pipelineType = 4
        } else {
          pipelineType = mesh.material.type === 'lambert' ? 0 : 1
        }
        pipeline = this._getSidePipeline(pipelineType, cullMode)
      } else {
        // Default cull mode: use existing pipelines (fast path)
        if (hasVC) {
          pipeline = mesh._isSkinned ? this.lambertSkinnedVCPipeline : this.lambertVCPipeline
        } else if (mesh._isSkinned) {
          pipeline = mesh.material.type === 'lambert' ? this.lambertSkinnedPipeline : this.basicSkinnedPipeline
        } else if (mesh.material._hasTextures && mesh.material.type === 'lambert') {
          pipeline = this.lambertTexturedPipeline
        } else {
          pipeline = mesh.material.type === 'lambert' ? this.lambertPipeline : this.basicPipeline
        }
      }
      if (hasVC || (mesh.material._hasTextures && !mesh._isSkinned)) {
        scenePass.setBindGroup(3, this._ensureTextureBindGroup(mesh.material))
      }
      drawMeshGPU(scenePass, mesh, pipeline)
    }

    // ─── Transparent draw loop (back-to-front, blend + no depth write) ──
    for (let si = transparentStart; si < meshes.length; si++) {
      const mesh = meshes[sortedIndices[si]!]!

      // Determine cull mode from material side
      const side = mesh.material.side
      let cullMode: GPUCullMode = side === 'back' ? 'front' : side === 'double' ? 'none' : 'back'

      // Outlined Lambert meshes use cullMode:none (shader handles front-face discard)
      if (mesh._outlineThickness > 0 && mesh.material.type === 'lambert') cullMode = 'none'

      const hasVC = !!mesh.geometry.colors && mesh.material.type === 'lambert'

      let pipeline: GPURenderPipeline
      if (cullMode !== 'none') {
        // Non-default cull mode for transparent: use lazy pipeline cache
        let pipelineType: number
        if (hasVC) {
          pipelineType = mesh._isSkinned ? 13 : 12
        } else if (mesh._isSkinned) {
          pipelineType = mesh.material.type === 'lambert' ? 7 : 8
        } else if (mesh.material._hasTextures && mesh.material.type === 'lambert') {
          pipelineType = 9
        } else {
          pipelineType = mesh.material.type === 'lambert' ? 5 : 6
        }
        pipeline = this._getSidePipeline(pipelineType, cullMode)
      } else {
        // Default cull mode for transparent: use existing pipelines (fast path)
        if (hasVC) {
          pipeline = mesh._isSkinned ? this.lambertSkinnedVCTransparentPipeline : this.lambertVCTransparentPipeline
        } else if (mesh._isSkinned) {
          pipeline =
            mesh.material.type === 'lambert'
              ? this.lambertSkinnedTransparentPipeline
              : this.basicSkinnedTransparentPipeline
        } else if (mesh.material._hasTextures && mesh.material.type === 'lambert') {
          pipeline = this.lambertTexturedTransparentPipeline
        } else {
          pipeline = mesh.material.type === 'lambert' ? this.lambertTransparentPipeline : this.basicTransparentPipeline
        }
      }
      if (hasVC || (mesh.material._hasTextures && !mesh._isSkinned)) {
        scenePass.setBindGroup(3, this._ensureTextureBindGroup(mesh.material))
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
