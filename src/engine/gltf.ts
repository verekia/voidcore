import type { Geometry } from './geometry.ts'

export interface GLTFPrimitive {
  geometry: Geometry
  color?: [number, number, number, number]
  materialIndices?: Uint8Array
  skinJoints?: Uint8Array
  skinWeights?: Float32Array
  uvs?: Float32Array
}

export interface GLTFMesh {
  name?: string
  primitives: GLTFPrimitive[]
  skinIndex?: number
}

export interface GltfSkin {
  jointNodeIndices: number[]
  inverseBindMatrices: Float32Array
}

export interface GltfAnimationChannel {
  targetNodeIndex: number
  path: 'translation' | 'rotation' | 'scale'
  interpolation: string
  inputTimes: Float32Array
  outputValues: Float32Array
}

export interface GltfAnimation {
  name: string
  channels: GltfAnimationChannel[]
  duration: number
}

export interface GltfNodeTransform {
  name: string
  nodeIndex: number
  parentIndex: number
  translation: Float32Array
  rotation: Float32Array
  scale: Float32Array
}

export interface GLTFResult {
  meshes: GLTFMesh[]
  skins: GltfSkin[]
  animations: GltfAnimation[]
  nodeTransforms: GltfNodeTransform[]
}

export interface GLTFOptions {
  dracoDecoderPath?: string
  materialColors?: Map<number, [number, number, number]>
  materialBloom?: Map<number, number>
}

// glTF 2.0 JSON schema types (subset)
interface GLTFJson {
  accessors?: GLTFAccessor[]
  bufferViews?: GLTFBufferView[]
  buffers?: GLTFBuffer[]
  meshes?: GLTFJsonMesh[]
  materials?: GLTFMaterial[]
  extensionsUsed?: string[]
  nodes?: GLTFJsonNode[]
  skins?: GLTFJsonSkin[]
  animations?: GLTFJsonAnimation[]
  scenes?: GLTFJsonScene[]
  scene?: number
}

interface GLTFAccessor {
  bufferView?: number
  byteOffset?: number
  componentType: number
  count: number
  type: string
  max?: number[]
  min?: number[]
}

interface GLTFBufferView {
  buffer: number
  byteOffset?: number
  byteLength: number
  byteStride?: number
}

interface GLTFBuffer {
  uri?: string
  byteLength: number
}

interface GLTFJsonMesh {
  name?: string
  primitives: GLTFJsonPrimitive[]
}

interface GLTFJsonPrimitive {
  attributes: Record<string, number>
  indices?: number
  material?: number
  extensions?: {
    KHR_draco_mesh_compression?: {
      bufferView: number
      attributes: Record<string, number>
    }
  }
}

interface GLTFMaterial {
  pbrMetallicRoughness?: {
    baseColorFactor?: [number, number, number, number]
  }
}

interface GLTFJsonNode {
  name?: string
  mesh?: number
  skin?: number
  children?: number[]
  translation?: [number, number, number]
  rotation?: [number, number, number, number]
  scale?: [number, number, number]
}

interface GLTFJsonSkin {
  joints: number[]
  inverseBindMatrices?: number
  skeleton?: number
}

interface GLTFJsonAnimation {
  name?: string
  channels: GLTFJsonAnimChannel[]
  samplers: GLTFJsonAnimSampler[]
}

interface GLTFJsonAnimChannel {
  sampler: number
  target: { node: number; path: string }
}

interface GLTFJsonAnimSampler {
  input: number
  output: number
  interpolation?: string
}

interface GLTFJsonScene {
  nodes: number[]
}

// glTF component type constants
const GL_BYTE = 5120
const GL_UNSIGNED_BYTE = 5121
const GL_SHORT = 5122
const GL_UNSIGNED_SHORT = 5123
const GL_UNSIGNED_INT = 5125
const GL_FLOAT = 5126

// GLB constants
const GLB_MAGIC = 0x46546c67
const GLB_CHUNK_JSON = 0x4e4f534a
const GLB_CHUNK_BIN = 0x004e4942

// Draco decoder singleton
let dracoDecoderPromise: Promise<DracoModule> | null = null

interface DracoModule {
  Decoder: new () => DracoDecoder
  DecoderBuffer: new () => DracoDecoderBuffer
  Mesh: new () => DracoMesh
  TRIANGULAR_MESH: number
  DT_FLOAT32: number
  DT_UINT32: number
  _malloc(size: number): number
  _free(ptr: number): void
  HEAPF32: Float32Array
  HEAPU32: Uint32Array
  destroy(obj: unknown): void
}

interface DracoDecoder {
  GetEncodedGeometryType(buffer: DracoDecoderBuffer): number
  DecodeBufferToMesh(buffer: DracoDecoderBuffer, mesh: DracoMesh): { ok(): boolean }
  GetAttributeByUniqueId(mesh: DracoMesh, id: number): DracoAttribute
  GetAttributeDataArrayForAllPoints(
    mesh: DracoMesh,
    attr: DracoAttribute,
    dataType: number,
    byteLength: number,
    outPtr: number,
  ): boolean
  GetTrianglesUInt32Array(mesh: DracoMesh, byteLength: number, outPtr: number): boolean
}

interface DracoDecoderBuffer {
  Init(data: Int8Array, length: number): void
}

interface DracoMesh {
  num_points(): number
  num_faces(): number
}

interface DracoAttribute {
  num_components(): number
}

function initDracoDecoder(decoderPath: string): Promise<DracoModule> {
  if (dracoDecoderPromise) return dracoDecoderPromise

  dracoDecoderPromise = (async () => {
    const jsUrl = `${decoderPath}draco_decoder_gltf.js`
    const wasmUrl = `${decoderPath}draco_decoder_gltf.wasm`

    const [jsResponse, wasmResponse] = await Promise.all([fetch(jsUrl), fetch(wasmUrl)])
    const [jsText, wasmBinary] = await Promise.all([jsResponse.text(), wasmResponse.arrayBuffer()])

    // The Draco JS file defines a DracoDecoderModule factory
    const factory = new Function(`${jsText}; return DracoDecoderModule;`)()
    const module: DracoModule = await factory({ wasmBinary })
    return module
  })()

  return dracoDecoderPromise
}

function getComponentByteSize(componentType: number): number {
  switch (componentType) {
    case GL_BYTE:
    case GL_UNSIGNED_BYTE:
      return 1
    case GL_SHORT:
    case GL_UNSIGNED_SHORT:
      return 2
    case GL_UNSIGNED_INT:
    case GL_FLOAT:
      return 4
    default:
      throw new Error(`Unknown glTF component type: ${componentType}`)
  }
}

function getTypeElementCount(type: string): number {
  switch (type) {
    case 'SCALAR':
      return 1
    case 'VEC2':
      return 2
    case 'VEC3':
      return 3
    case 'VEC4':
      return 4
    case 'MAT2':
      return 4
    case 'MAT3':
      return 9
    case 'MAT4':
      return 16
    default:
      throw new Error(`Unknown glTF accessor type: ${type}`)
  }
}

function getTypedArray(
  data: ArrayBuffer,
  componentType: number,
  byteOffset: number,
  count: number,
): Float32Array | Uint16Array | Uint32Array | Int16Array | Int8Array | Uint8Array {
  switch (componentType) {
    case GL_FLOAT:
      return new Float32Array(data, byteOffset, count)
    case GL_UNSIGNED_INT:
      return new Uint32Array(data, byteOffset, count)
    case GL_UNSIGNED_SHORT:
      return new Uint16Array(data, byteOffset, count)
    case GL_SHORT:
      return new Int16Array(data, byteOffset, count)
    case GL_UNSIGNED_BYTE:
      return new Uint8Array(data, byteOffset, count)
    case GL_BYTE:
      return new Int8Array(data, byteOffset, count)
    default:
      throw new Error(`Unknown glTF component type: ${componentType}`)
  }
}

function resolveAccessor(
  json: GLTFJson,
  accessorIndex: number,
  buffers: ArrayBuffer[],
): {
  data: Float32Array | Uint16Array | Uint32Array | Int16Array | Int8Array | Uint8Array
  elementCount: number
  componentCount: number
} {
  const accessor = json.accessors![accessorIndex]!
  const bufferView = json.bufferViews![accessor.bufferView!]!
  const buffer = buffers[bufferView.buffer]!
  const byteOffset = (bufferView.byteOffset ?? 0) + (accessor.byteOffset ?? 0)
  const componentCount = getTypeElementCount(accessor.type)
  const totalElements = accessor.count * componentCount

  // Handle interleaved data (byteStride)
  const componentSize = getComponentByteSize(accessor.componentType)
  const naturalStride = componentCount * componentSize
  const stride = bufferView.byteStride ?? naturalStride

  if (stride !== naturalStride) {
    // Interleaved: must de-interleave
    const result = new Float32Array(totalElements)
    for (let i = 0; i < accessor.count; i++) {
      const srcOffset = byteOffset + i * stride
      const src = new Float32Array(buffer, srcOffset, componentCount)
      result.set(src, i * componentCount)
    }
    return { data: result, elementCount: accessor.count, componentCount }
  }

  const data = getTypedArray(buffer, accessor.componentType, byteOffset, totalElements)
  return { data, elementCount: accessor.count, componentCount }
}

function resolveAccessorFloat32(json: GLTFJson, accessorIndex: number, buffers: ArrayBuffer[]): Float32Array {
  const result = resolveAccessor(json, accessorIndex, buffers)
  return result.data instanceof Float32Array ? result.data : new Float32Array(result.data)
}

function resolveAccessorUint8(json: GLTFJson, accessorIndex: number, buffers: ArrayBuffer[]): Uint8Array {
  const accessor = json.accessors![accessorIndex]!
  const bufferView = json.bufferViews![accessor.bufferView!]!
  const buffer = buffers[bufferView.buffer]!
  const byteOffset = (bufferView.byteOffset ?? 0) + (accessor.byteOffset ?? 0)
  const componentCount = getTypeElementCount(accessor.type)
  const total = accessor.count * componentCount

  if (accessor.componentType === GL_UNSIGNED_BYTE) {
    return new Uint8Array(buffer, byteOffset, total)
  }
  // Unsigned short joints → downconvert to uint8
  if (accessor.componentType === GL_UNSIGNED_SHORT) {
    const u16 = new Uint16Array(buffer, byteOffset, total)
    const out = new Uint8Array(total)
    for (let i = 0; i < total; i++) out[i] = u16[i]!
    return out
  }
  return new Uint8Array(buffer, byteOffset, total)
}

function computeFlatNormals(positions: Float32Array, indices: Uint16Array | Uint32Array): Float32Array {
  const normals = new Float32Array(positions.length)

  for (let i = 0; i < indices.length; i += 3) {
    const i0 = indices[i]!
    const i1 = indices[i + 1]!
    const i2 = indices[i + 2]!

    const ax = positions[i0 * 3]!,
      ay = positions[i0 * 3 + 1]!,
      az = positions[i0 * 3 + 2]!
    const bx = positions[i1 * 3]!,
      by = positions[i1 * 3 + 1]!,
      bz = positions[i1 * 3 + 2]!
    const cx = positions[i2 * 3]!,
      cy = positions[i2 * 3 + 1]!,
      cz = positions[i2 * 3 + 2]!

    const e1x = bx - ax,
      e1y = by - ay,
      e1z = bz - az
    const e2x = cx - ax,
      e2y = cy - ay,
      e2z = cz - az

    const nx = e1y * e2z - e1z * e2y
    const ny = e1z * e2x - e1x * e2z
    const nz = e1x * e2y - e1y * e2x

    // Accumulate (handles shared vertices)
    for (const idx of [i0, i1, i2]) {
      const base = idx * 3
      normals[base] = normals[base]! + nx
      normals[base + 1] = normals[base + 1]! + ny
      normals[base + 2] = normals[base + 2]! + nz
    }
  }

  // Normalize
  for (let i = 0; i < normals.length; i += 3) {
    const nx = normals[i]!
    const ny = normals[i + 1]!
    const nz = normals[i + 2]!
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz)
    if (len > 0) {
      normals[i] = nx / len
      normals[i + 1] = ny / len
      normals[i + 2] = nz / len
    }
  }

  return normals
}

function interleaveVertices(
  positions: Float32Array,
  normals: Float32Array,
  vertexCount: number,
  materialIndices?: Uint8Array,
  materialColors?: Map<number, [number, number, number]>,
  materialBloom?: Map<number, number>,
): Float32Array {
  const vertices = new Float32Array(vertexCount * 10)
  for (let i = 0; i < vertexCount; i++) {
    const vi = i * 10
    vertices[vi] = positions[i * 3]!
    vertices[vi + 1] = positions[i * 3 + 1]!
    vertices[vi + 2] = positions[i * 3 + 2]!
    vertices[vi + 3] = normals[i * 3]!
    vertices[vi + 4] = normals[i * 3 + 1]!
    vertices[vi + 5] = normals[i * 3 + 2]!

    if (materialIndices && materialColors) {
      const idx = materialIndices[i]!
      const c = materialColors.get(idx) ?? [1, 1, 1]
      vertices[vi + 6] = c[0]
      vertices[vi + 7] = c[1]
      vertices[vi + 8] = c[2]
    } else {
      vertices[vi + 6] = 1
      vertices[vi + 7] = 1
      vertices[vi + 8] = 1
    }

    // Bloom value from material index
    if (materialIndices && materialBloom) {
      vertices[vi + 9] = materialBloom.get(materialIndices[i]!) ?? 0
    } else {
      vertices[vi + 9] = 0
    }
  }
  return vertices
}

async function decodeDracoPrimitive(
  json: GLTFJson,
  primitive: GLTFJsonPrimitive,
  buffers: ArrayBuffer[],
  decoderPath: string,
): Promise<{
  positions: Float32Array
  normals: Float32Array | null
  indices: Uint16Array | Uint32Array
  vertexCount: number
  materialIndices: Uint8Array | undefined
  skinJoints: Uint8Array | undefined
  skinWeights: Float32Array | undefined
  uvs: Float32Array | undefined
}> {
  const draco = await initDracoDecoder(decoderPath)
  const ext = primitive.extensions!.KHR_draco_mesh_compression!

  const bufferView = json.bufferViews![ext.bufferView]!
  const buffer = buffers[bufferView.buffer]!
  const byteOffset = bufferView.byteOffset ?? 0
  const compressedData = new Int8Array(buffer, byteOffset, bufferView.byteLength)

  const decoder = new draco.Decoder()
  const decoderBuffer = new draco.DecoderBuffer()
  decoderBuffer.Init(compressedData, compressedData.length)

  const geometryType = decoder.GetEncodedGeometryType(decoderBuffer)
  if (geometryType !== draco.TRIANGULAR_MESH) {
    draco.destroy(decoder)
    draco.destroy(decoderBuffer)
    throw new Error('Draco: expected triangular mesh')
  }

  const dracoMesh = new draco.Mesh()
  const status = decoder.DecodeBufferToMesh(decoderBuffer, dracoMesh)
  if (!status.ok()) {
    draco.destroy(decoder)
    draco.destroy(decoderBuffer)
    throw new Error('Draco: decode failed')
  }

  const vertexCount = dracoMesh.num_points()
  const faceCount = dracoMesh.num_faces()

  // Helper: extract attribute via direct WASM heap access (same approach as three.js)
  const extractFloat32Attribute = (attr: DracoAttribute): Float32Array => {
    const numComponents = attr.num_components()
    const numValues = vertexCount * numComponents
    const byteLength = numValues * Float32Array.BYTES_PER_ELEMENT
    const ptr = draco._malloc(byteLength)
    decoder.GetAttributeDataArrayForAllPoints(dracoMesh, attr, draco.DT_FLOAT32, byteLength, ptr)
    const result = new Float32Array(draco.HEAPF32.buffer, ptr, numValues).slice()
    draco._free(ptr)
    return result
  }

  // Extract positions
  const posAttr = decoder.GetAttributeByUniqueId(dracoMesh, ext.attributes.POSITION!)
  const positions = extractFloat32Attribute(posAttr)

  // Extract normals (if present)
  let normals: Float32Array | null = null
  if (ext.attributes.NORMAL !== undefined) {
    const normAttr = decoder.GetAttributeByUniqueId(dracoMesh, ext.attributes.NORMAL)
    normals = extractFloat32Attribute(normAttr)
  }

  // Extract _materialindex (if present)
  const matIdxKey = Object.keys(ext.attributes).find(k => k.toLowerCase() === '_materialindex')
  let materialIndices: Uint8Array | undefined
  if (matIdxKey !== undefined) {
    const matIdxAttr = decoder.GetAttributeByUniqueId(dracoMesh, ext.attributes[matIdxKey]!)
    const matIdxFloat = extractFloat32Attribute(matIdxAttr)
    materialIndices = new Uint8Array(vertexCount)
    for (let i = 0; i < vertexCount; i++) {
      materialIndices[i] = Math.round(matIdxFloat[i]!)
    }
  }

  // Extract UVs (TEXCOORD_0) if present
  let uvs: Float32Array | undefined
  const texCoordKey = Object.keys(ext.attributes).find(k => k === 'TEXCOORD_0')
  if (texCoordKey !== undefined) {
    const uvAttr = decoder.GetAttributeByUniqueId(dracoMesh, ext.attributes[texCoordKey]!)
    uvs = extractFloat32Attribute(uvAttr)
  }

  // Extract skin attributes (JOINTS_0 / WEIGHTS_0) if present
  let skinJoints: Uint8Array | undefined
  let skinWeights: Float32Array | undefined
  if (ext.attributes.JOINTS_0 !== undefined && ext.attributes.WEIGHTS_0 !== undefined) {
    const jointsAttr = decoder.GetAttributeByUniqueId(dracoMesh, ext.attributes.JOINTS_0)
    const jointsFloat = extractFloat32Attribute(jointsAttr)
    skinJoints = new Uint8Array(jointsFloat.length)
    for (let i = 0; i < jointsFloat.length; i++) {
      skinJoints[i] = Math.round(jointsFloat[i]!)
    }
    const weightsAttr = decoder.GetAttributeByUniqueId(dracoMesh, ext.attributes.WEIGHTS_0)
    skinWeights = extractFloat32Attribute(weightsAttr)
  }

  // Extract indices (always use uint32 then downcast — matches three.js approach)
  const indexCount = faceCount * 3
  const idxByteLength = indexCount * Uint32Array.BYTES_PER_ELEMENT
  const idxPtr = draco._malloc(idxByteLength)
  decoder.GetTrianglesUInt32Array(dracoMesh, idxByteLength, idxPtr)
  const rawIndices = new Uint32Array(draco.HEAPU32.buffer, idxPtr, indexCount).slice()
  draco._free(idxPtr)

  const indices: Uint16Array | Uint32Array = vertexCount <= 65535 ? new Uint16Array(rawIndices) : rawIndices

  draco.destroy(dracoMesh)
  draco.destroy(decoder)
  draco.destroy(decoderBuffer)

  return { positions, normals, indices, vertexCount, materialIndices, skinJoints, skinWeights, uvs }
}

function parsePrimitive(
  json: GLTFJson,
  primitive: GLTFJsonPrimitive,
  buffers: ArrayBuffer[],
): {
  positions: Float32Array
  normals: Float32Array | null
  indices: Uint16Array | Uint32Array
  vertexCount: number
  uvs: Float32Array | undefined
} {
  // Resolve indices
  if (primitive.indices === undefined) {
    throw new Error('glTF primitives without indices are not supported')
  }
  const indicesAccessor = resolveAccessor(json, primitive.indices, buffers)
  const rawIndices = indicesAccessor.data

  // Resolve positions
  const posAccessorIndex = primitive.attributes.POSITION
  if (posAccessorIndex === undefined) {
    throw new Error('glTF primitive missing POSITION attribute')
  }
  const posResult = resolveAccessor(json, posAccessorIndex, buffers)
  const positions = posResult.data instanceof Float32Array ? posResult.data : new Float32Array(posResult.data)
  const vertexCount = posResult.elementCount

  // Resolve normals (optional)
  let normals: Float32Array | null = null
  if (primitive.attributes.NORMAL !== undefined) {
    const normResult = resolveAccessor(json, primitive.attributes.NORMAL, buffers)
    normals = normResult.data instanceof Float32Array ? normResult.data : new Float32Array(normResult.data)
  }

  // Resolve UVs (optional)
  let uvs: Float32Array | undefined
  if (primitive.attributes.TEXCOORD_0 !== undefined) {
    uvs = resolveAccessorFloat32(json, primitive.attributes.TEXCOORD_0, buffers)
  }

  // Convert indices to appropriate type
  let indices: Uint16Array | Uint32Array
  if (vertexCount <= 65535) {
    indices = rawIndices instanceof Uint16Array ? rawIndices : new Uint16Array(rawIndices)
  } else {
    indices = rawIndices instanceof Uint32Array ? rawIndices : new Uint32Array(rawIndices)
  }

  return { positions, normals, indices, vertexCount, uvs }
}

function parseSkinAttributes(
  json: GLTFJson,
  primitive: GLTFJsonPrimitive,
  buffers: ArrayBuffer[],
): { skinJoints: Uint8Array; skinWeights: Float32Array } | null {
  if (primitive.attributes.JOINTS_0 === undefined || primitive.attributes.WEIGHTS_0 === undefined) {
    return null
  }
  const skinJoints = resolveAccessorUint8(json, primitive.attributes.JOINTS_0, buffers)
  const skinWeights = resolveAccessorFloat32(json, primitive.attributes.WEIGHTS_0, buffers)
  return { skinJoints, skinWeights }
}

export async function loadGLTF(url: string, options?: GLTFOptions): Promise<GLTFResult> {
  const decoderPath = options?.dracoDecoderPath ?? '/draco-1.5.7/'
  const materialColors = options?.materialColors
  const materialBloom = options?.materialBloom

  const response = await fetch(url)
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`)

  const arrayBuffer = await response.arrayBuffer()
  const dataView = new DataView(arrayBuffer)

  let json: GLTFJson
  let buffers: ArrayBuffer[]

  // Detect GLB by magic bytes
  const isGLB = dataView.byteLength >= 4 && dataView.getUint32(0, true) === GLB_MAGIC

  if (isGLB) {
    // Parse GLB header (12 bytes: magic + version + length)
    const version = dataView.getUint32(4, true)
    if (version !== 2) throw new Error(`Unsupported GLB version: ${version}`)

    // Parse chunks
    let offset = 12
    let jsonChunk: string | null = null
    let binChunk: ArrayBuffer | null = null

    while (offset < dataView.byteLength) {
      const chunkLength = dataView.getUint32(offset, true)
      const chunkType = dataView.getUint32(offset + 4, true)
      offset += 8

      if (chunkType === GLB_CHUNK_JSON) {
        jsonChunk = new TextDecoder().decode(new Uint8Array(arrayBuffer, offset, chunkLength))
      } else if (chunkType === GLB_CHUNK_BIN) {
        binChunk = arrayBuffer.slice(offset, offset + chunkLength)
      }

      offset += chunkLength
    }

    if (!jsonChunk) throw new Error('GLB missing JSON chunk')
    json = JSON.parse(jsonChunk) as GLTFJson
    buffers = binChunk ? [binChunk] : []
  } else {
    // Regular .gltf JSON
    const text = new TextDecoder().decode(arrayBuffer)
    json = JSON.parse(text) as GLTFJson

    // Fetch external buffer files
    const baseUrl = url.substring(0, url.lastIndexOf('/') + 1)
    buffers = []
    if (json.buffers) {
      for (const buf of json.buffers) {
        if (buf.uri) {
          if (buf.uri.startsWith('data:')) {
            // Data URI
            const base64 = buf.uri.split(',')[1]!
            const binary = atob(base64)
            const bytes = new Uint8Array(binary.length)
            for (let i = 0; i < binary.length; i++) {
              bytes[i] = binary.charCodeAt(i)
            }
            buffers.push(bytes.buffer)
          } else {
            const bufResponse = await fetch(baseUrl + buf.uri)
            buffers.push(await bufResponse.arrayBuffer())
          }
        }
      }
    }
  }

  const usesDraco = json.extensionsUsed?.includes('KHR_draco_mesh_compression') ?? false

  // ── Build node hierarchy ──────────────────────────────────────────
  const nodes = json.nodes ?? []
  const nodeTransforms: GltfNodeTransform[] = []
  const childToParent = new Map<number, number>()

  for (let ni = 0; ni < nodes.length; ni++) {
    const node = nodes[ni]!
    if (node.children) {
      for (const childIdx of node.children) {
        childToParent.set(childIdx, ni)
      }
    }
  }

  for (let ni = 0; ni < nodes.length; ni++) {
    const node = nodes[ni]!
    const t = new Float32Array(3)
    const r = new Float32Array([0, 0, 0, 1])
    const s = new Float32Array([1, 1, 1])
    if (node.translation) {
      t[0] = node.translation[0]
      t[1] = node.translation[1]
      t[2] = node.translation[2]
    }
    if (node.rotation) {
      r[0] = node.rotation[0]
      r[1] = node.rotation[1]
      r[2] = node.rotation[2]
      r[3] = node.rotation[3]
    }
    if (node.scale) {
      s[0] = node.scale[0]
      s[1] = node.scale[1]
      s[2] = node.scale[2]
    }
    nodeTransforms.push({
      name: node.name ?? '',
      nodeIndex: ni,
      parentIndex: childToParent.get(ni) ?? -1,
      translation: t,
      rotation: r,
      scale: s,
    })
  }

  // ── Map mesh index to skin index (via node references) ──────────
  const meshSkinMap = new Map<number, number>()
  for (const node of nodes) {
    if (node.mesh !== undefined && node.skin !== undefined) {
      meshSkinMap.set(node.mesh, node.skin)
    }
  }

  // ── Parse skins ───────────────────────────────────────────────────
  const skins: GltfSkin[] = []
  if (json.skins) {
    for (const skin of json.skins) {
      const jointNodeIndices: number[] = skin.joints
      let inverseBindMatrices: Float32Array
      if (skin.inverseBindMatrices !== undefined) {
        inverseBindMatrices = resolveAccessorFloat32(json, skin.inverseBindMatrices, buffers)
      } else {
        // Identity matrices
        inverseBindMatrices = new Float32Array(jointNodeIndices.length * 16)
        for (let j = 0; j < jointNodeIndices.length; j++) {
          const o = j * 16
          inverseBindMatrices[o] = 1
          inverseBindMatrices[o + 5] = 1
          inverseBindMatrices[o + 10] = 1
          inverseBindMatrices[o + 15] = 1
        }
      }
      skins.push({ jointNodeIndices, inverseBindMatrices })
    }
  }

  // ── Parse animations ──────────────────────────────────────────────
  const animations: GltfAnimation[] = []
  if (json.animations) {
    for (const anim of json.animations) {
      const channels: GltfAnimationChannel[] = []
      let duration = 0
      for (const ch of anim.channels) {
        const sampler = anim.samplers[ch.sampler]!
        const inputTimes = resolveAccessorFloat32(json, sampler.input, buffers)
        const outputValues = resolveAccessorFloat32(json, sampler.output, buffers)
        const maxTime = inputTimes[inputTimes.length - 1]!
        if (maxTime > duration) duration = maxTime
        channels.push({
          targetNodeIndex: ch.target.node,
          path: ch.target.path as 'translation' | 'rotation' | 'scale',
          interpolation: sampler.interpolation ?? 'LINEAR',
          inputTimes,
          outputValues,
        })
      }
      animations.push({ name: anim.name ?? '', channels, duration })
    }
  }

  // ── Process meshes ────────────────────────────────────────────────
  const meshes: GLTFMesh[] = []

  const jsonMeshes = json.meshes ?? []
  for (let mi = 0; mi < jsonMeshes.length; mi++) {
    const jsonMesh = jsonMeshes[mi]!
    const primitives: GLTFPrimitive[] = []
    const skinIndex = meshSkinMap.get(mi)
    const hasSkin = skinIndex !== undefined

    for (const jsonPrim of jsonMesh.primitives) {
      let positions: Float32Array
      let normals: Float32Array | null
      let indices: Uint16Array | Uint32Array
      let vertexCount: number
      let matIndices: Uint8Array | undefined
      let dracoSkinJoints: Uint8Array | undefined
      let dracoSkinWeights: Float32Array | undefined
      let uvs: Float32Array | undefined

      if (usesDraco && jsonPrim.extensions?.KHR_draco_mesh_compression) {
        const result = await decodeDracoPrimitive(json, jsonPrim, buffers, decoderPath)
        positions = result.positions
        normals = result.normals
        indices = result.indices
        vertexCount = result.vertexCount
        matIndices = result.materialIndices
        dracoSkinJoints = result.skinJoints
        dracoSkinWeights = result.skinWeights
        uvs = result.uvs
      } else {
        const result = parsePrimitive(json, jsonPrim, buffers)
        positions = result.positions
        normals = result.normals
        indices = result.indices
        vertexCount = result.vertexCount
        uvs = result.uvs
      }

      // Compute flat normals if not provided
      if (!normals) {
        normals = computeFlatNormals(positions, indices)
      }

      // Interleave into VoidCore vertex format [px, py, pz, nx, ny, nz, cr, cg, cb, bloom]
      const vertices = interleaveVertices(positions, normals, vertexCount, matIndices, materialColors, materialBloom)

      // Extract material color
      let color: [number, number, number, number] | undefined
      if (jsonPrim.material !== undefined) {
        const mat = json.materials?.[jsonPrim.material]
        const baseColor = mat?.pbrMetallicRoughness?.baseColorFactor
        if (baseColor) {
          color = [baseColor[0], baseColor[1], baseColor[2], baseColor[3]]
        }
      }

      // Extract skin attributes if this mesh has a skin
      const geometry: Geometry = uvs ? { vertices, indices, uvs } : { vertices, indices }
      const prim: GLTFPrimitive = { geometry, color, materialIndices: matIndices, uvs }
      if (hasSkin) {
        if (dracoSkinJoints && dracoSkinWeights) {
          prim.skinJoints = dracoSkinJoints
          prim.skinWeights = dracoSkinWeights
        } else {
          const skinData = parseSkinAttributes(json, jsonPrim, buffers)
          if (skinData) {
            prim.skinJoints = skinData.skinJoints
            prim.skinWeights = skinData.skinWeights
          }
        }
      }

      primitives.push(prim)
    }

    meshes.push({ name: jsonMesh.name, primitives, skinIndex })
  }

  return { meshes, skins, animations, nodeTransforms }
}
