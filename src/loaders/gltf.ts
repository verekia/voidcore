import { Geometry } from '../geometry/geometry.ts'
import { Material, createLambertMaterial } from '../materials/material.ts'
import { Group } from '../scene/group.ts'
import { Mesh } from '../scene/mesh.ts'
import { Node } from '../scene/node.ts'

export interface LoadOptions {
  draco?: { decoderPath: string }
  ktx2?: { transcoderPath: string }
}

export interface GLTFResult {
  scene: Group
  scenes: Group[]
  meshes: Mesh[]
  animations: unknown[]
  textures: unknown[]
  dispose: () => void
}

// ─── GLB Binary Parser ───────────────────────────────────────────────

interface GLBData {
  json: any
  bin: ArrayBuffer
}

const parseGLB = (buffer: ArrayBuffer): GLBData => {
  const view = new DataView(buffer)
  const magic = view.getUint32(0, true)
  if (magic !== 0x46546c67) throw new Error('Invalid GLB magic')
  const version = view.getUint32(4, true)
  if (version !== 2) throw new Error(`Unsupported GLB version: ${version}`)

  let offset = 12
  let json: any = null
  let bin: ArrayBuffer = new ArrayBuffer(0)

  while (offset < buffer.byteLength) {
    const chunkLength = view.getUint32(offset, true)
    const chunkType = view.getUint32(offset + 4, true)
    offset += 8

    if (chunkType === 0x4e4f534a) {
      // JSON chunk
      const decoder = new TextDecoder()
      json = JSON.parse(decoder.decode(new Uint8Array(buffer, offset, chunkLength)))
    } else if (chunkType === 0x004e4942) {
      // BIN chunk
      bin = buffer.slice(offset, offset + chunkLength)
    }

    offset += chunkLength
  }

  if (!json) throw new Error('GLB missing JSON chunk')
  return { json, bin }
}

// ─── Accessor reading ────────────────────────────────────────────────

const TYPE_COUNTS: Record<string, number> = {
  SCALAR: 1,
  VEC2: 2,
  VEC3: 3,
  VEC4: 4,
  MAT2: 4,
  MAT3: 9,
  MAT4: 16,
}

const readAccessor = (json: any, bin: ArrayBuffer, accessorIndex: number): { data: ArrayBufferView; count: number } => {
  const accessor = json.accessors[accessorIndex]
  const bufferView = json.bufferViews[accessor.bufferView]
  const typeCount = TYPE_COUNTS[accessor.type] ?? 1
  const byteOffset = (bufferView.byteOffset ?? 0) + (accessor.byteOffset ?? 0)
  const count = accessor.count
  const totalComponents = count * typeCount

  switch (accessor.componentType) {
    case 5126: // FLOAT
      return { data: new Float32Array(bin, byteOffset, totalComponents), count }
    case 5123: // UNSIGNED_SHORT
      return { data: new Uint16Array(bin, byteOffset, totalComponents), count }
    case 5125: // UNSIGNED_INT
      return { data: new Uint32Array(bin, byteOffset, totalComponents), count }
    case 5121: // UNSIGNED_BYTE
      return { data: new Uint8Array(bin, byteOffset, totalComponents), count }
    case 5122: // SHORT
      return { data: new Int16Array(bin, byteOffset, totalComponents), count }
    case 5120: // BYTE
      return { data: new Int8Array(bin, byteOffset, totalComponents), count }
    default:
      return { data: new Float32Array(bin, byteOffset, totalComponents), count }
  }
}

// ─── Draco decoder ───────────────────────────────────────────────────

let dracoDecoderPromise: Promise<any> | null = null

const loadDracoDecoder = async (decoderPath: string): Promise<any> => {
  if (dracoDecoderPromise) return dracoDecoderPromise

  dracoDecoderPromise = (async () => {
    const url = decoderPath.endsWith('/') ? decoderPath : decoderPath + '/'

    // Load the Draco decoder script and inject as a <script> tag
    return new Promise<any>((resolve, reject) => {
      const script = document.createElement('script')
      script.src = `${url}draco_decoder_gltf.js`
      script.onload = () => {
        const factory = (globalThis as any).DracoDecoderModule
        if (!factory) {
          reject(new Error('DracoDecoderModule not found after script load'))
          return
        }
        factory({ locateFile: (path: string) => `${url}${path}` }).then((mod: any) => {
          resolve(mod)
        })
      }
      script.onerror = () => reject(new Error('Failed to load Draco decoder'))
      document.head.appendChild(script)
    })
  })()

  return dracoDecoderPromise
}

const readDracoAttribute = (
  dracoModule: any,
  decoder: any,
  geometry: any,
  attrId: any,
  numVertices: number,
  components: number,
): Float32Array => {
  const arr = new dracoModule.DracoFloat32Array()
  decoder.GetAttributeFloatForAllPoints(geometry, attrId, arr)
  const out = new Float32Array(numVertices * components)
  for (let i = 0; i < out.length; i++) out[i] = arr.GetValue(i)
  dracoModule.destroy(arr)
  return out
}

const decodeDraco = async (
  dracoModule: any,
  bufferView: ArrayBuffer,
  attributes: Record<string, number>,
): Promise<{
  positions: Float32Array
  normals: Float32Array
  indices: Uint16Array | Uint32Array
  uvs?: Float32Array
  materialIndices?: Uint8Array
}> => {
  const decoder = new dracoModule.Decoder()
  const decoderBuffer = new dracoModule.DecoderBuffer()
  decoderBuffer.Init(new Int8Array(bufferView), bufferView.byteLength)

  const geometryType = decoder.GetEncodedGeometryType(decoderBuffer)
  if (geometryType !== dracoModule.TRIANGULAR_MESH) {
    dracoModule.destroy(decoderBuffer)
    dracoModule.destroy(decoder)
    throw new Error('Draco: unsupported geometry type')
  }

  const dracoGeometry = new dracoModule.Mesh()
  const status = decoder.DecodeBufferToMesh(decoderBuffer, dracoGeometry)
  if (!status.ok()) {
    const msg = status.error_msg()
    dracoModule.destroy(dracoGeometry)
    dracoModule.destroy(decoderBuffer)
    dracoModule.destroy(decoder)
    throw new Error(`Draco decode failed: ${msg}`)
  }

  const numVertices = dracoGeometry.num_points()
  const numFaces = dracoGeometry.num_faces()

  // Positions
  const posAttr =
    attributes.POSITION !== undefined
      ? decoder.GetAttributeByUniqueId(dracoGeometry, attributes.POSITION)
      : decoder.GetAttribute(dracoGeometry, dracoModule.POSITION)
  const positions = readDracoAttribute(dracoModule, decoder, dracoGeometry, posAttr, numVertices, 3)

  // Normals
  let normals: Float32Array
  const normAttr =
    attributes.NORMAL !== undefined
      ? decoder.GetAttributeByUniqueId(dracoGeometry, attributes.NORMAL)
      : decoder.GetAttribute(dracoGeometry, dracoModule.NORMAL)
  if (normAttr && normAttr.ptr !== 0) {
    normals = readDracoAttribute(dracoModule, decoder, dracoGeometry, normAttr, numVertices, 3)
  } else {
    normals = new Float32Array(numVertices * 3) // Will be computed later
  }

  // Indices
  const indices = numVertices > 65535 ? new Uint32Array(numFaces * 3) : new Uint16Array(numFaces * 3)
  const faceArr = new dracoModule.DracoInt32Array()
  for (let i = 0; i < numFaces; i++) {
    decoder.GetFaceFromMesh(dracoGeometry, i, faceArr)
    indices[i * 3] = faceArr.GetValue(0)
    indices[i * 3 + 1] = faceArr.GetValue(1)
    indices[i * 3 + 2] = faceArr.GetValue(2)
  }
  dracoModule.destroy(faceArr)

  // UVs
  let uvs: Float32Array | undefined
  const uvAttr =
    attributes.TEXCOORD_0 !== undefined
      ? decoder.GetAttributeByUniqueId(dracoGeometry, attributes.TEXCOORD_0)
      : decoder.GetAttribute(dracoGeometry, dracoModule.TEX_COORD)
  if (uvAttr && uvAttr.ptr !== 0) {
    uvs = readDracoAttribute(dracoModule, decoder, dracoGeometry, uvAttr, numVertices, 2)
  }

  // Material indices
  let materialIndices: Uint8Array | undefined
  if (attributes._MATERIALINDEX !== undefined) {
    const matAttr = decoder.GetAttributeByUniqueId(dracoGeometry, attributes._MATERIALINDEX)
    if (matAttr && matAttr.ptr !== 0) {
      const floats = readDracoAttribute(dracoModule, decoder, dracoGeometry, matAttr, numVertices, 1)
      materialIndices = new Uint8Array(numVertices)
      for (let i = 0; i < numVertices; i++) materialIndices[i] = Math.round(floats[i]!)
    }
  }

  dracoModule.destroy(dracoGeometry)
  dracoModule.destroy(decoderBuffer)
  dracoModule.destroy(decoder)

  return { positions, normals, indices, uvs, materialIndices }
}

// ─── Main loader ─────────────────────────────────────────────────────

const loadCache = new Map<string, Promise<GLTFResult>>()

export const loadGLTF = async (url: string, _engine: unknown, options?: LoadOptions): Promise<GLTFResult> => {
  let promise = loadCache.get(url)
  if (!promise) {
    promise = loadGLTFImpl(url, options)
    loadCache.set(url, promise)
  }
  return promise
}

const loadGLTFImpl = async (url: string, options?: LoadOptions): Promise<GLTFResult> => {
  const response = await fetch(url)
  const buffer = await response.arrayBuffer()
  const { json, bin } = parseGLB(buffer)

  const meshes: Mesh[] = []
  const allGeometries: Geometry[] = []

  // Load Draco decoder if needed
  let dracoModule: any = null
  const hasDraco = json.extensionsUsed?.includes('KHR_draco_mesh_compression')
  if (hasDraco) {
    if (!options?.draco?.decoderPath) {
      throw new Error('Model uses Draco compression but no decoderPath was provided')
    }
    dracoModule = await loadDracoDecoder(options.draco.decoderPath)
  }

  // Process mesh nodes
  const processNode = async (nodeIdx: number, parent: Node): Promise<void> => {
    const nodeDef = json.nodes[nodeIdx]
    let node: Node

    if (nodeDef.mesh !== undefined) {
      const meshDef = json.meshes[nodeDef.mesh]

      for (const primitive of meshDef.primitives) {
        let geometry: Geometry

        // Check for Draco compression
        const dracoExt = primitive.extensions?.KHR_draco_mesh_compression
        if (dracoExt && dracoModule) {
          const bv = json.bufferViews[dracoExt.bufferView]
          const dracoData = bin.slice(bv.byteOffset ?? 0, (bv.byteOffset ?? 0) + bv.byteLength)
          const decoded = await decodeDraco(dracoModule, dracoData, dracoExt.attributes)
          geometry = new Geometry({
            positions: decoded.positions,
            normals: decoded.normals,
            indices: decoded.indices,
            uvs: decoded.uvs,
            materialIndices: decoded.materialIndices,
          })
        } else {
          // Standard (non-Draco) mesh
          const attrs = primitive.attributes
          const posData = readAccessor(json, bin, attrs.POSITION)
          const normData = attrs.NORMAL !== undefined ? readAccessor(json, bin, attrs.NORMAL) : null
          const uvData = attrs.TEXCOORD_0 !== undefined ? readAccessor(json, bin, attrs.TEXCOORD_0) : null
          const matIdxData = attrs._MATERIALINDEX !== undefined ? readAccessor(json, bin, attrs._MATERIALINDEX) : null

          const positions = new Float32Array(posData.data.buffer, posData.data.byteOffset, posData.count * 3)
          const normals = normData
            ? new Float32Array(normData.data.buffer, normData.data.byteOffset, normData.count * 3)
            : generateFlatNormals(positions, readIndices(json, bin, primitive.indices))

          let indices: Uint16Array | Uint32Array
          if (primitive.indices !== undefined) {
            indices = readIndices(json, bin, primitive.indices)
          } else {
            // Generate sequential indices
            indices =
              posData.count > 65535
                ? new Uint32Array(posData.count).map((_, i) => i)
                : new Uint16Array(posData.count).map((_, i) => i)
          }

          geometry = new Geometry({
            positions,
            normals,
            indices,
            uvs: uvData ? new Float32Array(uvData.data.buffer, uvData.data.byteOffset, uvData.count * 2) : undefined,
            materialIndices: matIdxData
              ? new Uint8Array(matIdxData.data.buffer, matIdxData.data.byteOffset, matIdxData.count)
              : undefined,
          })
        }

        allGeometries.push(geometry)

        // Create material (simplified PBR → Lambert mapping)
        const matIdx = primitive.material
        let material: Material
        if (matIdx !== undefined && json.materials?.[matIdx]) {
          const matDef = json.materials[matIdx]
          const pbr = matDef.pbrMetallicRoughness ?? {}
          const baseColor = pbr.baseColorFactor ?? [1, 1, 1, 1]
          const isUnlit = matDef.extensions?.KHR_materials_unlit !== undefined

          material = new Material(isUnlit ? 'basic' : 'lambert', {
            color: [baseColor[0], baseColor[1], baseColor[2]],
            opacity: baseColor[3] ?? 1.0,
            transparent: matDef.alphaMode === 'BLEND',
          })
        } else {
          material = createLambertMaterial()
        }

        const mesh = new Mesh(geometry, material)
        mesh.name = meshDef.name ?? ''
        meshes.push(mesh)
        node = mesh
      }

      // If multiple primitives, use the last one as the node
      node = meshes[meshes.length - 1]!
    } else {
      node = new Group()
    }

    node.name = nodeDef.name ?? ''

    // Apply transforms
    if (nodeDef.translation) {
      node.position[0] = nodeDef.translation[0]
      node.position[1] = nodeDef.translation[1]
      node.position[2] = nodeDef.translation[2]
      node._dirtyLocal = true
    }
    if (nodeDef.rotation) {
      node.rotation[0] = nodeDef.rotation[0]
      node.rotation[1] = nodeDef.rotation[1]
      node.rotation[2] = nodeDef.rotation[2]
      node.rotation[3] = nodeDef.rotation[3]
      node._dirtyLocal = true
    }
    if (nodeDef.scale) {
      node.scale[0] = nodeDef.scale[0]
      node.scale[1] = nodeDef.scale[1]
      node.scale[2] = nodeDef.scale[2]
      node._dirtyLocal = true
    }

    parent.add(node)

    // Process children
    if (nodeDef.children) {
      for (const childIdx of nodeDef.children) {
        await processNode(childIdx, node)
      }
    }
  }

  const rootGroup = new Group()
  rootGroup.name = 'gltf_root'

  // Process scenes
  const sceneDef = json.scenes?.[json.scene ?? 0]
  if (sceneDef?.nodes) {
    for (const nodeIdx of sceneDef.nodes) {
      await processNode(nodeIdx, rootGroup)
    }
  }

  return {
    scene: rootGroup,
    scenes: [rootGroup],
    meshes,
    animations: [],
    textures: [],
    dispose: () => {
      for (const geom of allGeometries) geom.dispose()
    },
  }
}

// ─── Helpers ─────────────────────────────────────────────────────────

const readIndices = (json: any, bin: ArrayBuffer, accessorIndex: number): Uint16Array | Uint32Array => {
  const accessor = json.accessors[accessorIndex]
  const bufferView = json.bufferViews[accessor.bufferView]
  const byteOffset = (bufferView.byteOffset ?? 0) + (accessor.byteOffset ?? 0)
  const count = accessor.count

  if (accessor.componentType === 5125) {
    return new Uint32Array(bin, byteOffset, count)
  }
  if (accessor.componentType === 5123) {
    return new Uint16Array(bin, byteOffset, count)
  }
  // Uint8 indices -> convert to Uint16
  const bytes = new Uint8Array(bin, byteOffset, count)
  return new Uint16Array(bytes)
}

const generateFlatNormals = (positions: Float32Array, indices: Uint16Array | Uint32Array): Float32Array => {
  const normals = new Float32Array(positions.length)
  for (let i = 0; i < indices.length; i += 3) {
    const i0 = indices[i]! * 3,
      i1 = indices[i + 1]! * 3,
      i2 = indices[i + 2]! * 3
    const ax = positions[i1]! - positions[i0]!,
      ay = positions[i1 + 1]! - positions[i0 + 1]!,
      az = positions[i1 + 2]! - positions[i0 + 2]!
    const bx = positions[i2]! - positions[i0]!,
      by = positions[i2 + 1]! - positions[i0 + 1]!,
      bz = positions[i2 + 2]! - positions[i0 + 2]!
    const nx = ay * bz - az * by,
      ny = az * bx - ax * bz,
      nz = ax * by - ay * bx
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz)
    const inv = len > 1e-6 ? 1 / len : 0
    normals[i0]! += nx * inv
    normals[i0 + 1]! += ny * inv
    normals[i0 + 2]! += nz * inv
    normals[i1]! += nx * inv
    normals[i1 + 1]! += ny * inv
    normals[i1 + 2]! += nz * inv
    normals[i2]! += nx * inv
    normals[i2 + 1]! += ny * inv
    normals[i2 + 2]! += nz * inv
  }
  // Normalize
  for (let i = 0; i < normals.length; i += 3) {
    const len = Math.sqrt(normals[i]! ** 2 + normals[i + 1]! ** 2 + normals[i + 2]! ** 2)
    if (len > 1e-6) {
      normals[i] = normals[i]! / len
      normals[i + 1] = normals[i + 1]! / len
      normals[i + 2] = normals[i + 2]! / len
    }
  }
  return normals
}
