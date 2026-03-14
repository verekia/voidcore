// glTF Loader – Imports 3D models from the glTF/GLB binary format.
//
// glTF (GL Transmission Format) is the standard format for 3D models on the web (like JPEG
// for images). A .glb file is a single binary containing JSON metadata + raw binary buffers
// for vertex data, animations, and textures.
//
// The loader works in several passes:
//   1. Parse the GLB binary: extract the JSON chunk (scene description) and BIN chunk (raw data)
//   2. Create nodes: for each glTF node, create a Mesh, Group, or bone with its transform
//      - For each mesh primitive, read vertex attributes (positions, normals, UVs, indices)
//        from binary accessors, or decode from Draco compression if present
//      - Create materials from glTF PBR definitions (simplified to basic/lambert)
//   3. Build hierarchy: connect parent-child relationships between nodes
//   4. Resolve skins: create Skeletons from joint lists and inverse bind matrices, attach
//      them to the corresponding meshes
//   5. Parse animations: read keyframe tracks (time → value arrays) for bone transforms
//   6. Assemble the final scene graph under a root Group
//
// loadGLTF() – Async function that fetches a .glb URL and returns meshes, skeletons,
//              animations, and a scene graph ready to add to your Scene.

import { Skeleton } from '../animation/skeleton'
import { Geometry } from '../geometry/geometry'
import { Material, LambertMaterial } from '../materials/material'
import { Group } from '../scene/group'
import { Mesh } from '../scene/mesh'
import { Node } from '../scene/node'

import type { AnimationClip, KeyframeTrack } from '../animation/clip'
import type { Mat4 } from '../math/index'

export interface LoadOptions {
  draco?: { decoderPath: string }
  ktx2?: { transcoderPath: string }
}

export interface GLTFResult {
  scene: Group
  scenes: Group[]
  meshes: Mesh[]
  skeletons: Skeleton[]
  animations: AnimationClip[]
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

/** Create a typed array view, copying if the byte offset is not aligned for the element type. */
const alignedView = <T extends Float32Array | Uint16Array | Uint32Array | Int16Array>(
  Ctor: new (buffer: ArrayBuffer, byteOffset: number, length: number) => T,
  bin: ArrayBuffer,
  byteOffset: number,
  length: number,
  bytesPerElement: number,
): T => {
  if (byteOffset % bytesPerElement === 0) {
    return new Ctor(bin, byteOffset, length)
  }
  // Unaligned — copy into a new buffer
  const copy = new Uint8Array(length * bytesPerElement)
  copy.set(new Uint8Array(bin, byteOffset, length * bytesPerElement))
  return new Ctor(copy.buffer, 0, length)
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
      return { data: alignedView(Float32Array, bin, byteOffset, totalComponents, 4), count }
    case 5123: // UNSIGNED_SHORT
      return { data: alignedView(Uint16Array, bin, byteOffset, totalComponents, 2), count }
    case 5125: // UNSIGNED_INT
      return { data: alignedView(Uint32Array, bin, byteOffset, totalComponents, 4), count }
    case 5121: // UNSIGNED_BYTE
      return { data: new Uint8Array(bin, byteOffset, totalComponents), count }
    case 5122: // SHORT
      return { data: alignedView(Int16Array, bin, byteOffset, totalComponents, 2), count }
    case 5120: // BYTE
      return { data: new Int8Array(bin, byteOffset, totalComponents), count }
    default:
      return { data: alignedView(Float32Array, bin, byteOffset, totalComponents, 4), count }
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
  joints?: Uint8Array
  weights?: Float32Array
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

  // Joints (read as float, convert to Uint8)
  let joints: Uint8Array | undefined
  if (attributes.JOINTS_0 !== undefined) {
    const jointAttr = decoder.GetAttributeByUniqueId(dracoGeometry, attributes.JOINTS_0)
    if (jointAttr && jointAttr.ptr !== 0) {
      const floats = readDracoAttribute(dracoModule, decoder, dracoGeometry, jointAttr, numVertices, 4)
      joints = new Uint8Array(numVertices * 4)
      for (let i = 0; i < joints.length; i++) joints[i] = Math.round(floats[i]!)
    }
  }

  // Weights
  let weights: Float32Array | undefined
  if (attributes.WEIGHTS_0 !== undefined) {
    const weightAttr = decoder.GetAttributeByUniqueId(dracoGeometry, attributes.WEIGHTS_0)
    if (weightAttr && weightAttr.ptr !== 0) {
      weights = readDracoAttribute(dracoModule, decoder, dracoGeometry, weightAttr, numVertices, 4)
    }
  }

  dracoModule.destroy(dracoGeometry)
  dracoModule.destroy(decoderBuffer)
  dracoModule.destroy(decoder)

  return { positions, normals, indices, uvs, materialIndices, joints, weights }
}

// ─── Main loader ─────────────────────────────────────────────────────

export const loadGLTF = async (url: string, options?: LoadOptions): Promise<GLTFResult> => {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`)
  const buffer = await response.arrayBuffer()
  const { json, bin } = parseGLB(buffer)

  const meshes: Mesh[] = []
  const allGeometries: Geometry[] = []
  const skeletons: Skeleton[] = []

  // Load Draco decoder if needed
  let dracoModule: any = null
  const hasDraco = json.extensionsUsed?.includes('KHR_draco_mesh_compression')
  if (hasDraco) {
    if (!options?.draco?.decoderPath) {
      throw new Error('Model uses Draco compression but no decoderPath was provided')
    }
    dracoModule = await loadDracoDecoder(options.draco.decoderPath)
  }

  // ─── Pass 1: Create all nodes with transforms ─────────────────────
  const nodeMap = new Map<number, Node>()
  const meshNodeMap = new Map<number, Mesh[]>() // nodeIndex → meshes created for it

  for (let nodeIdx = 0; nodeIdx < (json.nodes?.length ?? 0); nodeIdx++) {
    const nodeDef = json.nodes[nodeIdx]
    let node: Node
    const nodeMeshes: Mesh[] = []

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
            joints: decoded.joints,
            weights: decoded.weights,
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
            : generateSmoothNormals(positions, readIndices(json, bin, primitive.indices))

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

          // Read joints and weights for skinning
          let joints: Uint8Array | Uint16Array | undefined
          let weights: Float32Array | undefined

          if (attrs.JOINTS_0 !== undefined) {
            const jointsData = readAccessor(json, bin, attrs.JOINTS_0)
            if (jointsData.data instanceof Uint8Array) {
              joints = jointsData.data as Uint8Array
            } else if (jointsData.data instanceof Uint16Array) {
              joints = jointsData.data as Uint16Array
            } else {
              // Convert to Uint8Array
              joints = new Uint8Array(jointsData.count * 4)
              for (let i = 0; i < jointsData.count * 4; i++) {
                joints[i] = (jointsData.data as any)[i]
              }
            }
          }

          if (attrs.WEIGHTS_0 !== undefined) {
            const weightsData = readAccessor(json, bin, attrs.WEIGHTS_0)
            if (weightsData.data instanceof Float32Array) {
              weights = weightsData.data as Float32Array
            } else {
              // Normalize from uint8/uint16 to float
              weights = new Float32Array(weightsData.count * 4)
              const src = weightsData.data
              const max = src instanceof Uint8Array ? 255 : 65535
              for (let i = 0; i < weightsData.count * 4; i++) {
                weights[i] = (src as any)[i] / max
              }
            }
          }

          geometry = new Geometry({
            positions,
            normals,
            indices,
            uvs: uvData ? new Float32Array(uvData.data.buffer, uvData.data.byteOffset, uvData.count * 2) : undefined,
            materialIndices: matIdxData
              ? new Uint8Array(matIdxData.data.buffer, matIdxData.data.byteOffset, matIdxData.count)
              : undefined,
            joints,
            weights,
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
          const isBlend = matDef.alphaMode === 'BLEND'
          const alpha = baseColor[3] ?? 1

          material = new Material(isUnlit ? 'basic' : 'lambert', {
            color: [baseColor[0], baseColor[1], baseColor[2]],
            opacity: alpha,
            transparent: isBlend || alpha < 1,
          })
        } else {
          material = new LambertMaterial()
        }

        const mesh = new Mesh(geometry, material)
        mesh.name = meshDef.name ?? ''
        meshes.push(mesh)
        nodeMeshes.push(mesh)
      }

      if (nodeMeshes.length === 1) {
        node = nodeMeshes[0]!
      } else {
        // Multi-primitive: wrap in a group so all primitives are in the scene graph
        node = new Group()
        for (const m of nodeMeshes) node.add(m)
      }
    } else {
      node = new Group()
    }

    node.name = nodeDef.name ?? ''

    // Apply transforms — glTF nodes use either TRS properties or a column-major matrix
    if (nodeDef.matrix) {
      // Decompose column-major 4x4 matrix into TRS
      const m = nodeDef.matrix
      // Translation from column 3
      node.setPosition(m[12], m[13], m[14])
      // Scale = length of each column (0-2)
      const sx = Math.sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2])
      const sy = Math.sqrt(m[4] * m[4] + m[5] * m[5] + m[6] * m[6])
      const sz = Math.sqrt(m[8] * m[8] + m[9] * m[9] + m[10] * m[10])
      node.setScale(sx, sy, sz)
      // Rotation: extract from normalized rotation matrix, convert to quaternion
      const isx = sx > 1e-6 ? 1 / sx : 0
      const isy = sy > 1e-6 ? 1 / sy : 0
      const isz = sz > 1e-6 ? 1 / sz : 0
      const r00 = m[0] * isx, r01 = m[4] * isy, r02 = m[8] * isz
      const r10 = m[1] * isx, r11 = m[5] * isy, r12 = m[9] * isz
      const r20 = m[2] * isx, r21 = m[6] * isy, r22 = m[10] * isz
      const trace = r00 + r11 + r22
      let qx: number, qy: number, qz: number, qw: number
      if (trace > 0) {
        const s = 0.5 / Math.sqrt(trace + 1)
        qw = 0.25 / s
        qx = (r21 - r12) * s
        qy = (r02 - r20) * s
        qz = (r10 - r01) * s
      } else if (r00 > r11 && r00 > r22) {
        const s = 2 * Math.sqrt(1 + r00 - r11 - r22)
        qw = (r21 - r12) / s
        qx = 0.25 * s
        qy = (r01 + r10) / s
        qz = (r02 + r20) / s
      } else if (r11 > r22) {
        const s = 2 * Math.sqrt(1 + r11 - r00 - r22)
        qw = (r02 - r20) / s
        qx = (r01 + r10) / s
        qy = 0.25 * s
        qz = (r12 + r21) / s
      } else {
        const s = 2 * Math.sqrt(1 + r22 - r00 - r11)
        qw = (r10 - r01) / s
        qx = (r02 + r20) / s
        qy = (r12 + r21) / s
        qz = 0.25 * s
      }
      node.setRotation(qx, qy, qz, qw)
    } else {
      if (nodeDef.translation) {
        node.setPosition(nodeDef.translation[0], nodeDef.translation[1], nodeDef.translation[2])
      }
      if (nodeDef.rotation) {
        node.setRotation(nodeDef.rotation[0], nodeDef.rotation[1], nodeDef.rotation[2], nodeDef.rotation[3])
      }
      if (nodeDef.scale) {
        node.setScale(nodeDef.scale[0], nodeDef.scale[1], nodeDef.scale[2])
      }
    }

    nodeMap.set(nodeIdx, node)
    if (nodeMeshes.length > 0) {
      meshNodeMap.set(nodeIdx, nodeMeshes)
    }
  }

  // ─── Build hierarchy ──────────────────────────────────────────────
  for (let nodeIdx = 0; nodeIdx < (json.nodes?.length ?? 0); nodeIdx++) {
    const nodeDef = json.nodes[nodeIdx]
    const parentNode = nodeMap.get(nodeIdx)!
    if (nodeDef.children) {
      for (const childIdx of nodeDef.children) {
        const childNode = nodeMap.get(childIdx)
        if (childNode) parentNode.add(childNode)
      }
    }
  }

  // ─── Pass 2: Resolve skins → Skeletons ────────────────────────────
  if (json.skins) {
    for (let skinIdx = 0; skinIdx < json.skins.length; skinIdx++) {
      const skinDef = json.skins[skinIdx]
      const jointIndices: number[] = skinDef.joints
      const bones: Node[] = []
      const inverseBindMatrices: Mat4[] = []

      // Read inverse bind matrices
      let ibmData: Float32Array | null = null
      if (skinDef.inverseBindMatrices !== undefined) {
        const acc = readAccessor(json, bin, skinDef.inverseBindMatrices)
        ibmData = new Float32Array(acc.data.buffer, acc.data.byteOffset, acc.count * 16)
      }

      for (let j = 0; j < jointIndices.length; j++) {
        const boneNode = nodeMap.get(jointIndices[j]!)
        if (boneNode) bones.push(boneNode)

        const ibm = new Float32Array(16)
        if (ibmData) {
          ibm.set(ibmData.subarray(j * 16, j * 16 + 16))
        } else {
          // Identity
          ibm[0] = 1
          ibm[5] = 1
          ibm[10] = 1
          ibm[15] = 1
        }
        inverseBindMatrices.push(ibm)
      }

      const skeleton = new Skeleton(bones, inverseBindMatrices)
      skeletons.push(skeleton)
    }

    // Attach skeletons to meshes
    for (let nodeIdx = 0; nodeIdx < (json.nodes?.length ?? 0); nodeIdx++) {
      const nodeDef = json.nodes[nodeIdx]
      if (nodeDef.skin !== undefined && nodeDef.mesh !== undefined) {
        const skinSkeleton = skeletons[nodeDef.skin]
        const nodeMeshes = meshNodeMap.get(nodeIdx)
        if (skinSkeleton && nodeMeshes) {
          for (const m of nodeMeshes) {
            m.skeleton = skinSkeleton
          }
        }
      }
    }
  }

  // ─── Parse animations ─────────────────────────────────────────────
  const animations: AnimationClip[] = []

  if (json.animations) {
    for (let animIdx = 0; animIdx < json.animations.length; animIdx++) {
      const animDef = json.animations[animIdx]
      const tracks: KeyframeTrack[] = []
      let duration = 0

      for (const channel of animDef.channels) {
        const sampler = animDef.samplers[channel.sampler]
        const targetNodeIdx = channel.target.node
        const targetPath = channel.target.path as 'translation' | 'rotation' | 'scale'

        // Only support translation/rotation/scale
        if (targetPath !== 'translation' && targetPath !== 'rotation' && targetPath !== 'scale') continue

        // Find bone index in the first skeleton
        let boneIndex = -1
        if (skeletons.length > 0) {
          const skeleton = skeletons[0]!
          const targetNode = nodeMap.get(targetNodeIdx)
          if (targetNode) {
            boneIndex = skeleton.bones.indexOf(targetNode)
          }
        }
        if (boneIndex < 0) continue

        // Read keyframe data
        const inputAcc = readAccessor(json, bin, sampler.input)
        const outputAcc = readAccessor(json, bin, sampler.output)

        const times = new Float32Array(inputAcc.data.buffer, inputAcc.data.byteOffset, inputAcc.count)
        const stride = targetPath === 'rotation' ? 4 : 3
        const values = new Float32Array(outputAcc.data.buffer, outputAcc.data.byteOffset, outputAcc.count * stride)

        // Track max time for duration
        if (times.length > 0) {
          const maxTime = times[times.length - 1]!
          if (maxTime > duration) duration = maxTime
        }

        const interpolation = sampler.interpolation === 'STEP' ? 'STEP' : 'LINEAR'

        tracks.push({ boneIndex, path: targetPath, times, values, interpolation } as KeyframeTrack)
      }

      animations.push({ name: animDef.name ?? `animation_${animIdx}`, duration, tracks })
    }
  }

  // ─── Assemble scene ───────────────────────────────────────────────
  const rootGroup = new Group()
  rootGroup.name = 'gltf_root'

  // Process scenes
  const sceneDef = json.scenes?.[json.scene ?? 0]
  if (sceneDef?.nodes) {
    for (const nodeIdx of sceneDef.nodes) {
      const node = nodeMap.get(nodeIdx)
      if (node) rootGroup.add(node)
    }
  }

  return {
    scene: rootGroup,
    scenes: [rootGroup],
    meshes,
    skeletons,
    animations,
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
    return alignedView(Uint32Array, bin, byteOffset, count, 4)
  }
  if (accessor.componentType === 5123) {
    return alignedView(Uint16Array, bin, byteOffset, count, 2)
  }
  // Uint8 indices -> convert to Uint16
  const bytes = new Uint8Array(bin, byteOffset, count)
  return new Uint16Array(bytes)
}

const generateSmoothNormals = (positions: Float32Array, indices: Uint16Array | Uint32Array): Float32Array => {
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
    normals[i0] = normals[i0]! + nx * inv
    normals[i0 + 1] = normals[i0 + 1]! + ny * inv
    normals[i0 + 2] = normals[i0 + 2]! + nz * inv
    normals[i1] = normals[i1]! + nx * inv
    normals[i1 + 1] = normals[i1 + 1]! + ny * inv
    normals[i1 + 2] = normals[i1 + 2]! + nz * inv
    normals[i2] = normals[i2]! + nx * inv
    normals[i2 + 1] = normals[i2 + 1]! + ny * inv
    normals[i2 + 2] = normals[i2 + 2]! + nz * inv
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
