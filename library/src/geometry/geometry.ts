// Geometry – Container for mesh vertex data (positions, normals, UVs, indices, etc.).
//
// A Geometry holds all the raw arrays that describe the shape of a 3D object. The GPU
// needs this data uploaded as buffers to draw triangles. Each vertex has a position in 3D
// space, a normal vector (used for lighting), optional UV coordinates (for textures), and
// an index buffer that defines which vertices form each triangle.
//
// Geometry also computes an axis-aligned bounding box (AABB) used for frustum culling
// (skipping objects that are off-screen) and raycasting (click detection).
//
// Vertex colors replace the palette system for per-vertex coloring. Instead of material
// uniforms, colors and emissive values are baked directly into the geometry as vertex
// attributes. The bakePalette() utility resolves materialIndices + palette entries into
// per-vertex colors and emissiveColors arrays. Palette entries can include per-material
// tiled AO textures — bakePalette() packs the layer index into colors.a (unorm8) and
// the tiled AO intensity into emissiveColors.a (float16). Unique textures are collected
// into tiledAoTextures/tiledAoScales arrays on the result geometry, which renderers
// upload as a 2D array texture for world-space tiled AO sampling.
//
// new Geometry(data)                  – Wraps raw arrays into a Geometry object.
// bakePalette(geometry, palette)      – Bakes palette entries into vertex colors/emissiveColors (cached by reference).
// clearColoredGeometryCache(geo?)     – Clears cached baked geometries (all, or for a specific geometry).
// mergeGeometries(geos)               – Merges multiple geometries into one (preserves vertex colors).
// mergeStaticIntoSkinned(skinned, …)  – Merges a static geometry into a skinned one, binding vertices to a bone.
// computeSmoothNormals(geometry)      – Computes position-averaged normals for gap-free inverted hull outlines.
// geometry.hasAttribute()             – Checks if optional attributes (UVs, colors, joints, etc.) exist.
// geometry.dispose()                  – Releases GPU buffer references.

import { aabbFromPoints, mat4Create, mat4Invert, mat4Multiply, vec3TransformMat4 } from '../math/index'

import type { PaletteEntry } from '../materials/material'
import type { AABB, Mat4 } from '../math/index'

export interface GeometryData {
  positions: Float32Array
  normals: Float32Array
  indices: Uint16Array | Uint32Array
  uvs?: Float32Array
  colors?: Float32Array
  emissiveColors?: Float32Array
  materialIndices?: Uint8Array
  joints?: Uint8Array | Uint16Array
  weights?: Float32Array
}

export class Geometry {
  positions: Float32Array
  normals: Float32Array
  indices: Uint16Array | Uint32Array
  uvs?: Float32Array
  colors?: Float32Array
  emissiveColors?: Float32Array
  materialIndices?: Uint8Array
  joints?: Uint8Array | Uint16Array
  weights?: Float32Array

  vertexCount: number
  indexCount: number
  aabb: AABB

  tiledAoTextures?: import('../materials/texture').Texture[]
  tiledAoScales?: Float32Array

  _smoothNormals?: Float32Array
  _gpuBuffers: unknown = null
  needsUpdate = false

  constructor(data: GeometryData) {
    this.positions = data.positions
    this.normals = data.normals
    this.indices = data.indices
    this.uvs = data.uvs
    this.colors = data.colors
    this.emissiveColors = data.emissiveColors
    this.materialIndices = data.materialIndices
    this.joints = data.joints
    this.weights = data.weights
    this.vertexCount = data.positions.length / 3
    this.indexCount = data.indices.length
    this.aabb = aabbFromPoints(new Float32Array(6), data.positions, this.vertexCount)
  }

  hasAttribute(name: string): boolean {
    switch (name) {
      case 'uv':
        return !!this.uvs
      case 'color':
        return !!this.colors
      case 'emissiveColor':
        return !!this.emissiveColors
      case 'materialIndex':
        return !!this.materialIndices
      case 'joints':
        return !!this.joints
      case 'weights':
        return !!this.weights
      default:
        return false
    }
  }

  dispose() {
    this._smoothNormals = undefined
    this._gpuBuffers = null
  }
}

/**
 * Bakes palette entries into per-vertex colors and emissiveColors.
 * Reads materialIndices from the geometry, looks up each palette entry,
 * and writes RGBA colors and RGBA emissiveColors (RGB pre-multiplied by intensity, A=1).
 * Returns a new Geometry without materialIndices.
 */
let coloredGeometryCache = new WeakMap<Geometry, WeakMap<PaletteEntry[], Geometry>>()

export const clearColoredGeometryCache = (geometry?: Geometry) => {
  if (geometry) {
    coloredGeometryCache.delete(geometry)
  } else {
    coloredGeometryCache = new WeakMap()
  }
}

export const bakePalette = (geometry: Geometry, palette: PaletteEntry[]): Geometry => {
  let byPalette = coloredGeometryCache.get(geometry)
  if (byPalette) {
    const cached = byPalette.get(palette)
    if (cached) return cached
  }

  // Collect unique tiled AO textures and assign 1-based layer indices
  const tiledAoTextureMap = new Map<import('../materials/texture').Texture, number>()
  const tiledAoTextures: import('../materials/texture').Texture[] = []
  const tiledAoScalesList: number[] = []
  for (const entry of palette) {
    if (entry.tiledAo && !tiledAoTextureMap.has(entry.tiledAo)) {
      const layerIndex = tiledAoTextures.length + 1 // 1-based (0 = none)
      tiledAoTextureMap.set(entry.tiledAo, layerIndex)
      tiledAoTextures.push(entry.tiledAo)
      tiledAoScalesList.push(entry.tiledAoScale ?? 1.0)
    }
  }

  const vc = geometry.vertexCount
  const colors = new Float32Array(vc * 4)
  const emissiveColors = new Float32Array(vc * 4)
  const matIndices = geometry.materialIndices

  for (let i = 0; i < vc; i++) {
    const idx = matIndices ? matIndices[i]! : 0
    const entry: PaletteEntry = palette[idx] ?? { color: [1, 1, 1] }
    const o = i * 4
    colors[o] = entry.color[0]
    colors[o + 1] = entry.color[1]
    colors[o + 2] = entry.color[2]
    // Pack tiled AO layer index into colors.a (unorm8: 0=none, 1-255=layer)
    const layerIndex = entry.tiledAo ? (tiledAoTextureMap.get(entry.tiledAo) ?? 0) : 0
    colors[o + 3] = layerIndex / 255
    const intensity = entry.emissiveIntensity ?? 0
    emissiveColors[o] = (entry.emissive?.[0] ?? 0) * intensity
    emissiveColors[o + 1] = (entry.emissive?.[1] ?? 0) * intensity
    emissiveColors[o + 2] = (entry.emissive?.[2] ?? 0) * intensity
    // Pack tiled AO intensity into emissiveColors.a (float16, 0 = no tiled AO)
    emissiveColors[o + 3] = entry.tiledAo ? (entry.tiledAoIntensity ?? 1.0) : 0
  }

  const result = new Geometry({
    positions: geometry.positions,
    normals: geometry.normals,
    indices: geometry.indices,
    uvs: geometry.uvs,
    colors,
    emissiveColors,
    joints: geometry.joints,
    weights: geometry.weights,
  })

  // Attach tiled AO data if any textures were found
  if (tiledAoTextures.length > 0) {
    result.tiledAoTextures = tiledAoTextures
    result.tiledAoScales = new Float32Array(tiledAoScalesList)
  }

  if (!byPalette) {
    byPalette = new WeakMap()
    coloredGeometryCache.set(geometry, byPalette)
  }
  byPalette.set(palette, result)

  return result
}

/**
 * Merges multiple geometries into one. If any input has vertex colors,
 * all inputs must have colors (non-colored inputs get white).
 */
export const mergeGeometries = (geometries: Geometry[]): Geometry => {
  let totalVertices = 0
  let totalIndices = 0
  let hasUVs = false
  let hasColors = false
  let hasMaterialIndices = false
  for (const geo of geometries) {
    totalVertices += geo.vertexCount
    totalIndices += geo.indexCount
    if (geo.uvs) hasUVs = true
    if (geo.colors) hasColors = true
    if (geo.materialIndices) hasMaterialIndices = true
  }

  const positions = new Float32Array(totalVertices * 3)
  const normals = new Float32Array(totalVertices * 3)
  const indices = totalVertices > 65535 ? new Uint32Array(totalIndices) : new Uint16Array(totalIndices)
  const uvs = hasUVs ? new Float32Array(totalVertices * 2) : undefined
  const colors = hasColors ? new Float32Array(totalVertices * 4) : undefined
  const emissiveColors = hasColors ? new Float32Array(totalVertices * 4) : undefined
  const materialIndices = hasMaterialIndices ? new Uint8Array(totalVertices) : undefined

  let vOff = 0
  let iOff = 0
  for (let i = 0; i < geometries.length; i++) {
    const geo = geometries[i]!
    positions.set(geo.positions, vOff * 3)
    normals.set(geo.normals, vOff * 3)
    if (uvs && geo.uvs) uvs.set(geo.uvs, vOff * 2)
    if (colors) {
      if (geo.colors) {
        colors.set(geo.colors, vOff * 4)
      } else {
        // Default white for non-colored geometries
        for (let j = 0; j < geo.vertexCount; j++) {
          const o = (vOff + j) * 4
          colors[o] = 1
          colors[o + 1] = 1
          colors[o + 2] = 1
          colors[o + 3] = 1
        }
      }
    }
    if (emissiveColors) {
      if (geo.emissiveColors) {
        emissiveColors.set(geo.emissiveColors, vOff * 4)
      } else {
        // Default: no emissive, but A=1
        for (let j = 0; j < geo.vertexCount; j++) {
          emissiveColors[(vOff + j) * 4 + 3] = 1
        }
      }
    }
    if (materialIndices && geo.materialIndices) {
      materialIndices.set(geo.materialIndices, vOff)
    }
    for (let j = 0; j < geo.indexCount; j++) {
      indices[iOff + j] = geo.indices[j]! + vOff
    }
    vOff += geo.vertexCount
    iOff += geo.indexCount
  }

  const result = new Geometry({ positions, normals, indices, uvs, colors, emissiveColors, materialIndices })

  // Propagate tiled AO data (all source geometries must share the same arrays by reference)
  for (const geo of geometries) {
    if (geo.tiledAoTextures) {
      result.tiledAoTextures = geo.tiledAoTextures
      result.tiledAoScales = geo.tiledAoScales
      break
    }
  }

  return result
}

/**
 * Merges a static geometry into a skinned geometry, binding all static vertices
 * to a single bone. This lets you attach a weapon (static mesh) to a character's
 * hand bone and render both in a single draw call.
 *
 * Static positions/normals are transformed into bind-pose space so they animate
 * correctly with the skeleton. The static vertices get joints=[boneIndex,0,0,0]
 * and weights=[1,0,0,0] (100% influence from the target bone).
 */
export const mergeStaticIntoSkinned = (
  skinned: Geometry,
  static_: Geometry,
  boneIndex: number,
  inverseBindMatrix: Mat4,
  localTransform?: Mat4,
): Geometry => {
  // Compute bind-pose transform: inverse(IBM) brings us from bone-local to bind-pose space
  const bindPoseMatrix = mat4Create()
  if (!mat4Invert(bindPoseMatrix, inverseBindMatrix)) return skinned

  const finalTransform = mat4Create()
  if (localTransform) {
    mat4Multiply(finalTransform, bindPoseMatrix, localTransform)
  } else {
    finalTransform.set(bindPoseMatrix)
  }

  const skinnedVerts = skinned.vertexCount
  const staticVerts = static_.vertexCount
  const totalVerts = skinnedVerts + staticVerts
  const totalIndices = skinned.indexCount + static_.indexCount

  // Transform static positions into bind-pose space
  const positions = new Float32Array(totalVerts * 3)
  positions.set(skinned.positions)
  const tmpVec = new Float32Array(3)
  for (let i = 0; i < staticVerts; i++) {
    tmpVec[0] = static_.positions[i * 3]!
    tmpVec[1] = static_.positions[i * 3 + 1]!
    tmpVec[2] = static_.positions[i * 3 + 2]!
    vec3TransformMat4(tmpVec, tmpVec, finalTransform)
    positions[(skinnedVerts + i) * 3] = tmpVec[0]!
    positions[(skinnedVerts + i) * 3 + 1] = tmpVec[1]!
    positions[(skinnedVerts + i) * 3 + 2] = tmpVec[2]!
  }

  // Transform static normals using the upper-left 3x3 (rotation only, no translation)
  const normals = new Float32Array(totalVerts * 3)
  normals.set(skinned.normals)
  const m = finalTransform
  for (let i = 0; i < staticVerts; i++) {
    const nx = static_.normals[i * 3]!
    const ny = static_.normals[i * 3 + 1]!
    const nz = static_.normals[i * 3 + 2]!
    let rx = m[0]! * nx + m[4]! * ny + m[8]! * nz
    let ry = m[1]! * nx + m[5]! * ny + m[9]! * nz
    let rz = m[2]! * nx + m[6]! * ny + m[10]! * nz
    const len = Math.sqrt(rx * rx + ry * ry + rz * rz)
    if (len > 1e-6) {
      const inv = 1 / len
      rx *= inv
      ry *= inv
      rz *= inv
    }
    normals[(skinnedVerts + i) * 3] = rx
    normals[(skinnedVerts + i) * 3 + 1] = ry
    normals[(skinnedVerts + i) * 3 + 2] = rz
  }

  // Merge indices (offset static indices by skinned vertex count)
  const indices = totalVerts > 65535 ? new Uint32Array(totalIndices) : new Uint16Array(totalIndices)
  indices.set(skinned.indices)
  for (let i = 0; i < static_.indexCount; i++) {
    indices[skinned.indexCount + i] = static_.indices[i]! + skinnedVerts
  }

  // Merge joints: skinned joints + static joints (all bound to boneIndex)
  const useUint16 = skinned.joints instanceof Uint16Array || boneIndex > 255
  const JointsArray = useUint16 ? Uint16Array : Uint8Array
  const joints = new JointsArray(totalVerts * 4)
  if (skinned.joints) joints.set(skinned.joints)
  for (let i = 0; i < staticVerts; i++) {
    joints[(skinnedVerts + i) * 4] = boneIndex
  }

  // Merge weights: skinned weights + static weights (100% on target bone)
  const weights = new Float32Array(totalVerts * 4)
  if (skinned.weights) weights.set(skinned.weights)
  for (let i = 0; i < staticVerts; i++) {
    weights[(skinnedVerts + i) * 4] = 1
  }

  // Merge vertex colors
  let colors: Float32Array | undefined
  if (skinned.colors || static_.colors) {
    colors = new Float32Array(totalVerts * 4)
    if (skinned.colors) {
      colors.set(skinned.colors)
    } else {
      for (let i = 0; i < skinnedVerts; i++) {
        const o = i * 4
        colors[o] = 1
        colors[o + 1] = 1
        colors[o + 2] = 1
        colors[o + 3] = 1
      }
    }
    if (static_.colors) {
      colors.set(static_.colors, skinnedVerts * 4)
    } else {
      for (let i = 0; i < staticVerts; i++) {
        const o = (skinnedVerts + i) * 4
        colors[o] = 1
        colors[o + 1] = 1
        colors[o + 2] = 1
        colors[o + 3] = 1
      }
    }
  }

  // Merge emissive colors
  let emissiveColors: Float32Array | undefined
  if (skinned.emissiveColors || static_.emissiveColors) {
    emissiveColors = new Float32Array(totalVerts * 4)
    if (skinned.emissiveColors) {
      emissiveColors.set(skinned.emissiveColors)
    } else {
      for (let i = 0; i < skinnedVerts; i++) {
        emissiveColors[i * 4 + 3] = 1
      }
    }
    if (static_.emissiveColors) {
      emissiveColors.set(static_.emissiveColors, skinnedVerts * 4)
    } else {
      for (let i = 0; i < staticVerts; i++) {
        emissiveColors[(skinnedVerts + i) * 4 + 3] = 1
      }
    }
  }

  // Merge UVs
  let uvs: Float32Array | undefined
  if (skinned.uvs || static_.uvs) {
    uvs = new Float32Array(totalVerts * 2)
    if (skinned.uvs) uvs.set(skinned.uvs)
    if (static_.uvs) uvs.set(static_.uvs, skinnedVerts * 2)
  }

  const result = new Geometry({ positions, normals, indices, joints, weights, uvs, colors, emissiveColors })

  // Propagate tiled AO data from either source geometry
  const tiledSrc = skinned.tiledAoTextures ? skinned : static_.tiledAoTextures ? static_ : null
  if (tiledSrc) {
    result.tiledAoTextures = tiledSrc.tiledAoTextures
    result.tiledAoScales = tiledSrc.tiledAoScales
  }

  return result
}

/**
 * Computes position-averaged smooth normals for gap-free inverted hull outlines.
 * Vertices sharing the same position get the same averaged normal so the outline
 * inflates uniformly at hard edges, eliminating visible gaps.
 */
export const computeSmoothNormals = (geometry: Geometry): Float32Array => {
  const positions = geometry.positions
  const normals = geometry.normals
  const count = geometry.vertexCount
  const smooth = new Float32Array(count * 3)

  // Accumulate normals per unique position
  const accum = new Map<string, { x: number; y: number; z: number }>()
  const PRECISION = 1e4

  for (let i = 0; i < count; i++) {
    const px = Math.round(positions[i * 3]! * PRECISION)
    const py = Math.round(positions[i * 3 + 1]! * PRECISION)
    const pz = Math.round(positions[i * 3 + 2]! * PRECISION)
    const key = `${px},${py},${pz}`

    const nx = normals[i * 3]!
    const ny = normals[i * 3 + 1]!
    const nz = normals[i * 3 + 2]!

    const entry = accum.get(key)
    if (entry) {
      entry.x += nx
      entry.y += ny
      entry.z += nz
    } else {
      accum.set(key, { x: nx, y: ny, z: nz })
    }
  }

  // Write normalized results
  for (let i = 0; i < count; i++) {
    const px = Math.round(positions[i * 3]! * PRECISION)
    const py = Math.round(positions[i * 3 + 1]! * PRECISION)
    const pz = Math.round(positions[i * 3 + 2]! * PRECISION)
    const key = `${px},${py},${pz}`

    const entry = accum.get(key)!
    let len = Math.sqrt(entry.x * entry.x + entry.y * entry.y + entry.z * entry.z)
    if (len < 1e-8) len = 1
    smooth[i * 3] = entry.x / len
    smooth[i * 3 + 1] = entry.y / len
    smooth[i * 3 + 2] = entry.z / len
  }

  geometry._smoothNormals = smooth
  return smooth
}
