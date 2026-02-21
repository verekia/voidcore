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
// new Geometry(data)             – Wraps raw arrays into a Geometry object.
// mergeGeometries(geos)          – Merges multiple geometries into one with per-geometry material indices.
// computeSmoothNormals(geometry) – Computes position-averaged normals for gap-free inverted hull outlines.
// geometry.hasAttribute()        – Checks if optional attributes (UVs, colors, joints, etc.) exist.
// geometry.dispose()             – Releases GPU buffer references.

import { aabbFromPoints } from '../math/index'

import type { AABB } from '../math/index'

export interface GeometryData {
  positions: Float32Array
  normals: Float32Array
  indices: Uint16Array | Uint32Array
  uvs?: Float32Array
  colors?: Float32Array
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
  materialIndices?: Uint8Array
  joints?: Uint8Array | Uint16Array
  weights?: Float32Array

  vertexCount: number
  indexCount: number
  aabb: AABB

  _smoothNormals?: Float32Array
  _gpuBuffers: unknown = null
  needsUpdate = false

  constructor(data: GeometryData) {
    this.positions = data.positions
    this.normals = data.normals
    this.indices = data.indices
    this.uvs = data.uvs
    this.colors = data.colors
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
 * Merges multiple geometries into one, assigning each geometry a material index
 * (0, 1, 2, …) so a palette material can color each sub-mesh independently.
 */
export const mergeGeometries = (geometries: Geometry[]): Geometry => {
  let totalVertices = 0
  let totalIndices = 0
  let hasUVs = false
  for (const geo of geometries) {
    totalVertices += geo.vertexCount
    totalIndices += geo.indexCount
    if (geo.uvs) hasUVs = true
  }

  const positions = new Float32Array(totalVertices * 3)
  const normals = new Float32Array(totalVertices * 3)
  const indices = totalVertices > 65535 ? new Uint32Array(totalIndices) : new Uint16Array(totalIndices)
  const materialIndices = new Uint8Array(totalVertices)
  const uvs = hasUVs ? new Float32Array(totalVertices * 2) : undefined

  let vOff = 0
  let iOff = 0
  for (let i = 0; i < geometries.length; i++) {
    const geo = geometries[i]!
    positions.set(geo.positions, vOff * 3)
    normals.set(geo.normals, vOff * 3)
    if (uvs && geo.uvs) uvs.set(geo.uvs, vOff * 2)
    materialIndices.fill(i, vOff, vOff + geo.vertexCount)
    for (let j = 0; j < geo.indexCount; j++) {
      indices[iOff + j] = geo.indices[j]! + vOff
    }
    vOff += geo.vertexCount
    iOff += geo.indexCount
  }

  return new Geometry({ positions, normals, indices, materialIndices, uvs })
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
