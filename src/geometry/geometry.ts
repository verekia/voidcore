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
// new Geometry(data)      – Wraps raw arrays into a Geometry object.
// geometry.hasAttribute() – Checks if optional attributes (UVs, colors, joints, etc.) exist.
// geometry.dispose()      – Releases GPU buffer references.

import { aabbFromPoints } from '../math/index.ts'

import type { AABB } from '../math/index.ts'

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
    this._gpuBuffers = null
  }
}
