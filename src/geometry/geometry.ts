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

export interface GPUBuffers {
  position: WebGLBuffer
  normal: WebGLBuffer
  index: WebGLBuffer
  uv?: WebGLBuffer
  color?: WebGLBuffer
  materialIndex?: WebGLBuffer
  joints?: WebGLBuffer
  weights?: WebGLBuffer
  vao?: WebGLVertexArrayObject
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

  _gpuBuffers: GPUBuffers | null = null
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

export const createGeometry = (data: GeometryData): Geometry => new Geometry(data)
