import type { Mesh } from './mesh.ts'

export interface BVH {
  offset: number // Byte offset into WASM memory where BVH header lives
  triCount: number
}

export interface RaycastHit {
  hit: boolean
  distance: number
  pointX: number
  pointY: number
  pointZ: number
  normalX: number
  normalY: number
  normalZ: number
  faceIndex: number
  mesh: Mesh | null
}

export function createRaycastHit(): RaycastHit {
  return {
    hit: false,
    distance: Infinity,
    pointX: 0,
    pointY: 0,
    pointZ: 0,
    normalX: 0,
    normalY: 0,
    normalZ: 0,
    faceIndex: -1,
    mesh: null,
  }
}
