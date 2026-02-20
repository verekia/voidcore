import { expect, test, describe } from 'bun:test'

import { mat4Create, mat4Identity } from '../math/index.ts'
import { createSortState, sortMeshes } from './sort.ts'

import type { PerspectiveCamera } from '../scene/camera.ts'
import type { Mesh } from '../scene/mesh.ts'

/** Create a minimal mock mesh for sort testing */
const mockMesh = (opts: {
  x: number
  y: number
  z: number
  materialId: number
  materialType?: 'basic' | 'lambert'
}): Mesh => {
  const wm = mat4Create()
  mat4Identity(wm)
  wm[12] = opts.x
  wm[13] = opts.y
  wm[14] = opts.z
  return {
    _worldMatrix: wm,
    material: {
      _id: opts.materialId,
      type: opts.materialType ?? 'basic',
    },
    skeleton: null,
    geometry: {},
    _isSkinned: false,
  } as unknown as Mesh
}

/** Create a mock camera at a position with a given far plane */
const mockCamera = (x: number, y: number, z: number, far: number): PerspectiveCamera => {
  const wm = mat4Create()
  mat4Identity(wm)
  wm[12] = x
  wm[13] = y
  wm[14] = z
  return { _worldMatrix: wm, far } as unknown as PerspectiveCamera
}

// ─── createSortState ───────────────────────────────────────────────

describe('createSortState', () => {
  test('allocates buffers with correct capacity', () => {
    const state = createSortState(100)
    expect(state.keys.length).toBe(100)
    expect(state.indices.length).toBe(100)
    expect(state.tempIndices.length).toBe(100)
    expect(state.counts.length).toBe(256)
    expect(state.capacity).toBe(100)
  })

  test('has no temp property', () => {
    const state = createSortState(10)
    expect('temp' in state).toBe(false)
  })
})

// ─── sortMeshes ────────────────────────────────────────────────────

describe('sortMeshes', () => {
  test('handles empty mesh list', () => {
    const state = createSortState(10)
    sortMeshes(state, [], 0, mockCamera(0, 0, 0, 100))
    // Should not throw
  })

  test('sorts nearest-first', () => {
    const state = createSortState(10)
    // Same material so depth is the tiebreaker
    const meshes = [
      mockMesh({ x: 50, y: 0, z: 0, materialId: 1 }), // far
      mockMesh({ x: 5, y: 0, z: 0, materialId: 1 }), // near
      mockMesh({ x: 25, y: 0, z: 0, materialId: 1 }), // mid
    ]
    const camera = mockCamera(0, 0, 0, 100)
    sortMeshes(state, meshes, 3, camera)

    const sorted = Array.from(state.indices.subarray(0, 3)).map(i => meshes[i]!)
    const distances = sorted.map(m => m._worldMatrix[12]!)
    // Nearest first
    expect(distances[0]).toBeLessThanOrEqual(distances[1]!)
    expect(distances[1]).toBeLessThanOrEqual(distances[2]!)
  })

  test('groups by pipeline (lambert vs basic)', () => {
    const state = createSortState(10)
    const meshes = [
      mockMesh({ x: 10, y: 0, z: 0, materialId: 1, materialType: 'lambert' }),
      mockMesh({ x: 10, y: 0, z: 0, materialId: 1, materialType: 'basic' }),
      mockMesh({ x: 10, y: 0, z: 0, materialId: 1, materialType: 'lambert' }),
      mockMesh({ x: 10, y: 0, z: 0, materialId: 1, materialType: 'basic' }),
    ]
    const camera = mockCamera(0, 0, 0, 100)
    sortMeshes(state, meshes, 4, camera)

    const sorted = Array.from(state.indices.subarray(0, 4)).map(i => meshes[i]!)
    const types = sorted.map(m => m.material.type)
    // Basic (0) should come before lambert (1)
    expect(types[0]).toBe('basic')
    expect(types[1]).toBe('basic')
    expect(types[2]).toBe('lambert')
    expect(types[3]).toBe('lambert')
  })

  test('groups by material within same pipeline', () => {
    const state = createSortState(10)
    const meshes = [
      mockMesh({ x: 10, y: 0, z: 0, materialId: 3 }),
      mockMesh({ x: 10, y: 0, z: 0, materialId: 1 }),
      mockMesh({ x: 10, y: 0, z: 0, materialId: 3 }),
      mockMesh({ x: 10, y: 0, z: 0, materialId: 1 }),
    ]
    const camera = mockCamera(0, 0, 0, 100)
    sortMeshes(state, meshes, 4, camera)

    const sorted = Array.from(state.indices.subarray(0, 4)).map(i => meshes[i]!)
    const ids = sorted.map(m => m.material._id)
    // Lower material ID first
    expect(ids[0]).toBe(1)
    expect(ids[1]).toBe(1)
    expect(ids[2]).toBe(3)
    expect(ids[3]).toBe(3)
  })

  test('auto-grows when meshCount exceeds capacity', () => {
    const state = createSortState(2)
    const meshes = [
      mockMesh({ x: 30, y: 0, z: 0, materialId: 1 }),
      mockMesh({ x: 10, y: 0, z: 0, materialId: 1 }),
      mockMesh({ x: 20, y: 0, z: 0, materialId: 1 }),
      mockMesh({ x: 5, y: 0, z: 0, materialId: 1 }),
    ]
    const camera = mockCamera(0, 0, 0, 100)
    // Should not throw despite initial capacity of 2
    sortMeshes(state, meshes, 4, camera)

    expect(state.capacity).toBeGreaterThanOrEqual(4)
    const sorted = Array.from(state.indices.subarray(0, 4)).map(i => meshes[i]!)
    const distances = sorted.map(m => m._worldMatrix[12]!)
    expect(distances[0]).toBeLessThanOrEqual(distances[1]!)
    expect(distances[1]).toBeLessThanOrEqual(distances[2]!)
    expect(distances[2]).toBeLessThanOrEqual(distances[3]!)
  })

  test('sort is stable (preserves insertion order for equal keys)', () => {
    const state = createSortState(10)
    // All identical keys — sort should preserve original order
    const meshes = [
      mockMesh({ x: 10, y: 0, z: 0, materialId: 1 }),
      mockMesh({ x: 10, y: 0, z: 0, materialId: 1 }),
      mockMesh({ x: 10, y: 0, z: 0, materialId: 1 }),
    ]
    const camera = mockCamera(0, 0, 0, 100)
    sortMeshes(state, meshes, 3, camera)

    expect(state.indices[0]).toBe(0)
    expect(state.indices[1]).toBe(1)
    expect(state.indices[2]).toBe(2)
  })
})
