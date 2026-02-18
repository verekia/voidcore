import type { PerspectiveCamera } from '../scene/camera.ts'
import type { Mesh } from '../scene/mesh.ts'

export interface SortState {
  keys: Uint32Array
  indices: Uint32Array
  temp: Uint32Array
  tempIndices: Uint32Array
  counts: Uint32Array
}

export const createSortState = (maxObjects: number): SortState => ({
  keys: new Uint32Array(maxObjects),
  indices: new Uint32Array(maxObjects),
  temp: new Uint32Array(maxObjects),
  tempIndices: new Uint32Array(maxObjects),
  counts: new Uint32Array(256),
})

export const sortMeshes = (state: SortState, meshes: Mesh[], meshCount: number, camera: PerspectiveCamera): void => {
  if (meshCount === 0) return

  const { keys, indices, temp, tempIndices, counts } = state

  // Camera position from view matrix inverse (world position is in _worldMatrix elements 12,13,14)
  const camX = camera._worldMatrix[12]!
  const camY = camera._worldMatrix[13]!
  const camZ = camera._worldMatrix[14]!
  const invFar = 1 / camera.far

  // Build sort keys and initial indices
  for (let i = 0; i < meshCount; i++) {
    const mesh = meshes[i]!
    const material = mesh.material
    const hasSkeleton = !!(mesh.skeleton && mesh.geometry.joints && mesh.geometry.weights)

    // Layer: bits 31-30 (opaque=0, transparent=1)
    const layer = material.transparent ? 1 : 0

    // Pipeline ID: bits 29-22
    const pipelineId = (material.type === 'lambert' ? 1 : 0) | (hasSkeleton ? 2 : 0)

    // Material ID: bits 21-10 (masked to 12 bits)
    const materialId = material._id & 0xfff

    // Depth: bits 9-0 (10 bits, quantized distance)
    const wm = mesh._worldMatrix
    const dx = wm[12]! - camX
    const dy = wm[13]! - camY
    const dz = wm[14]! - camZ
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)
    const depth = Math.min(dist * invFar * 1023, 1023) | 0

    keys[i] = (layer << 30) | (pipelineId << 22) | (materialId << 10) | depth
    indices[i] = i
  }

  // 4-pass LSD radix sort (8-bit radix per pass)
  for (let pass = 0; pass < 4; pass++) {
    const shift = pass * 8

    // Clear histogram
    counts.fill(0)

    // Build histogram
    for (let i = 0; i < meshCount; i++) {
      const bucket = (keys[indices[i]!]! >>> shift) & 0xff
      counts[bucket]!++
    }

    // Prefix sum
    let sum = 0
    for (let i = 0; i < 256; i++) {
      const c = counts[i]!
      counts[i] = sum
      sum += c
    }

    // Scatter
    for (let i = 0; i < meshCount; i++) {
      const idx = indices[i]!
      const bucket = (keys[idx]! >>> shift) & 0xff
      const dest = counts[bucket]!
      temp[dest] = keys[idx]!
      tempIndices[dest] = idx
      counts[bucket] = dest + 1
    }

    // Copy back (swap references would be better but typed arrays can't be swapped)
    for (let i = 0; i < meshCount; i++) {
      indices[i] = tempIndices[i]!
    }
  }
}
