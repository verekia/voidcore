// Mesh Sorting – Sorts meshes to minimize GPU state changes and ensure correct draw order.
//
// Changing the active shader program, material, or blend mode on the GPU is expensive.
// By sorting meshes before drawing, we group identical states together and reduce
// these costly switches.
//
// Sort key layout (32-bit unsigned integer):
//   Bits 31-30: (reserved)
//   Bits 29-22: Pipeline ID – Groups by shader program (lambert/basic × skinned/static)
//   Bits 21-10: Material ID – Groups by material to reduce uniform uploads
//   Bits 9-0:   Depth       – Nearest first (early-Z optimization).
//
// Uses a 4-pass LSD (Least Significant Digit) radix sort with an 8-bit radix.
// Radix sort is O(n) and stable, making it ideal for the thousands of meshes a scene
// might contain. The sort state (buffers, histograms) is pre-allocated to avoid
// per-frame allocations. Passes ping-pong between two index arrays to eliminate
// per-pass copy-back overhead.
//
// sortMeshes() – Sorts meshes in-place by the composite key described above.

import type { PerspectiveCamera } from '../scene/camera.ts'
import type { Mesh } from '../scene/mesh.ts'

export interface SortState {
  keys: Uint32Array
  indices: Uint32Array
  tempIndices: Uint32Array
  counts: Uint32Array
  capacity: number
}

export const createSortState = (maxObjects: number): SortState => ({
  keys: new Uint32Array(maxObjects),
  indices: new Uint32Array(maxObjects),
  tempIndices: new Uint32Array(maxObjects),
  counts: new Uint32Array(256),
  capacity: maxObjects,
})

/** Grow sort buffers if needed. Uses doubling strategy to amortize allocations. */
const ensureSortCapacity = (state: SortState, needed: number): void => {
  if (needed <= state.capacity) return
  const cap = Math.max(needed, state.capacity * 2)
  state.keys = new Uint32Array(cap)
  state.indices = new Uint32Array(cap)
  state.tempIndices = new Uint32Array(cap)
  state.capacity = cap
}

export const sortMeshes = (state: SortState, meshes: Mesh[], meshCount: number, camera: PerspectiveCamera): void => {
  if (meshCount === 0) return
  ensureSortCapacity(state, meshCount)

  const { keys, indices, tempIndices, counts } = state

  // Camera position from view matrix inverse (world position is in _worldMatrix elements 12,13,14)
  const camX = camera._worldMatrix[12]!
  const camY = camera._worldMatrix[13]!
  const camZ = camera._worldMatrix[14]!
  const invFar = 1 / camera.far

  // Build sort keys and initial indices
  for (let i = 0; i < meshCount; i++) {
    const mesh = meshes[i]!
    const material = mesh.material
    mesh._isSkinned = !!mesh.skeleton && !!mesh.geometry.joints && !!mesh.geometry.weights

    // Pipeline ID: bits 29-22
    const pipelineId = (material.type === 'lambert' ? 1 : 0) | (mesh._isSkinned ? 2 : 0)

    // Material ID: bits 21-10 (masked to 12 bits)
    const materialId = material._id & 0xfff

    // Depth: bits 9-0 (10 bits, quantized distance, nearest first for early-Z)
    const wm = mesh._worldMatrix
    const dx = wm[12]! - camX
    const dy = wm[13]! - camY
    const dz = wm[14]! - camZ
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)
    const depth = Math.min(dist * invFar * 1023, 1023) | 0

    keys[i] = (pipelineId << 22) | (materialId << 10) | depth
    indices[i] = i
  }

  // 4-pass LSD radix sort (8-bit radix per pass)
  // Ping-pong between indices and tempIndices to avoid per-pass copy-back.
  // Even passes: read from src, scatter to dst.
  // After 4 passes (even count), result is back in indices.
  let src = indices
  let dst = tempIndices

  for (let pass = 0; pass < 4; pass++) {
    const shift = pass * 8

    // Clear histogram
    counts.fill(0)

    // Build histogram
    for (let i = 0; i < meshCount; i++) {
      const bucket = (keys[src[i]!]! >>> shift) & 0xff
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
      const idx = src[i]!
      const bucket = (keys[idx]! >>> shift) & 0xff
      const dest = counts[bucket]!
      dst[dest] = idx
      counts[bucket] = dest + 1
    }

    // Swap src and dst for next pass
    const tmp = src
    src = dst
    dst = tmp
  }
  // After 4 passes, src points to 'indices' (since we swapped an even number of times).
  // Result is already in state.indices — no final copy needed.
}
