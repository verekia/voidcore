// Raycasting & BVH – Shoots rays into the scene and finds which triangles they hit.
//
// Raycasting shoots an invisible line (ray) into the 3D scene and finds which triangles it
// intersects. Common uses include collision detection, line-of-sight checks, ground placement
// (projecting objects onto terrain), pseudo-physics (e.g. gravity via downward rays), and
// mouse/pointer picking.
//
// To avoid testing every triangle of every mesh (too slow), we use a BVH (Bounding Volume
// Hierarchy): a tree of nested bounding boxes. We first test the ray against large boxes,
// then progressively smaller ones, and only test actual triangles when we reach a leaf node.
// This turns O(n) triangle tests into roughly O(log n).
//
// BVH construction uses "binned SAH" (Surface Area Heuristic): it evaluates multiple ways
// to split triangles along each axis, picks the split that minimizes expected ray-test cost,
// and recurses. The resulting tree is stored in a flat array for cache-friendly traversal.
//
// buildMeshBVH()            – Builds a BVH for a mesh's triangles (done once, cached).
// prebuildBVH()             – Pre-builds and caches a BVH for a geometry (avoids first-raycast stall).
// setBVH()                  – Injects an externally-built BVH (e.g. from a web worker) into the cache.
// Raycaster.set()           – Sets the ray from an origin and direction.
// Raycaster.setFromCamera() – Creates a ray from screen coordinates through the camera.
// Raycaster.intersectObject() / intersectObjects() – Returns sorted hits (nearest first).
// createRaycastHit()           – Creates a pre-allocated hit for zero-allocation raycasting.
//
// Key algorithms:
//   Ray-AABB:     Slab method – tests ray against axis-aligned box entry/exit distances.
//   Ray-Triangle: Möller-Trumbore – fast ray-triangle intersection using cross products.

import { mat4Create, mat4Invert, vec3TransformMat4 } from '../math/index'

import type { Geometry } from '../geometry/geometry'
import type { Vec2, Vec3 } from '../math/index'
import type { PerspectiveCamera } from '../scene/camera'
import type { Mesh } from '../scene/mesh'
import type { Node } from '../scene/node'

// ─── Public Types ──────────────────────────────────────────────────────────────

export interface RaycastHit {
  distance: number // World-space distance from ray origin
  point: Vec3 // World-space intersection point
  normal: Vec3 // Interpolated surface normal (world space)
  uv: Vec2 | null // Interpolated UV (if geometry has UVs)
  triangleIndex: number // Index of the triangle that was hit
  object: Mesh | null // The mesh that was hit
}

/** Create a pre-allocated RaycastHit to reuse across frames (avoids per-hit allocations). */
export const createRaycastHit = (): RaycastHit => ({
  distance: 0,
  point: new Float32Array(3) as Vec3,
  normal: new Float32Array(3) as Vec3,
  uv: new Float32Array(2) as Vec2,
  triangleIndex: -1,
  object: null,
})

// ─── BVH Types ────────────────────────────────────────────────────────────────

export interface MeshBVH {
  floatNodes: Float32Array // BVH node AABB bounds (float view)
  intNodes: Int32Array // BVH node child/count data (int view, same buffer)
  triOrder: Uint32Array // Reordered triangle indices for cache-friendly access
}

// ─── Constants ─────────────────────────────────────────────────────────────────

const BIN_COUNT = 12
const MAX_LEAF_TRIANGLES = 4
const TRAVERSAL_COST = 1.0
const INTERSECTION_COST = 1.5
const EPSILON = 1e-7

// ─── Module-level pre-allocated buffers ────────────────────────────────────────

// Pre-allocated traversal stack: depth of log2(N/4) * 2 ≈ 30 for 100K triangles
const _stack = new Int32Array(128)
const _invWorldMat = mat4Create()

// Mutable outputs from rayIntersectsTriangle (set on each successful hit)
let _lastU = 0
let _lastV = 0

// ─── BVH Node Layout ───────────────────────────────────────────────────────────
// 32 bytes per node, 8 floats / 8 ints (aliased over same buffer):
//   floats[o+0..2] = minX, minY, minZ
//   ints[o+3]      = rightChildOrCount  (<0: leaf with -triCount, >0: right child index)
//   floats[o+4..6] = maxX, maxY, maxZ
//   ints[o+7]      = leftChildOrStart   (inner: left child index, leaf: first tri in triOrder)

// ─── Ray-AABB Intersection (slab method) ──────────────────────────────────────
// Returns tmin (entry distance) if hit and tmin < maxDist, else -1

const rayIntersectsAABBNode = (
  floatNodes: Float32Array,
  o: number, // node offset (nodeIndex * 8)
  ox: number,
  oy: number,
  oz: number, // ray origin
  idx: number,
  idy: number,
  idz: number, // ray inverse direction
  maxDist: number,
): number => {
  const t1 = (floatNodes[o]! - ox) * idx
  const t2 = (floatNodes[o + 4]! - ox) * idx
  const t3 = (floatNodes[o + 1]! - oy) * idy
  const t4 = (floatNodes[o + 5]! - oy) * idy
  const t5 = (floatNodes[o + 2]! - oz) * idz
  const t6 = (floatNodes[o + 6]! - oz) * idz

  const tmin = Math.max(Math.min(t1, t2), Math.min(t3, t4), Math.min(t5, t6))
  const tmax = Math.min(Math.max(t1, t2), Math.max(t3, t4), Math.max(t5, t6))

  if (tmax < Math.max(0, tmin) || tmin >= maxDist) return -1
  return tmin
}

// ─── Ray-Triangle Intersection (Möller-Trumbore) ──────────────────────────────
// Returns distance t if hit (t > 0), else -1.
// Sets _lastU, _lastV (barycentric coords) on hit for normal/UV interpolation.

const rayIntersectsTriangle = (
  ox: number,
  oy: number,
  oz: number, // ray origin
  dx: number,
  dy: number,
  dz: number, // ray direction
  positions: Float32Array,
  indices: Uint16Array | Uint32Array,
  triIndex: number,
): number => {
  const i0 = indices[triIndex * 3]! * 3
  const i1 = indices[triIndex * 3 + 1]! * 3
  const i2 = indices[triIndex * 3 + 2]! * 3

  const v0x = positions[i0]!,
    v0y = positions[i0 + 1]!,
    v0z = positions[i0 + 2]!
  const e1x = positions[i1]! - v0x,
    e1y = positions[i1 + 1]! - v0y,
    e1z = positions[i1 + 2]! - v0z
  const e2x = positions[i2]! - v0x,
    e2y = positions[i2 + 1]! - v0y,
    e2z = positions[i2 + 2]! - v0z

  // h = cross(dir, e2)
  const hx = dy * e2z - dz * e2y
  const hy = dz * e2x - dx * e2z
  const hz = dx * e2y - dy * e2x

  const det = e1x * hx + e1y * hy + e1z * hz
  if (Math.abs(det) < EPSILON) return -1 // Ray parallel to triangle

  const invDet = 1 / det

  // s = origin - v0
  const sx = ox - v0x,
    sy = oy - v0y,
    sz = oz - v0z

  const u = (sx * hx + sy * hy + sz * hz) * invDet
  if (u < 0 || u > 1) return -1

  // q = cross(s, e1)
  const qx = sy * e1z - sz * e1y
  const qy = sz * e1x - sx * e1z
  const qz = sx * e1y - sy * e1x

  const v = (dx * qx + dy * qy + dz * qz) * invDet
  if (v < 0 || u + v > 1) return -1

  const t = (e2x * qx + e2y * qy + e2z * qz) * invDet
  if (t < EPSILON) return -1

  _lastU = u
  _lastV = v
  return t
}

// ─── BVH Construction ──────────────────────────────────────────────────────────

const surfaceArea = (minX: number, minY: number, minZ: number, maxX: number, maxY: number, maxZ: number): number => {
  const ex = maxX - minX,
    ey = maxY - minY,
    ez = maxZ - minZ
  return 2 * (ex * ey + ey * ez + ez * ex)
}

export const buildMeshBVH = (positions: Float32Array, indices: Uint16Array | Uint32Array): MeshBVH => {
  const triCount = (indices.length / 3) | 0

  // Per-triangle centroids [cx,cy,cz] and AABBs [minX..maxZ]
  const centroids = new Float32Array(triCount * 3)
  const triAABBs = new Float32Array(triCount * 6)

  for (let t = 0; t < triCount; t++) {
    const i0 = indices[t * 3]! * 3
    const i1 = indices[t * 3 + 1]! * 3
    const i2 = indices[t * 3 + 2]! * 3
    const x0 = positions[i0]!,
      y0 = positions[i0 + 1]!,
      z0 = positions[i0 + 2]!
    const x1 = positions[i1]!,
      y1 = positions[i1 + 1]!,
      z1 = positions[i1 + 2]!
    const x2 = positions[i2]!,
      y2 = positions[i2 + 1]!,
      z2 = positions[i2 + 2]!

    centroids[t * 3] = (x0 + x1 + x2) / 3
    centroids[t * 3 + 1] = (y0 + y1 + y2) / 3
    centroids[t * 3 + 2] = (z0 + z1 + z2) / 3

    triAABBs[t * 6] = Math.min(x0, x1, x2)
    triAABBs[t * 6 + 1] = Math.min(y0, y1, y2)
    triAABBs[t * 6 + 2] = Math.min(z0, z1, z2)
    triAABBs[t * 6 + 3] = Math.max(x0, x1, x2)
    triAABBs[t * 6 + 4] = Math.max(y0, y1, y2)
    triAABBs[t * 6 + 5] = Math.max(z0, z1, z2)
  }

  // Working array partitioned in-place during build
  const triOrder = new Uint32Array(triCount)
  for (let i = 0; i < triCount; i++) triOrder[i] = i

  // Pre-allocate max possible nodes (2 * triCount is a safe upper bound)
  const maxNodes = Math.max(1, triCount * 2)
  const nodeBuffer = new ArrayBuffer(maxNodes * 32)
  const floatNodes = new Float32Array(nodeBuffer)
  const intNodes = new Int32Array(nodeBuffer)
  let nodeCount = 0

  // Bin data reused across recursive calls (heap-allocated once)
  const binMin = new Float32Array(BIN_COUNT * 3) // [minX, minY, minZ] per bin
  const binMax = new Float32Array(BIN_COUNT * 3) // [maxX, maxY, maxZ] per bin
  const binCnt = new Int32Array(BIN_COUNT)

  const prefixMin = new Float32Array(BIN_COUNT * 3)
  const prefixMax = new Float32Array(BIN_COUNT * 3)
  const prefixCnt = new Int32Array(BIN_COUNT)
  const suffixMin = new Float32Array(BIN_COUNT * 3)
  const suffixMax = new Float32Array(BIN_COUNT * 3)
  const suffixCnt = new Int32Array(BIN_COUNT)

  const buildNode = (start: number, end: number): number => {
    const nodeIdx = nodeCount++
    const count = end - start
    const o = nodeIdx * 8

    // Compute AABB of all triangles in this range
    let minX = Infinity,
      minY = Infinity,
      minZ = Infinity
    let maxX = -Infinity,
      maxY = -Infinity,
      maxZ = -Infinity
    for (let i = start; i < end; i++) {
      const t = triOrder[i]! * 6
      if (triAABBs[t]! < minX) minX = triAABBs[t]!
      if (triAABBs[t + 1]! < minY) minY = triAABBs[t + 1]!
      if (triAABBs[t + 2]! < minZ) minZ = triAABBs[t + 2]!
      if (triAABBs[t + 3]! > maxX) maxX = triAABBs[t + 3]!
      if (triAABBs[t + 4]! > maxY) maxY = triAABBs[t + 4]!
      if (triAABBs[t + 5]! > maxZ) maxZ = triAABBs[t + 5]!
    }

    floatNodes[o] = minX
    floatNodes[o + 1] = minY
    floatNodes[o + 2] = minZ
    floatNodes[o + 4] = maxX
    floatNodes[o + 5] = maxY
    floatNodes[o + 6] = maxZ

    // Create leaf if small enough
    if (count <= MAX_LEAF_TRIANGLES) {
      intNodes[o + 3] = -count
      intNodes[o + 7] = start
      return nodeIdx
    }

    // Compute centroid AABB for SAH binning
    let cMinX = Infinity,
      cMinY = Infinity,
      cMinZ = Infinity
    let cMaxX = -Infinity,
      cMaxY = -Infinity,
      cMaxZ = -Infinity
    for (let i = start; i < end; i++) {
      const t = triOrder[i]! * 3
      const cx = centroids[t]!,
        cy = centroids[t + 1]!,
        cz = centroids[t + 2]!
      if (cx < cMinX) cMinX = cx
      if (cy < cMinY) cMinY = cy
      if (cz < cMinZ) cMinZ = cz
      if (cx > cMaxX) cMaxX = cx
      if (cy > cMaxY) cMaxY = cy
      if (cz > cMaxZ) cMaxZ = cz
    }

    const parentSA = surfaceArea(minX, minY, minZ, maxX, maxY, maxZ)
    const leafCost = count * INTERSECTION_COST

    let bestCost = Infinity
    let bestAxis = -1
    let bestBin = -1

    // Evaluate SAH for each axis
    for (let axis = 0; axis < 3; axis++) {
      const cMin = axis === 0 ? cMinX : axis === 1 ? cMinY : cMinZ
      const cMax = axis === 0 ? cMaxX : axis === 1 ? cMaxY : cMaxZ
      if (cMax - cMin < 1e-6) continue

      const binInvSize = BIN_COUNT / (cMax - cMin)

      // Initialize bins
      for (let b = 0; b < BIN_COUNT; b++) {
        binMin[b * 3] = Infinity
        binMin[b * 3 + 1] = Infinity
        binMin[b * 3 + 2] = Infinity
        binMax[b * 3] = -Infinity
        binMax[b * 3 + 1] = -Infinity
        binMax[b * 3 + 2] = -Infinity
        binCnt[b] = 0
      }

      // Distribute triangles into bins by centroid
      for (let i = start; i < end; i++) {
        const t = triOrder[i]!
        const c = centroids[t * 3 + axis]!
        let b = ((c - cMin) * binInvSize) | 0
        if (b >= BIN_COUNT) b = BIN_COUNT - 1

        binCnt[b] = (binCnt[b] ?? 0) + 1
        const ta = t * 6
        const bm = b * 3
        if (triAABBs[ta]! < binMin[bm]!) binMin[bm] = triAABBs[ta]!
        if (triAABBs[ta + 1]! < binMin[bm + 1]!) binMin[bm + 1] = triAABBs[ta + 1]!
        if (triAABBs[ta + 2]! < binMin[bm + 2]!) binMin[bm + 2] = triAABBs[ta + 2]!
        if (triAABBs[ta + 3]! > binMax[bm]!) binMax[bm] = triAABBs[ta + 3]!
        if (triAABBs[ta + 4]! > binMax[bm + 1]!) binMax[bm + 1] = triAABBs[ta + 4]!
        if (triAABBs[ta + 5]! > binMax[bm + 2]!) binMax[bm + 2] = triAABBs[ta + 5]!
      }

      // Prefix scan (left half for each split)
      prefixCnt[0] = binCnt[0]!
      prefixMin[0] = binMin[0]!
      prefixMin[1] = binMin[1]!
      prefixMin[2] = binMin[2]!
      prefixMax[0] = binMax[0]!
      prefixMax[1] = binMax[1]!
      prefixMax[2] = binMax[2]!
      for (let b = 1; b < BIN_COUNT; b++) {
        prefixCnt[b] = prefixCnt[b - 1]! + binCnt[b]!
        const p = b * 3,
          pp = (b - 1) * 3,
          bm = b * 3
        prefixMin[p] = Math.min(prefixMin[pp]!, binMin[bm]!)
        prefixMin[p + 1] = Math.min(prefixMin[pp + 1]!, binMin[bm + 1]!)
        prefixMin[p + 2] = Math.min(prefixMin[pp + 2]!, binMin[bm + 2]!)
        prefixMax[p] = Math.max(prefixMax[pp]!, binMax[bm]!)
        prefixMax[p + 1] = Math.max(prefixMax[pp + 1]!, binMax[bm + 1]!)
        prefixMax[p + 2] = Math.max(prefixMax[pp + 2]!, binMax[bm + 2]!)
      }

      // Suffix scan (right half for each split)
      const last = BIN_COUNT - 1
      suffixCnt[last] = binCnt[last]!
      suffixMin[last * 3] = binMin[last * 3]!
      suffixMin[last * 3 + 1] = binMin[last * 3 + 1]!
      suffixMin[last * 3 + 2] = binMin[last * 3 + 2]!
      suffixMax[last * 3] = binMax[last * 3]!
      suffixMax[last * 3 + 1] = binMax[last * 3 + 1]!
      suffixMax[last * 3 + 2] = binMax[last * 3 + 2]!
      for (let b = last - 1; b >= 0; b--) {
        suffixCnt[b] = suffixCnt[b + 1]! + binCnt[b]!
        const s = b * 3,
          sn = (b + 1) * 3,
          bm = b * 3
        suffixMin[s] = Math.min(suffixMin[sn]!, binMin[bm]!)
        suffixMin[s + 1] = Math.min(suffixMin[sn + 1]!, binMin[bm + 1]!)
        suffixMin[s + 2] = Math.min(suffixMin[sn + 2]!, binMin[bm + 2]!)
        suffixMax[s] = Math.max(suffixMax[sn]!, binMax[bm]!)
        suffixMax[s + 1] = Math.max(suffixMax[sn + 1]!, binMax[bm + 1]!)
        suffixMax[s + 2] = Math.max(suffixMax[sn + 2]!, binMax[bm + 2]!)
      }

      // Evaluate 11 candidate split planes
      for (let b = 0; b < BIN_COUNT - 1; b++) {
        const leftCnt = prefixCnt[b]!
        const rightCnt = suffixCnt[b + 1]!
        if (leftCnt === 0 || rightCnt === 0) continue

        const p = b * 3,
          sn = (b + 1) * 3
        const leftSA = surfaceArea(
          prefixMin[p]!,
          prefixMin[p + 1]!,
          prefixMin[p + 2]!,
          prefixMax[p]!,
          prefixMax[p + 1]!,
          prefixMax[p + 2]!,
        )
        const rightSA = surfaceArea(
          suffixMin[sn]!,
          suffixMin[sn + 1]!,
          suffixMin[sn + 2]!,
          suffixMax[sn]!,
          suffixMax[sn + 1]!,
          suffixMax[sn + 2]!,
        )

        const cost =
          TRAVERSAL_COST +
          (leftSA / parentSA) * leftCnt * INTERSECTION_COST +
          (rightSA / parentSA) * rightCnt * INTERSECTION_COST

        if (cost < bestCost) {
          bestCost = cost
          bestAxis = axis
          bestBin = b
        }
      }
    }

    // If no split improves on leaf cost, create leaf
    if (bestAxis === -1 || bestCost >= leafCost) {
      intNodes[o + 3] = -count
      intNodes[o + 7] = start
      return nodeIdx
    }

    // Partition triOrder in-place around the best split bin
    const cMin = bestAxis === 0 ? cMinX : bestAxis === 1 ? cMinY : cMinZ
    const cMax = bestAxis === 0 ? cMaxX : bestAxis === 1 ? cMaxY : cMaxZ
    const binInvSize = BIN_COUNT / (cMax - cMin)

    let mid = start
    for (let i = start; i < end; i++) {
      const t = triOrder[i]!
      const c = centroids[t * 3 + bestAxis]!
      let b = ((c - cMin) * binInvSize) | 0
      if (b >= BIN_COUNT) b = BIN_COUNT - 1
      if (b <= bestBin) {
        const tmp = triOrder[mid]!
        triOrder[mid] = triOrder[i]!
        triOrder[i] = tmp
        mid++
      }
    }

    // Degenerate fallback: force midpoint split
    if (mid === start || mid === end) mid = (start + end) >> 1

    // Recurse: left child is always nodeIdx+1 (depth-first layout)
    const leftChild = buildNode(start, mid)
    const rightChild = buildNode(mid, end)

    // Write inner node (o is still valid — pre-allocated buffer)
    intNodes[o + 3] = rightChild
    intNodes[o + 7] = leftChild

    return nodeIdx
  }

  buildNode(0, triCount)

  // Trim to actual used nodes
  const usedBuffer = new ArrayBuffer(nodeCount * 32)
  const usedFloat = new Float32Array(usedBuffer)
  usedFloat.set(floatNodes.subarray(0, nodeCount * 8))
  return {
    floatNodes: usedFloat,
    intNodes: new Int32Array(usedBuffer),
    triOrder,
  }
}

// ─── BVH Cache ─────────────────────────────────────────────────────────────────

const _bvhCache = new WeakMap<Geometry, MeshBVH>()

const getMeshBVH = (geometry: Geometry): MeshBVH => {
  if (geometry.needsUpdate) _bvhCache.delete(geometry)
  let bvh = _bvhCache.get(geometry)
  if (!bvh) {
    bvh = buildMeshBVH(geometry.positions, geometry.indices)
    _bvhCache.set(geometry, bvh)
  }
  return bvh
}

export const prebuildBVH = (geometry: Geometry): void => {
  getMeshBVH(geometry)
}

/** Inject an externally-built BVH (e.g. from a web worker) into the cache for a geometry. */
export const setBVH = (geometry: Geometry, bvh: MeshBVH): void => {
  _bvhCache.set(geometry, bvh)
}

// ─── Mesh Intersection ─────────────────────────────────────────────────────────

const _localOrigin = new Float32Array(3)

// Pre-allocated hit used for internal allocating path
const _tempHit: RaycastHit = {
  distance: 0,
  point: new Float32Array(3) as Vec3,
  normal: new Float32Array(3) as Vec3,
  uv: new Float32Array(2) as Vec2,
  triangleIndex: -1,
  object: null,
}

const intersectMeshInto = (
  worldOx: number,
  worldOy: number,
  worldOz: number,
  worldDx: number,
  worldDy: number,
  worldDz: number,
  mesh: Mesh,
  hit: RaycastHit,
): boolean => {
  if (!mat4Invert(_invWorldMat, mesh._worldMatrix)) return false

  // Transform ray origin to local space (point transform, with translation)
  _localOrigin[0] = worldOx
  _localOrigin[1] = worldOy
  _localOrigin[2] = worldOz
  vec3TransformMat4(_localOrigin, _localOrigin, _invWorldMat)
  const lox = _localOrigin[0]!,
    loy = _localOrigin[1]!,
    loz = _localOrigin[2]!

  // Transform ray direction to local space (direction, no translation)
  const m = _invWorldMat
  const ldx = m[0]! * worldDx + m[4]! * worldDy + m[8]! * worldDz
  const ldy = m[1]! * worldDx + m[5]! * worldDy + m[9]! * worldDz
  const ldz = m[2]! * worldDx + m[6]! * worldDy + m[10]! * worldDz

  const lidx = 1 / (ldx || 1e-20),
    lidy = 1 / (ldy || 1e-20),
    lidz = 1 / (ldz || 1e-20)

  const bvh = getMeshBVH(mesh.geometry)
  const { floatNodes, intNodes, triOrder } = bvh
  const { positions, indices } = mesh.geometry

  // Stack-based BVH traversal
  let stackPtr = 0
  _stack[stackPtr++] = 0

  let closestDist = Infinity
  let closestTriIdx = -1
  let closestU = 0
  let closestV = 0

  while (stackPtr > 0) {
    const nodeIdx = _stack[--stackPtr]!
    const o = nodeIdx * 8

    const tmin = rayIntersectsAABBNode(floatNodes, o, lox, loy, loz, lidx, lidy, lidz, closestDist)
    if (tmin < 0) continue

    const rightOrCount = intNodes[o + 3]!
    const leftOrStart = intNodes[o + 7]!

    if (rightOrCount < 0) {
      // Leaf: test all triangles
      const count = -rightOrCount
      const triStart = leftOrStart
      for (let i = triStart; i < triStart + count; i++) {
        const triIdx = triOrder[i]!
        const t = rayIntersectsTriangle(lox, loy, loz, ldx, ldy, ldz, positions, indices, triIdx)
        if (t > 0 && t < closestDist) {
          closestDist = t
          closestTriIdx = triIdx
          closestU = _lastU
          closestV = _lastV
        }
      }
    } else {
      // Inner node: push children with near-first ordering
      const leftChild = leftOrStart
      const rightChild = rightOrCount

      const tLeft = rayIntersectsAABBNode(floatNodes, leftChild * 8, lox, loy, loz, lidx, lidy, lidz, closestDist)
      const tRight = rayIntersectsAABBNode(floatNodes, rightChild * 8, lox, loy, loz, lidx, lidy, lidz, closestDist)

      if (tLeft >= 0 && tRight >= 0) {
        // Push far child first so near child is popped first
        if (tLeft <= tRight) {
          _stack[stackPtr++] = rightChild
          _stack[stackPtr++] = leftChild
        } else {
          _stack[stackPtr++] = leftChild
          _stack[stackPtr++] = rightChild
        }
      } else if (tLeft >= 0) {
        _stack[stackPtr++] = leftChild
      } else if (tRight >= 0) {
        _stack[stackPtr++] = rightChild
      }
    }
  }

  if (closestTriIdx < 0) return false

  // Local-space hit point
  const lhx = lox + ldx * closestDist
  const lhy = loy + ldy * closestDist
  const lhz = loz + ldz * closestDist

  // Transform hit point to world space
  const hitPoint = hit.point
  hitPoint[0] = lhx
  hitPoint[1] = lhy
  hitPoint[2] = lhz
  vec3TransformMat4(hitPoint, hitPoint, mesh._worldMatrix)

  // World-space distance from ray origin
  const dx = hitPoint[0]! - worldOx,
    dy = hitPoint[1]! - worldOy,
    dz = hitPoint[2]! - worldOz

  hit.distance = Math.sqrt(dx * dx + dy * dy + dz * dz)
  hit.triangleIndex = closestTriIdx
  hit.object = mesh

  // Interpolate surface normal using barycentric coordinates (u, v, 1-u-v)
  const { normals } = mesh.geometry
  const i0 = indices[closestTriIdx * 3]! * 3
  const i1 = indices[closestTriIdx * 3 + 1]! * 3
  const i2 = indices[closestTriIdx * 3 + 2]! * 3
  const w = 1 - closestU - closestV
  const lnx = normals[i0]! * w + normals[i1]! * closestU + normals[i2]! * closestV
  const lny = normals[i0 + 1]! * w + normals[i1 + 1]! * closestU + normals[i2 + 1]! * closestV
  const lnz = normals[i0 + 2]! * w + normals[i1 + 2]! * closestU + normals[i2 + 2]! * closestV

  // Transform normal to world space using transpose(invWorld) = normalMatrix
  // For column-major invWorldMat: transpose-3x3 rows are its columns
  const im = _invWorldMat
  const wnx = im[0]! * lnx + im[1]! * lny + im[2]! * lnz
  const wny = im[4]! * lnx + im[5]! * lny + im[6]! * lnz
  const wnz = im[8]! * lnx + im[9]! * lny + im[10]! * lnz
  const nlen = Math.sqrt(wnx * wnx + wny * wny + wnz * wnz)

  const hitNormal = hit.normal
  if (nlen > 1e-6) {
    const inv = 1 / nlen
    hitNormal[0] = wnx * inv
    hitNormal[1] = wny * inv
    hitNormal[2] = wnz * inv
  } else {
    hitNormal[0] = 0
    hitNormal[1] = 0
    hitNormal[2] = 0
  }

  // Interpolate UV if available
  if (mesh.geometry.uvs) {
    const uvs = mesh.geometry.uvs
    const u0 = indices[closestTriIdx * 3]! * 2
    const u1 = indices[closestTriIdx * 3 + 1]! * 2
    const u2 = indices[closestTriIdx * 3 + 2]! * 2
    if (!hit.uv) hit.uv = new Float32Array(2) as Vec2
    hit.uv[0] = uvs[u0]! * w + uvs[u1]! * closestU + uvs[u2]! * closestV
    hit.uv[1] = uvs[u0 + 1]! * w + uvs[u1 + 1]! * closestU + uvs[u2 + 1]! * closestV
  } else {
    hit.uv = null
  }

  return true
}

// ─── Raycaster ─────────────────────────────────────────────────────────────────

export class Raycaster {
  origin: Float32Array = new Float32Array(3)
  direction: Float32Array = new Float32Array(3)
  _invDir: Float32Array = new Float32Array([Infinity, Infinity, Infinity])

  /** Set ray from arbitrary origin and direction (direction will be normalized). */
  set(origin: Vec3, direction: Vec3): void {
    this.origin[0] = origin[0]!
    this.origin[1] = origin[1]!
    this.origin[2] = origin[2]!
    const dx = direction[0]!,
      dy = direction[1]!,
      dz = direction[2]!
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz)
    if (len > 1e-6) {
      const inv = 1 / len
      this.direction[0] = dx * inv
      this.direction[1] = dy * inv
      this.direction[2] = dz * inv
    }
    this._invDir[0] = 1 / (this.direction[0]! || 1e-20)
    this._invDir[1] = 1 / (this.direction[1]! || 1e-20)
    this._invDir[2] = 1 / (this.direction[2]! || 1e-20)
  }

  /**
   * Set ray from camera and NDC screen coordinates.
   * NDC: x and y in [-1, +1], where (0,0) is screen center.
   * Y is +1 at top of screen (flipped from clientY).
   */
  setFromCamera(ndc: [number, number], camera: PerspectiveCamera): void {
    // Ray origin: camera world position
    this.origin[0] = camera.position[0]!
    this.origin[1] = camera.position[1]!
    this.origin[2] = camera.position[2]!

    // Compute view-space ray direction from NDC.
    // Camera looks in -Z in view space (standard right-handed convention).
    // At z=-1 (view space), the frustum extents are:
    //   x ∈ [-aspect * tan(fovY/2), +aspect * tan(fovY/2)]
    //   y ∈ [-tan(fovY/2), +tan(fovY/2)]
    const tanHalfFov = Math.tan(((camera.fov * Math.PI) / 180) * 0.5)
    const vx = ndc[0] * camera.aspect * tanHalfFov
    const vy = ndc[1] * tanHalfFov
    const vz = -1

    // Transform view-space direction to world space via transpose of view matrix
    // (view matrix rows = camera world axes, transpose gives world-space transform)
    const V = camera._viewMatrix
    const wx = V[0]! * vx + V[1]! * vy + V[2]! * vz
    const wy = V[4]! * vx + V[5]! * vy + V[6]! * vz
    const wz = V[8]! * vx + V[9]! * vy + V[10]! * vz

    const len = Math.sqrt(wx * wx + wy * wy + wz * wz)
    if (len > 1e-6) {
      const inv = 1 / len
      this.direction[0] = wx * inv
      this.direction[1] = wy * inv
      this.direction[2] = wz * inv
    }
    this._invDir[0] = 1 / (this.direction[0]! || 1e-20)
    this._invDir[1] = 1 / (this.direction[1]! || 1e-20)
    this._invDir[2] = 1 / (this.direction[2]! || 1e-20)
  }

  /**
   * Intersect a single object (and optionally its descendants).
   * Returns hits sorted by distance (nearest first).
   * Pass a pre-allocated `target` array (from `createRaycastHit()`) to avoid per-hit allocations.
   * When using a target, the return value is the number of hits written into the array.
   */
  intersectObject(object: Node, recursive?: boolean): RaycastHit[]
  intersectObject(object: Node, recursive: boolean, target: RaycastHit[]): number
  intersectObject(object: Node, recursive = false, target?: RaycastHit[]): RaycastHit[] | number {
    if (target) {
      this._hitCount = 0
      this._hitTarget = target
      this._collectHitsInto(object, recursive)
      const count = this._hitCount
      this._hitTarget = null
      // Sort only the filled portion
      if (count > 1) target.slice(0, count).sort((a, b) => a.distance - b.distance)
      return count
    }
    const hits: RaycastHit[] = []
    this._collectHits(object, recursive, hits)
    hits.sort((a, b) => a.distance - b.distance)
    return hits
  }

  /**
   * Intersect an array of objects (and optionally their descendants).
   * Returns hits sorted by distance (nearest first).
   * Pass a pre-allocated `target` array (from `createRaycastHit()`) to avoid per-hit allocations.
   * When using a target, the return value is the number of hits written into the array.
   */
  intersectObjects(objects: Node[], recursive?: boolean): RaycastHit[]
  intersectObjects(objects: Node[], recursive: boolean, target: RaycastHit[]): number
  intersectObjects(objects: Node[], recursive = false, target?: RaycastHit[]): RaycastHit[] | number {
    if (target) {
      this._hitCount = 0
      this._hitTarget = target
      for (let i = 0; i < objects.length; i++) {
        this._collectHitsInto(objects[i]!, recursive)
      }
      const count = this._hitCount
      this._hitTarget = null
      if (count > 1) target.slice(0, count).sort((a, b) => a.distance - b.distance)
      return count
    }
    const hits: RaycastHit[] = []
    for (let i = 0; i < objects.length; i++) {
      this._collectHits(objects[i]!, recursive, hits)
    }
    hits.sort((a, b) => a.distance - b.distance)
    return hits
  }

  // Zero-allocation path state
  _hitCount = 0
  _hitTarget: RaycastHit[] | null = null

  _collectHitsInto(node: Node, recursive: boolean): void {
    if (!node.visible) return
    const target = this._hitTarget!

    if (node.type === 'mesh') {
      const ox = this.origin[0]!,
        oy = this.origin[1]!,
        oz = this.origin[2]!
      const dx = this.direction[0]!,
        dy = this.direction[1]!,
        dz = this.direction[2]!
      // Grow pool if needed
      if (this._hitCount >= target.length) {
        target.push(createRaycastHit())
      }
      if (intersectMeshInto(ox, oy, oz, dx, dy, dz, node as Mesh, target[this._hitCount]!)) {
        this._hitCount++
      }
    }

    if (recursive) {
      for (let i = 0; i < node.children.length; i++) {
        this._collectHitsInto(node.children[i]!, true)
      }
    }
  }

  _collectHits(node: Node, recursive: boolean, hits: RaycastHit[]): void {
    if (!node.visible) return

    if (node.type === 'mesh') {
      const ox = this.origin[0]!,
        oy = this.origin[1]!,
        oz = this.origin[2]!
      const dx = this.direction[0]!,
        dy = this.direction[1]!,
        dz = this.direction[2]!
      if (intersectMeshInto(ox, oy, oz, dx, dy, dz, node as Mesh, _tempHit)) {
        hits.push({
          distance: _tempHit.distance,
          point: new Float32Array(_tempHit.point) as Vec3,
          normal: new Float32Array(_tempHit.normal) as Vec3,
          uv: _tempHit.uv ? (new Float32Array(_tempHit.uv) as Vec2) : null,
          triangleIndex: _tempHit.triangleIndex,
          object: _tempHit.object,
        })
      }
    }

    if (recursive) {
      for (let i = 0; i < node.children.length; i++) {
        this._collectHits(node.children[i]!, true, hits)
      }
    }
  }
}
