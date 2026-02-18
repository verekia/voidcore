# Voidcore — Raycasting & BVH

## Overview

Voidcore provides raycasting at the quality level of three-mesh-bvh using a two-level BVH (Bounding Volume Hierarchy):

1. **Scene BVH**: contains mesh-level AABBs — quickly rejects meshes the ray cannot hit
2. **Mesh BVH**: contains triangle-level data — precise intersection within a mesh

A ray first traverses the scene BVH to find candidate meshes, then traverses each candidate's mesh BVH for triangle-level intersection.

## BVH Construction

### Binned SAH with 12 Bins

Surface Area Heuristic (SAH) determines optimal split planes. The 12-bin approximation gives O(n log n) build time while producing near-optimal trees:

```typescript
const SAH_COST = {
  traversal: 1.0, // Cost of traversing an internal node
  intersection: 1.5, // Cost of testing a triangle
}
const BIN_COUNT = 12
const MAX_LEAF_TRIANGLES = 4
```

For each candidate split:

```
cost = traversalCost
     + (leftSurfaceArea / parentSurfaceArea) × leftTriCount × intersectionCost
     + (rightSurfaceArea / parentSurfaceArea) × rightTriCount × intersectionCost
```

### Build Algorithm

```typescript
const buildBVH = (positions: Float32Array, indices: Uint16Array | Uint32Array): BVH => {
  // 1. Compute centroids and AABBs for all triangles
  // 2. Recursively split:
  //    a. Compute overall AABB and centroid AABB
  //    b. For each axis (X, Y, Z):
  //       - Distribute triangles into 12 bins by centroid
  //       - Evaluate SAH cost at each of 11 bin boundaries
  //    c. Choose the split with lowest SAH cost across all 3 axes
  //    d. If no split improves over leaf cost, create a leaf node
  //    e. Partition triangles and recurse on left/right halves
  // 3. Flatten tree into contiguous 32-byte node array
}
```

## BVH Node Layout

32-byte flat array layout for cache-friendly traversal:

```
struct BVHNode {                             // 32 bytes total
  float minX, minY, minZ              // AABB min (12 bytes)
  int   rightChildOrCount             // >0: right child offset, <0: -(triCount) (4 bytes)
  float maxX, maxY, maxZ              // AABB max (12 bytes)
  int   leftChildOrStart              // Inner: left child offset, Leaf: first tri index (4 bytes)
}
```

**Leaf detection:** `rightChildOrCount < 0` → leaf node with `|rightChildOrCount|` triangles starting at index `leftChildOrStart`.

**Why flat array:** Linear memory access patterns, no pointer chasing, better cache utilization. The BVH is stored as a contiguous `ArrayBuffer` with `Float32Array` and `Int32Array` views aliased over the same memory:

```typescript
const buffer = new ArrayBuffer(nodeCount * 32)
const floatView = new Float32Array(buffer) // For AABB bounds
const intView = new Int32Array(buffer) // For child offsets / tri counts
```

## Traversal

Stack-based iterative traversal with near-first child ordering:

```typescript
const intersectBVH = (
  ray: Ray,
  nodes: Float32Array,
  intNodes: Int32Array,
  triangles: Float32Array,
  indices: Uint32Array,
): RaycastHit | null => {
  const stack = traversalStack // Pre-allocated, reused across frames
  let stackPtr = 0
  stack[stackPtr++] = 0 // Start at root node (index 0)
  let closest: RaycastHit | null = null

  while (stackPtr > 0) {
    const nodeIdx = stack[--stackPtr]
    const nodeOffset = nodeIdx * 8 // 8 floats per node (32 bytes / 4)

    // Ray-AABB test (early rejection with current closest distance)
    if (!rayIntersectsAABB(ray, floatView, nodeOffset, closest?.distance ?? Infinity)) {
      continue
    }

    const rightOrCount = intNodes[nodeOffset + 3]

    if (rightOrCount < 0) {
      // LEAF NODE: test triangles
      const triCount = -rightOrCount
      const triStart = intNodes[nodeOffset + 7]
      for (let i = triStart; i < triStart + triCount; i++) {
        const hit = rayIntersectsTriangle(ray, triangles, indices, i)
        if (hit && (!closest || hit.distance < closest.distance)) {
          closest = hit
        }
      }
    } else {
      // INNER NODE: push children (far first, near second → near processed first)
      const leftChild = intNodes[nodeOffset + 7]
      const rightChild = rightOrCount

      // Determine which child is nearer based on ray direction and split
      const leftFirst = shouldTraverseLeftFirst(ray, floatView, nodeOffset, leftChild, rightChild)

      if (leftFirst) {
        stack[stackPtr++] = rightChild // Far child pushed first
        stack[stackPtr++] = leftChild // Near child pushed second (popped first)
      } else {
        stack[stackPtr++] = leftChild
        stack[stackPtr++] = rightChild
      }
    }
  }

  return closest
}
```

**Near-first ordering** reduces average traversal depth by testing the child closer to the ray origin first. If a hit is found in the near child, the far child can often be skipped entirely (its AABB distance exceeds the hit distance).

## Ray-AABB Intersection

Slab method — fast, branchless:

```typescript
const rayIntersectsAABB = (ray: Ray, min: Vec3, max: Vec3, maxDist: number): boolean => {
  // Inverse direction pre-computed once per ray
  const t1 = (min[0] - ray.origin[0]) * ray.invDirection[0]
  const t2 = (max[0] - ray.origin[0]) * ray.invDirection[0]
  const t3 = (min[1] - ray.origin[1]) * ray.invDirection[1]
  const t4 = (max[1] - ray.origin[1]) * ray.invDirection[1]
  const t5 = (min[2] - ray.origin[2]) * ray.invDirection[2]
  const t6 = (max[2] - ray.origin[2]) * ray.invDirection[2]

  const tmin = Math.max(Math.min(t1, t2), Math.min(t3, t4), Math.min(t5, t6))
  const tmax = Math.min(Math.max(t1, t2), Math.max(t3, t4), Math.max(t5, t6))

  return tmax >= Math.max(0, tmin) && tmin < maxDist
}
```

`ray.invDirection` is pre-computed once per ray (`[1/dx, 1/dy, 1/dz]`) to avoid divisions in the hot loop.

## Ray-Triangle Intersection

Möller-Trumbore algorithm — fast, minimal branching, computes barycentric coordinates:

```typescript
const rayIntersectsTriangle = (
  ray: Ray,
  positions: Float32Array,
  indices: Uint32Array,
  triIndex: number,
): RaycastHit | null => {
  // Load triangle vertices
  const i0 = indices[triIndex * 3] * 3
  const i1 = indices[triIndex * 3 + 1] * 3
  const i2 = indices[triIndex * 3 + 2] * 3

  // Edge vectors
  const e1x = positions[i1] - positions[i0]
  const e1y = positions[i1 + 1] - positions[i0 + 1]
  const e1z = positions[i1 + 2] - positions[i0 + 2]
  const e2x = positions[i2] - positions[i0]
  const e2y = positions[i2 + 1] - positions[i0 + 1]
  const e2z = positions[i2 + 2] - positions[i0 + 2]

  // Möller-Trumbore determinant
  // ... cross products, dot products, barycentric coordinates
  // Returns: distance, barycentrics (u, v) for UV/normal interpolation
}
```

Barycentric coordinates `(u, v)` enable interpolation of UV coordinates and normals at the exact hit point.

## Two-Level BVH

### Scene BVH

Contains world-space AABBs of all meshes in the scene:

```typescript
// Built/rebuilt when scene structure changes (add/remove objects)
const sceneBVH = buildSceneBVH(scene.meshes)
```

A ray first traverses the scene BVH to find candidate meshes. This step is very cheap (~0.01ms for 2000 objects) because it only tests mesh-level AABBs.

### Mesh BVH

Contains triangle-level data for precise intersection within a single mesh:

```typescript
// Built lazily on first raycast against this mesh
const meshBVH = buildMeshBVH(geometry.positions, geometry.indices)
```

### Ray Intersection Flow

```
1. Create ray from camera + mouse position (or arbitrary origin/direction)
2. Traverse scene BVH → collect candidate meshes (those whose AABB is hit)
3. Sort candidates by ray-AABB entry distance (nearest first)
4. For each candidate:
   a. Transform ray to mesh's local space (inverse world matrix)
   b. Traverse mesh BVH → find closest triangle hit
   c. If hit, transform hit point/normal back to world space
   d. Early termination: if hit distance < next candidate's AABB distance, skip remaining
5. Return sorted hit results
```

## Lazy Construction

BVH is built on first use, then cached:

- **Mesh BVH**: built on the first raycast against that geometry, cached on the `Geometry` object. If geometry changes (`needsUpdate`), the cached BVH is invalidated.
- **Scene BVH**: rebuilt when the scene structure changes (objects added/removed). For scenes where objects frequently move, the scene BVH uses their world-space AABBs which are already updated each frame.
- **Optional worker offload**: BVH construction for large meshes (>50K triangles) can be offloaded to a Web Worker via the existing worker pool.

Build cost: ~2-5ms for 100K triangles (one-time cost per geometry).

## Frustum Culling via BVH

The scene-level BVH can also accelerate frustum culling for large scenes (>5000 objects). The frustum planes are tested against the BVH's AABB hierarchy, pruning entire subtrees of objects that are outside the frustum.

For ≤2000 objects (the engine's target), brute-force AABB testing is sufficient (~0.1ms) and the BVH is not used for culling.

## Raycaster API

```typescript
const raycaster = createRaycaster()

// From camera + screen coordinates (NDC: -1 to +1)
raycaster.setFromCamera([ndcX, ndcY], camera)

// From arbitrary ray
raycaster.set(origin, direction)

// Query: intersect all descendants of the given objects
const hits = raycaster.intersectObjects(scene.children, true) // true = recursive

// Query: intersect a single object
const hits = raycaster.intersectObject(mesh, true)
```

### Hit Result

```typescript
interface RaycastHit {
  distance: number // World-space distance from ray origin
  point: Vec3 // World-space intersection point
  normal: Vec3 // Interpolated surface normal at hit point
  uv: Vec2 | null // Interpolated UV coordinates (if geometry has UVs)
  triangleIndex: number // Index of the triangle that was hit
  object: Mesh // The mesh that was hit
}
```

Hits are returned sorted by distance (nearest first). The `point`, `normal`, and `uv` are interpolated using the barycentric coordinates from Möller-Trumbore.

### NDC from Mouse Event

Helper for converting mouse/touch events to NDC:

```typescript
// In user code:
canvas.addEventListener('click', event => {
  const ndcX = (event.clientX / canvas.clientWidth) * 2 - 1
  const ndcY = -(event.clientY / canvas.clientHeight) * 2 + 1 // Y flipped
  raycaster.setFromCamera([ndcX, ndcY], camera)
  const hits = raycaster.intersectObjects(scene.children, true)
  if (hits.length > 0) {
    console.log('Hit:', hits[0].object.name, 'at', hits[0].point)
  }
})
```

## Performance

```
Scene BVH traversal:      ~0.01ms per ray (2000 objects)
Mesh BVH traversal:       ~0.01-0.05ms per ray (depending on triangle count)
Total per ray:            ~0.02-0.06ms

BVH construction:         ~2-5ms for 100K triangles (one-time)
Scene BVH rebuild:        ~0.1ms for 2000 objects (on structure change)
```

Multiple rays per frame (e.g., hover detection + click) are feasible within the frame budget.
