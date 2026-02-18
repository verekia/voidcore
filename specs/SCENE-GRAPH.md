# Voidcore — Scene Graph

## Storage Strategy

Traditional object graph with contiguous matrix storage. Users interact with nodes that have `.parent`, `.children`, and named properties. Internally, world matrices are packed into a contiguous `Float32Array` for cache-friendly GPU upload.

## Node Types

All node types share a common base with: position, rotation, scale, visible, frustumCulled, parent, children, name.

### Base Node

Every node in the scene graph has:

```typescript
interface Node {
  // Identity
  name: string

  // Transform (setting any marks _dirtyLocal = true)
  position: Vec3 // Float32Array view, default [0,0,0]
  rotation: Quat // Float32Array, default [0,0,0,1] (identity)
  scale: Vec3 // Float32Array view, default [1,1,1]

  // Hierarchy
  parent: Node | null // Set automatically by add/remove
  children: Node[]

  // Flags
  visible: boolean // false = skip rendering and children traversal
  frustumCulled: boolean // true by default, test AABB against frustum

  // Shadow control (Mesh only)
  castShadow: boolean
  receiveShadow: boolean

  // Internal (not public API)
  _localMatrix: Mat4
  _worldMatrix: Mat4 // Also aliased into contiguous pool
  _poolIndex: number // Index into the contiguous matrix pool
  _dirtyLocal: boolean
  _dirtyWorld: boolean
}
```

### Node Type Hierarchy

- **Group** — transform-only container, no renderable data. Used for organizing the scene.
- **Mesh** — geometry + material reference. Rendered during opaque or transparent pass.
- **SkinnedMesh** — extends Mesh with a skeleton reference. Rendered with GPU skinning.
- **PerspectiveCamera** — perspective projection. Transform determines view position/orientation.
- **DirectionalLight** — direction derived from world rotation. Can be parented, transformed, animated.
- **AmbientLight** — has no spatial properties, stored as a property on the Scene directly.

## Transform Representation

Position (Vec3) + Rotation (Quat) + Scale (Vec3):

```typescript
mesh.position.set(1, 0, 3) // Vec3 (Float32Array view)
mesh.rotation = quatFromAxisAngle(out, [0, 0, 1], Math.PI / 4) // Quaternion [x,y,z,w]
mesh.scale.set(2, 2, 2) // Vec3 (Float32Array view)
```

Setting any transform property marks `_dirtyLocal = true`. Quaternions are the only rotation representation — no Euler angle conversion. This avoids gimbal lock and enables slerp blending.

The local matrix is computed from TRS via `mat4Compose`:

```
LocalMatrix = T × R × S
```

## Dirty Flag System

Two-level dirty flags with early-exit propagation:

- `_dirtyLocal` — local matrix needs recomputation (position, rotation, or scale changed)
- `_dirtyWorld` — world matrix needs recomputation (self or any ancestor changed)

```typescript
// Phase 1: Propagate dirty flags down the tree
const propagateDirty = (node: Node): void => {
  if (node._dirtyLocal) node._dirtyWorld = true
  if (node._dirtyWorld) {
    for (const child of node.children) {
      if (!child._dirtyWorld) {
        // Early exit: if already dirty, subtree already processed
        child._dirtyWorld = true
        propagateDirty(child)
      }
    }
  }
}

// Phase 2: Recompute only dirty matrices (depth-first, parent always before children)
const updateWorldMatrices = (node: Node): void => {
  if (node._dirtyLocal) {
    mat4Compose(node._localMatrix, node.position, node.rotation, node.scale)
    node._dirtyLocal = false
  }
  if (node._dirtyWorld) {
    if (node.parent) {
      mat4Multiply(node._worldMatrix, node.parent._worldMatrix, node._localMatrix)
    } else {
      mat4Copy(node._worldMatrix, node._localMatrix)
    }
    node._dirtyWorld = false
  }
  for (const child of node.children) updateWorldMatrices(child)
}
```

**Performance:** Typical frame has 2-5% of objects moving → 95-98% of matrix recomputations skipped. Cost: ~0.3ms for 1000 dirty nodes.

## Contiguous Matrix Storage

World matrices are stored in a contiguous `Float32Array(activeNodeCount * 16)` indexed by each node's `_poolIndex`. The renderer reads this array directly for GPU upload — no per-node object access during rendering.

**Pool management:**

- When a node is added to the scene, it receives a `_poolIndex` from a free list
- When removed, the slot is returned to the free list
- The pool grows geometrically if exhausted (double capacity, copy existing data)
- The Float32Array is uploaded directly to the per-object uniform buffer

## Frustum Culling

AABB-based frustum culling with hierarchical support:

- Each mesh has a world-space AABB computed from geometry bounds + world matrix
- Frustum planes extracted from the camera's view-projection matrix via Gribb-Hartmann method
- P-vertex/N-vertex AABB-plane test: 6 plane tests per object
- For 2000 objects, brute-force culling: ~0.1-0.2ms
- **Hierarchical culling:** if a Group's AABB is fully outside the frustum, skip all children
- BVH-accelerated culling available for scenes exceeding 5000 objects (using the scene-level BVH from the raycasting system)

```typescript
// P-vertex optimization: for each frustum plane, test the AABB vertex
// most in the direction of the plane normal (P-vertex).
// If the P-vertex is outside, the entire AABB is outside.
const frustumContainsAABB = (planes: Float32Array, aabb: AABB): boolean => {
  for (let i = 0; i < 6; i++) {
    const px = planes[i * 4] >= 0 ? aabb[3] : aabb[0] // maxX or minX
    const py = planes[i * 4 + 1] >= 0 ? aabb[4] : aabb[1]
    const pz = planes[i * 4 + 2] >= 0 ? aabb[5] : aabb[2]
    if (planes[i * 4] * px + planes[i * 4 + 1] * py + planes[i * 4 + 2] * pz + planes[i * 4 + 3] < 0) {
      return false
    }
  }
  return true
}
```

## Bones as Scene Nodes

Bones are regular scene nodes in the hierarchy. Bone attachment is standard parenting:

```typescript
const handBone = skeleton.getBone('hand_R')
handBone.add(swordMesh) // Sword inherits bone's animated world transform
```

No special attachment API needed. The standard scene graph hierarchy handles it. When the skeleton updates bone transforms during animation, any children (attached meshes) automatically inherit the updated world transform through the normal dirty flag propagation.

## Scene Graph Operations

```typescript
// Parent/child management
scene.add(mesh)
group.add(child1, child2) // Variadic
group.remove(child1)

// Traversal
scene.traverse(node => {
  // Depth-first, visits every node including the root
})

// Lookup
scene.getByName('player') // Recursive find by name

// Transform helpers
node.lookAt([10, 0, 0]) // Z-up aware lookAt (rotates node to face target)
```

### `scene.getByName` Performance

For scenes that frequently look up nodes by name, an internal `Map<string, Node>` is maintained. Updated on `add`/`remove`. Direct Map lookup is O(1) vs O(n) recursive search.

## Per-Node Properties Summary

```typescript
// Shared by all node types
node.visible = true // Skip rendering (and children) if false
node.frustumCulled = true // Test AABB against frustum (default true)
node.name = 'player' // For getByName lookup

// Mesh-specific
mesh.castShadow = true // Include in shadow pass
mesh.receiveShadow = true // Sample shadow map in fragment shader
mesh.geometry // Geometry reference
mesh.material // Material reference

// SkinnedMesh-specific
skinnedMesh.skeleton // Skeleton reference

// Camera-specific
camera.fov = 60 // Field of view in degrees
camera.near = 0.1
camera.far = 1000
camera.aspect // Set automatically from canvas

// DirectionalLight-specific
light.color = [1, 1, 0.95]
light.intensity = 1.0
light.castShadow = true
```

## Scene Object

The Scene is the root of the graph. It also holds global state:

```typescript
const scene = createScene()

// Ambient light is a scene property, not a node
scene.ambientLight = { color: [0.4, 0.45, 0.5], intensity: 0.3 }

// Scene graph operations
scene.add(mesh)
scene.traverse(callback)
scene.getByName('player')
```

Multiple scenes can exist simultaneously (e.g., for different rendering contexts), but only one is rendered per `engine.render()` call.
