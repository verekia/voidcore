# Voidcore — Geometry

## Vertex Layout

Separate buffers per attribute (not interleaved). Each attribute stored in its own GPU buffer:

| Attribute        | Type                | Size per vertex | Required     |
| ---------------- | ------------------- | --------------- | ------------ |
| `position`       | Float32×3           | 12 bytes        | Yes          |
| `normal`         | Float32×3           | 12 bytes        | Yes          |
| `uv`             | Float32×2           | 8 bytes         | Optional     |
| `color`          | Float32×4           | 16 bytes        | Optional     |
| `_materialindex` | Uint8×1             | 1 byte          | Optional     |
| `joints`         | Uint8×4 or Uint16×4 | 4-8 bytes       | Skinned only |
| `weights`        | Float32×4           | 16 bytes        | Skinned only |

**Why separate buffers:** Avoids wasted bandwidth when optional attributes are absent. The GPU only fetches the attributes actually used by the active shader variant. A mesh with only positions and normals (24 bytes/vertex) doesn't pay for UV, color, or material index data.

## Index Buffer

Automatic format selection based on vertex count:

```typescript
const indexFormat = vertexCount > 65535 ? 'uint32' : 'uint16'
```

Uint16 saves memory and bandwidth for small meshes (<65K vertices). Uint32 for larger meshes. The engine handles this transparently.

## Bounding Volumes

### AABB (Primary)

Computed at geometry creation from vertex positions:

```typescript
// Scan all positions to find min/max per axis
const aabb = aabbFromPositions(positions) // [minX, minY, minZ, maxX, maxY, maxZ]
```

Used for:

- Frustum culling (6 plane tests per object)
- BVH construction (scene-level and mesh-level)
- Coarse raycast rejection

World-space AABB recomputed when the node's world matrix changes (transforms the local AABB corners and re-envelopes).

### Bounding Sphere (Optional)

Available for distance-based calculations (e.g., LOD, overlay distance scaling):

```typescript
const center = aabbCenter(aabb)
const radius = maxDistanceFromCenter(positions, center)
```

## Parametric Primitives

Seven parametric primitives, all generated in Z-up orientation:

### Plane

```typescript
const plane = createPlaneGeometry({
  width: 10, // X extent (default 1)
  height: 10, // Y extent (default 1)
  widthSegments: 1, // Subdivisions along X (default 1)
  heightSegments: 1, // Subdivisions along Y (default 1)
})
```

Lies on XY plane with normal pointing +Z. Center at origin.

### Box

```typescript
const box = createBoxGeometry({
  width: 1, // X extent (default 1)
  height: 1, // Y extent (default 1)
  depth: 1, // Z extent (default 1)
})
```

Axis-aligned, centered at origin. Each face has outward normals. UVs mapped per face.

### Sphere

```typescript
const sphere = createSphereGeometry({
  radius: 1, // Default 1
  widthSegments: 32, // Horizontal segments (default 32)
  heightSegments: 16, // Vertical segments (default 16)
})
```

UV sphere. Poles along Z axis (Z-up). Smooth normals.

### Cone

```typescript
const cone = createConeGeometry({
  radius: 1, // Base radius (default 1)
  height: 2, // Height along Z (default 1)
  radialSegments: 32, // Default 32
})
```

Axis along Z. Base centered at origin, tip at +Z.

### Cylinder

```typescript
const cylinder = createCylinderGeometry({
  radiusTop: 1, // Top radius (default 1)
  radiusBottom: 1, // Bottom radius (default 1)
  height: 2, // Height along Z (default 1)
  radialSegments: 32, // Default 32
})
```

Axis along Z. Centered at origin (extends ±height/2 along Z).

### Capsule

```typescript
const capsule = createCapsuleGeometry({
  radius: 0.5, // Hemisphere radius (default 0.5)
  height: 2, // Total height including hemispheres (default 1)
  radialSegments: 32, // Default 32
  heightSegments: 1, // Cylinder segments (default 1)
})
```

Axis along Z. Two hemispheres + cylinder. Centered at origin.

### Circle

```typescript
const circle = createCircleGeometry({
  radius: 1, // Default 1
  segments: 32, // Default 32
})
```

Disc on XY plane, normal pointing +Z. Center at origin.

### Shared Properties

All primitives:

- Center at origin
- Use Z-up convention (axis-aligned shapes use Z as the "up" axis)
- Generate smooth normals
- Include UV coordinates
- Include index buffers
- Return a `Geometry` object ready to attach to a mesh

## Custom Geometry

```typescript
const custom = createGeometry({
  positions: new Float32Array([...]),          // Required: xyz per vertex
  normals: new Float32Array([...]),            // Required: xyz per vertex
  indices: new Uint16Array([...]),             // Required: triangle indices
  uvs: new Float32Array([...]),               // Optional: uv per vertex
  colors: new Float32Array([...]),            // Optional: rgba per vertex
  materialIndices: new Uint8Array([...]),     // Optional: index per vertex
  joints: new Uint8Array([...]),              // Optional: 4 joint indices per vertex
  weights: new Float32Array([...]),           // Optional: 4 weights per vertex
})
```

The factory validates inputs:

- `positions.length` must be divisible by 3
- `normals.length` must equal `positions.length`
- `indices` values must be within `[0, vertexCount)`
- `uvs.length` must be `(vertexCount * 2)` if provided
- `colors.length` must be `(vertexCount * 4)` if provided
- `materialIndices.length` must be `vertexCount` if provided

## GPU Buffer Management

### Lazy Upload

Geometry vertex/index data is kept in CPU-side typed arrays until the first frame that renders the geometry. At that point, GPU buffers are created and uploaded:

```typescript
// Internal: called by the renderer on first draw
const ensureGPUBuffers = (geometry: Geometry, device: Device): void => {
  if (geometry._gpuBuffers) return // Already uploaded

  geometry._gpuBuffers = {
    position: device.createBuffer({ data: geometry.positions, usage: BufferUsage.VERTEX }),
    normal: device.createBuffer({ data: geometry.normals, usage: BufferUsage.VERTEX }),
    index: device.createBuffer({ data: geometry.indices, usage: BufferUsage.INDEX }),
    // ... optional attributes only if present
  }
}
```

This avoids uploading geometry that may never be rendered (e.g., hidden objects, preloaded assets).

### Update Flow

After creation, CPU data is considered immutable. If the user modifies the underlying typed arrays, they must flag the geometry for re-upload:

```typescript
geometry.positions[0] = 5.0 // Modify CPU data
geometry.needsUpdate = true // Flag for re-upload next frame
```

On the next render, the engine detects `needsUpdate`, re-uploads the affected buffers, and clears the flag.

## Geometry Sharing

Multiple meshes can reference the same geometry:

```typescript
const sharedBox = createBoxGeometry({ width: 1, height: 1, depth: 1 })
const mesh1 = createMesh(sharedBox, material1)
const mesh2 = createMesh(sharedBox, material2) // Same geometry, different material
```

GPU buffers are allocated once, drawn multiple times. Geometry tracks a reference count internally and is only disposed when all referencing meshes are gone (or explicitly disposed by the user).

## BVH Integration

The BVH for a geometry is **not** built at creation time. It is built lazily on the first raycast that hits the geometry, then cached on the Geometry object:

```typescript
// Internal: called by raycaster
const ensureBVH = (geometry: Geometry): BVH => {
  if (!geometry._bvh) {
    geometry._bvh = buildBVH(geometry.positions, geometry.indices)
  }
  return geometry._bvh
}
```

This avoids wasting build time on geometry that is never raycasted. BVH construction for a 100K triangle mesh takes ~2-5ms (one-time cost).

## Disposal

```typescript
geometry.dispose() // Release GPU buffers and optionally free CPU data
```

- Releases all GPU buffers (vertex, index)
- Releases cached BVH
- In React mode, disposal is automatic on component unmount
- Calling `dispose` on a geometry still referenced by a mesh is safe — the geometry marks itself as disposed, and the mesh will skip rendering
