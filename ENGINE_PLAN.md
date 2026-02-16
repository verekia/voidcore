# LiteGPU — Lightweight High-Performance WebGPU Engine with R3F-like DX

A new engine designed around the 95% of performance wins (instancing, sorting, culling, zero GC) without the complexity of WASM/C. TypeScript only, easy to understand, modify, and maintain.

## Design Principles

1. **Automatic performance** — instancing, sorting, culling happen without user opt-in
2. **TypeScript only** — no C, no WASM, no build toolchain beyond bundling
3. **Thin React layer** — no custom reconciler, no fake Three.js objects
4. **Flat by default** — no deep scene graph; parent-child transforms are opt-in
5. **WebGPU first** — WebGL2 as fallback, not the primary target
6. **Zero per-frame allocations** — pre-allocated typed arrays and scratch math

---

## Architecture

```
┌─ React Layer ──────────────────────────────────────────────────────┐
│                                                                    │
│  <Canvas>              Thin React wrapper. Components call         │
│    <Mesh />            world.spawn() / world.despawn() via         │
│    <Camera />          useEffect. Props sync via refs.             │
│    <Light />           No custom reconciler.                       │
│  </Canvas>                                                         │
│                                                                    │
└───────────────────────────┬────────────────────────────────────────┘
                            │
┌─ World ───────────────────┴────────────────────────────────────────┐
│                                                                    │
│  Entity storage      Flat typed arrays (SoA layout in JS)          │
│  Material registry   Map<materialId, MaterialDesc>                 │
│  Geometry registry   Map<geometryId, GPUBufferHandles>             │
│  Render list         Sorted draw commands, auto-instanced          │
│                                                                    │
└───────────────────────────┬────────────────────────────────────────┘
                            │
┌─ Render Loop ─────────────┴────────────────────────────────────────┐
│                                                                    │
│  1. Recompute dirty matrices (only flagged entities)               │
│  2. Frustum cull → visible list                                    │
│  3. Sort visible by pipeline → material → geometry                 │
│  4. Merge same-geometry+material runs into instanced draws         │
│  5. Upload instance buffers + submit GPU draw calls                │
│  6. Post-processing (bloom, etc.)                                  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Performance Wins (ranked by impact)

### 1. Automatic Instancing

The single biggest win. If N entities share the same geometry + material, they become 1 draw call with N instances.

**How it works:**

- Each entity stores a `geometryId` and `materialId`
- After sorting, consecutive entities with the same `(pipelineId, materialId, geometryId)` tuple form a run
- Each run becomes one instanced draw call
- Per-instance data (world matrix, color, flags) is written into a dynamic instance buffer

**Expected impact:** Scenes go from thousands of draw calls to tens. A forest of 1,000 trees = 1 draw call instead of 1,000.

**Instance buffer layout** (per instance, written each frame):

```
worldMatrix:  mat4x4<f32>   (64 bytes)
color:        vec4<f32>     (16 bytes)
flags:        u32           (4 bytes)
padding:      12 bytes      (align to 96)
                            ─────────────
                            96 bytes/instance
```

### 2. Draw Call Sorting

Sort the visible entity list by `(pipelineId, materialId, geometryId)` before issuing draws. This minimizes GPU state changes:

- **Pipeline switches** (most expensive) — static vs skinned vs textured
- **Material switches** (moderate) — different uniforms/textures
- **Geometry switches** (cheap) — different vertex/index buffers

Even for non-instanced draws, sorting alone can yield 20-40% frame time improvement by reducing driver overhead.

**Sort key encoding:** Pack `(pipelineId, materialId, geometryId)` into a single `uint32` or `uint64` for a fast numeric sort. For example:

```
bits [31..28]  pipeline  (16 pipelines max)
bits [27..16]  material  (4096 materials max)
bits [15..0]   geometry  (65536 geometries max)
```

Then `Array.prototype.sort` on the sort keys. No radix sort needed for <10k draws.

### 3. Frustum Culling

Test every entity's bounding sphere against the 6 frustum planes. Skip entities that are fully outside.

- Extract frustum planes from the view-projection matrix (standard Gribb/Hartmann method)
- Each entity stores a bounding sphere (center + radius in world space)
- Sphere-plane dot product test: 6 multiplies + 6 adds per entity
- Iterate a tight typed array — no object access, cache-friendly

**Expected impact:** Culls 30-70% of entities in typical scenes. Saves both CPU (fewer draw commands to build) and GPU (fewer vertices to process).

### 4. WebGPU First

WebGPU has dramatically lower CPU overhead per draw call compared to WebGL:

- No synchronous driver validation per call
- Command buffers are recorded and submitted in batch
- Bind groups reduce state setting
- `drawIndexedIndirect` enables true GPU-driven rendering later

**Fallback:** WebGL2 for Safari and older browsers. Same API shape, different backend implementation (UBOs instead of bind groups, separate draw calls instead of indirect).

### 5. Flat Typed Array Storage (SoA in JS)

All per-entity data lives in flat `Float32Array` / `Uint32Array` buffers:

```ts
// Pre-allocated for MAX_ENTITIES (e.g. 50,000)
positions:      Float32Array(MAX * 3)
rotations:      Float32Array(MAX * 3)     // Euler XYZ
scales:         Float32Array(MAX * 3)
worldMatrices:  Float32Array(MAX * 16)
colors:         Float32Array(MAX * 4)     // RGBA
flags:          Uint32Array(MAX)
boundingSpheres: Float32Array(MAX * 4)    // cx, cy, cz, radius
geometryIds:    Uint32Array(MAX)
materialIds:    Uint32Array(MAX)
sortKeys:       Uint32Array(MAX)
```

**Why this matters:**

- **Zero GC** — no per-entity objects to collect
- **Cache-friendly** — iterating `flags` to find dirty entities reads contiguous memory
- **GPU-uploadable** — `worldMatrices` can be copied directly into a GPU buffer
- **Debuggable** — it's just TypeScript, inspect in devtools

### 6. Dirty Flags + Incremental Matrix Updates

Only recompute TRS → world matrix for entities that changed:

```ts
for (let i = 0; i < entityCount; i++) {
  if (flags[i] & DIRTY) {
    computeWorldMatrix(i)     // reads positions/rotations/scales, writes worldMatrices
    flags[i] &= ~DIRTY        // clear dirty bit
  }
}
```

Three.js calls `updateMatrixWorld()` on the entire scene tree every frame. With dirty flags, a static scene of 10,000 entities where 5 move = 5 matrix computations, not 10,000.

### 7. Zero Per-Frame Allocations

All math uses pre-allocated scratch variables:

```ts
// Module-level scratch — never allocate in the render loop
const _mat4A = new Float32Array(16)
const _mat4B = new Float32Array(16)
const _vec3A = new Float32Array(3)
const _vec3B = new Float32Array(3)
const _frustumPlanes = new Float32Array(24) // 6 planes × 4 floats
```

No `new Vector3()`, no `new Matrix4()`, no temp arrays. At 144fps you get 6.9ms per frame — GC pauses of even 1-2ms are visible as hitches.

---

## Entity System

### Spawn / Despawn

```ts
// Internally: finds a free slot, writes defaults into typed arrays
const entity = world.spawn({
  geometry: boxGeo,
  material: redMat,
  position: [0, 1, 0],
})

// Internally: marks slot as free, adds to freelist
world.despawn(entity)
```

Entities are integer IDs (indices into the typed arrays). A freelist tracks available slots to avoid fragmentation.

### Entity Handle

```ts
interface Entity {
  id: number                    // index into SoA arrays
  setPosition(x, y, z): void   // writes to positions[], sets dirty flag
  setRotation(x, y, z): void
  setScale(x, y, z): void
  setColor(r, g, b, a): void
  setFlag(flag: number): void
  destroy(): void               // calls world.despawn
}
```

Each setter writes directly into the typed array at the correct offset and sets the dirty flag. No intermediate objects.

---

## Geometry & Materials

### Geometry

```ts
interface Geometry {
  id: number
  vertexBuffer: GPUBuffer       // interleaved [pos, normal, color, ...]
  indexBuffer: GPUBuffer
  indexCount: number
  indexFormat: 'uint16' | 'uint32'
  boundingRadius: number        // for frustum culling
}
```

Built-in primitive generators: `createBox()`, `createSphere()`. Plus a generic `createGeometry(vertices, indices)` for custom/imported meshes.

### Materials

Start simple. Three material types matching three pipelines:

```ts
interface StaticMaterial {
  type: 'static'
  id: number
}

interface SkinnedMaterial {
  type: 'skinned'
  id: number
  // Joint matrices provided per-entity at draw time
}

interface TexturedMaterial {
  type: 'textured'
  id: number
  texture: GPUTexture
  sampler: GPUSampler
}
```

Material = pipeline selection + any extra bindings (textures, etc.). Color is per-entity, not per-material, so 500 red boxes and 500 blue boxes with the same geometry still instance into 1 draw call with color in the instance buffer.

---

## Lighting

Keep it simple. Lambert diffuse (directional + ambient) is enough for most stylized games:

```ts
interface LightState {
  direction: [number, number, number]
  color: [number, number, number]
  ambientColor: [number, number, number]
}
```

Uploaded as a uniform buffer, shared across all draws. Per-entity unlit flag for UI elements / skyboxes / emissive objects.

---

## React Layer

### `<Canvas>`

```tsx
function Canvas({ children, ...props }: CanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const worldRef = useRef<World>(null)

  useEffect(() => {
    const world = new World(canvasRef.current!)
    worldRef.current = world
    // Start render loop
    const loop = () => { world.render(); requestAnimationFrame(loop) }
    requestAnimationFrame(loop)
    return () => world.destroy()
  }, [])

  return (
    <WorldContext.Provider value={worldRef}>
      <canvas ref={canvasRef} {...props} />
      {children}
    </WorldContext.Provider>
  )
}
```

### `<Mesh>`

```tsx
function Mesh({ geometry, material, position, rotation, scale, color }: MeshProps) {
  const world = useWorld()
  const entityRef = useRef<Entity>(null)

  useEffect(() => {
    entityRef.current = world.spawn({ geometry, material })
    return () => entityRef.current?.destroy()
  }, [geometry, material])

  // Sync transforms without re-spawning
  useEffect(() => {
    if (position) entityRef.current?.setPosition(...position)
  }, [position?.[0], position?.[1], position?.[2]])

  useEffect(() => {
    if (rotation) entityRef.current?.setRotation(...rotation)
  }, [rotation?.[0], rotation?.[1], rotation?.[2]])

  useEffect(() => {
    if (scale) entityRef.current?.setScale(...scale)
  }, [scale?.[0], scale?.[1], scale?.[2]])

  useEffect(() => {
    if (color) entityRef.current?.setColor(...color)
  }, [color?.[0], color?.[1], color?.[2], color?.[3]])

  return null // No DOM output
}
```

### `<Camera>`

```tsx
function Camera({ eye, target, up, fov, near, far }: CameraProps) {
  const world = useWorld()

  useEffect(() => {
    world.setCamera({ eye, target, up, fov, near, far })
  }, [eye, target, up, fov, near, far])

  return null
}
```

### Why no custom reconciler?

A custom React reconciler (like R3F's) adds thousands of lines of code to map React's internal fiber tree to engine objects. It's powerful but complex and fragile across React versions.

`useEffect` + refs achieves the same result for our use case:

- Mount → spawn entity
- Prop change → update entity
- Unmount → despawn entity

It's 50 lines per component instead of a 2000-line reconciler. The tradeoff is you lose automatic deep prop spreading (`<mesh position-x={5} />`), but explicit props are clearer anyway.

---

## Render Loop (World.render())

```ts
render() {
  // 1. Recompute dirty matrices
  for (let i = 0; i < this.entityCount; i++) {
    if (this.flags[i] & DIRTY) {
      mat4_from_trs(this.worldMatrices, i, this.positions, this.rotations, this.scales)
      this.flags[i] &= ~DIRTY
    }
  }

  // 2. Update camera matrices
  this.camera.update()  // writes view, projection, VP into scratch arrays

  // 3. Frustum cull
  extractFrustumPlanes(this.camera.vp, _frustumPlanes)
  let visibleCount = 0
  for (let i = 0; i < this.entityCount; i++) {
    if (!(this.flags[i] & VISIBLE)) continue
    if (sphereInFrustum(this.boundingSpheres, i, _frustumPlanes)) {
      this.visibleList[visibleCount++] = i
    }
  }

  // 4. Build sort keys for visible entities
  for (let j = 0; j < visibleCount; j++) {
    const i = this.visibleList[j]
    this.sortKeys[j] = packSortKey(this.pipelineIds[i], this.materialIds[i], this.geometryIds[i])
  }

  // 5. Sort visible list by sort key
  sortByKey(this.visibleList, this.sortKeys, visibleCount)

  // 6. Build instanced draw commands (merge consecutive same-key runs)
  const draws = buildDrawCommands(this.visibleList, this.sortKeys, visibleCount)

  // 7. Upload instance data + submit GPU draws
  this.renderer.draw(draws, this)
}
```

---

## GPU Renderer Interface

```ts
interface Renderer {
  init(canvas: HTMLCanvasElement): Promise<void>
  draw(draws: DrawCommand[], world: World): void
  destroy(): void
}

interface DrawCommand {
  pipelineId: number
  materialId: number
  geometryId: number
  instanceCount: number
  // Offset into the visible list where this run starts
  startIndex: number
}
```

Two implementations: `WebGPURenderer` and `WebGLRenderer`, selected at init time based on `navigator.gpu` availability.

### WebGPU Renderer

- **Bind group 0**: Camera (view + projection matrices)
- **Bind group 1**: Light (direction, color, ambient)
- **Bind group 2**: Material-specific (textures, samplers)
- **Instance buffer**: Written each frame with per-instance data for all visible entities
- **Draw call**: `renderPass.drawIndexed(indexCount, instanceCount, 0, 0, firstInstance)`

### WebGL2 Renderer

- **UBO 0**: Camera
- **UBO 1**: Light
- Instancing via `ANGLE_instanced_arrays` / WebGL2 native
- `gl.drawElementsInstanced()` for each draw command

---

## Coordinate System & Math

- **Z-up, right-handed** (Blender convention, same as VoidCore)
- **Column-major mat4**
- **Euler ZXY rotation order**
- All math operates directly on `Float32Array` sections — no wrapper classes

Core math functions (all work on typed arrays by index):

```ts
mat4_from_trs(out: Float32Array, index: number, pos, rot, scale)
mat4_multiply(out: Float32Array, a: Float32Array, b: Float32Array)
mat4_perspective(out: Float32Array, fov, aspect, near, far)
mat4_lookAt(out: Float32Array, eye, target, up)
vec3_transformMat4(out: Float32Array, v: Float32Array, m: Float32Array)
extractFrustumPlanes(vp: Float32Array, out: Float32Array)
sphereInFrustum(spheres: Float32Array, index: number, planes: Float32Array): boolean
```

---

## What's Explicitly Out of Scope (for v1)

- **Scene graph hierarchy** — flat entity list only. Parent-child can be added later as an opt-in layer
- **Physics** — use Rapier or cannon-es alongside
- **Shadows** — significant complexity, add in v2
- **glTF loader** — import meshes as raw vertex/index arrays. A glTF adapter can be a separate package
- **Skeletal animation** — add in v2 once the core is solid
- **Post-processing** — except bloom (single MRT pass), keep it minimal
- **Text rendering** — use HTML/CSS overlays or SDF text as a separate package

---

## File Structure

```
src/
  engine/
    world.ts            World class: entity storage, render loop, spawn/despawn
    renderer.ts         Renderer interface + WebGPU implementation
    webgl-renderer.ts   WebGL2 fallback
    geometry.ts         Primitive generators (box, sphere, custom)
    material.ts         Material types and registry
    camera.ts           Camera state + matrix computation
    math.ts             mat4/vec3 operations on typed arrays
    culling.ts          Frustum plane extraction + sphere test
    sorting.ts          Sort key packing + draw command merging
    shaders.ts          WGSL shaders (static, textured pipelines)
    webgl-shaders.ts    GLSL 300 es shaders
    types.ts            Shared types and constants
    index.ts            Barrel exports

  react/
    context.ts          WorldContext + useWorld hook
    canvas.tsx          <Canvas> component
    mesh.tsx            <Mesh> component
    camera.tsx          <Camera> component
    light.tsx           <Light> component
    index.ts            Barrel exports

  index.ts              Package entry point
```

---

## Summary

The core thesis: **data layout and batching strategy matter more than language choice**. Flat typed arrays in TypeScript, automatic instancing, draw call sorting, and frustum culling get you 90%+ of the performance of a WASM engine — with full debuggability, one language, and a fraction of the complexity.
