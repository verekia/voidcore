# Voidcore — Architecture

## Layered Architecture

Strict one-way dependency flow. Lower layers never import from upper layers.

```
React Bindings (voidcore/react)      ← leaf, nothing depends on it
    │
Controls / HTML Overlay              ← leaf modules
    │
Animation / Loaders                  ← feature modules
    │
Renderer / Post-FX                   ← rendering pipeline
    │
Scene / Materials                    ← scene representation
    │
Spatial / Lighting                   ← spatial queries, light data
    │
Device                               ← GPU abstraction layer
    │
Math                                 ← zero dependencies
```

Math depends on nothing. Device depends only on Math. Scene depends on Device + Math. The renderer depends on Scene + Materials + Spatial + Lighting + Device + Math. React is a leaf — nothing in the engine core depends on it.

## Module Organization

Single `voidcore` package with internal module boundaries:

```
voidcore/
├── src/
│   ├── math/          # Vec3, Mat4, Quat, AABB, Frustum, Ray
│   ├── device/        # GPU abstraction (Device, Buffer, Texture, Pipeline, BindGroup)
│   │   ├── types.ts          # Shared interfaces
│   │   ├── webgpu.ts         # WebGPU implementation
│   │   └── webgl2.ts         # WebGL2 implementation
│   ├── renderer/      # Render loop, pass orchestration, draw call sorting
│   │   ├── renderer.ts       # Main render function
│   │   ├── sort.ts           # 32-bit radix sort
│   │   ├── passes/           # Shadow, opaque, transparent, composite, bloom, blit
│   │   └── uniforms.ts       # Shared uniform buffer management
│   ├── scene/         # Scene graph, nodes, transforms, dirty flags
│   │   ├── node.ts           # Base node (position, rotation, scale, dirty flags)
│   │   ├── mesh.ts           # Mesh = geometry + material
│   │   ├── skinned-mesh.ts   # SkinnedMesh with skeleton reference
│   │   ├── group.ts          # Transform-only container
│   │   ├── camera.ts         # PerspectiveCamera
│   │   ├── scene.ts          # Scene root (ambient light, traversal, lookup)
│   │   └── matrix-pool.ts    # Contiguous Float32Array for world matrices
│   ├── materials/     # BasicMaterial, LambertMaterial, shader management
│   │   ├── basic.ts
│   │   ├── lambert.ts
│   │   ├── shaders/          # WGSL + GLSL source files
│   │   └── variants.ts       # Feature-flag shader compilation and caching
│   ├── geometry/      # Parametric primitives, vertex layout, buffer management
│   │   ├── primitives/       # plane, box, sphere, cone, cylinder, capsule, circle
│   │   ├── geometry.ts       # createGeometry, attribute management, disposal
│   │   └── layout.ts         # Vertex attribute definitions
│   ├── animation/     # Skeleton, clips, mixer, crossfade
│   │   ├── skeleton.ts
│   │   ├── clip.ts
│   │   ├── mixer.ts
│   │   └── action.ts
│   ├── loaders/       # glTF, Draco, KTX2/Basis
│   │   └── gltf.ts           # Main glTF loader
│   ├── lighting/      # Directional light, ambient light, shadows
│   │   ├── directional.ts
│   │   └── shadow.ts          # Shadow matrix computation
│   ├── spatial/       # BVH, raycasting, frustum culling
│   │   ├── bvh.ts            # SAH construction, flat node layout
│   │   ├── raycaster.ts      # Two-level BVH traversal
│   │   └── frustum.ts        # AABB frustum culling
│   ├── postfx/        # Bloom, OIT composite
│   │   ├── bloom.ts          # Downsample/upsample chain
│   │   └── oit-composite.ts  # WBOIT composite pass
│   ├── controls/
│   │   └── orbit.ts          # OrbitControls
│   ├── overlay/
│   │   └── overlay.ts        # HTML overlay manager
│   ├── react/         # React bindings (voidcore/react)
│   │   ├── index.ts          # Public exports
│   │   ├── reconciler.ts     # Custom react-reconciler host config
│   │   ├── Canvas.tsx        # Root component
│   │   ├── hooks.ts          # useFrame, useEngine, useLoader, useGLTF, useAnimations
│   │   ├── Html.tsx          # HTML overlay React component
│   │   └── components.ts     # JSX intrinsic element type definitions
│   └── index.ts       # Public exports for voidcore
└── package.json
```

## Data Storage: Object Graph with SoA Internals

The public API is object-oriented — users create nodes, meshes, and materials as objects with named properties. Internally, hot rendering data lives in contiguous typed arrays.

```typescript
// PUBLIC API: familiar object-oriented usage
const mesh = createMesh(geometry, material)
mesh.position.set(1, 0, 3)
mesh.castShadow = true
scene.add(mesh)

// INTERNAL: the renderer iterates flat arrays, never walks the object graph
// worldMatrices: Float32Array  — contiguous, GPU-uploadable
// sortKeys: Uint32Array        — 32-bit composite keys for radix sort
// visibilityFlags: Uint8Array  — frustum culling results
// aabbPool: Float32Array       — per-node AABBs for culling
```

When a user sets `mesh.position`, it marks the node dirty (`_dirtyLocal = true`). The renderer never reads from individual objects during rendering — it reads the contiguous arrays. Dirty flags trigger selective updates from the object graph into the arrays once per frame, before rendering.

### Why This Hybrid Works

- **For users**: familiar, debuggable. `console.log(mesh.position)` shows `[1, 0, 3]`, not an opaque index.
- **For the renderer**: cache-friendly iteration. Frustum-culling 2000 AABBs in a contiguous `Float32Array` is ~10x faster than chasing 2000 heap-allocated objects.
- **For the GPU**: the world matrix array uploads directly — no per-object copies needed.

## Scene Graph Update Strategy

Deferred dirty propagation with depth-first update, executed once per frame before rendering.

**Two-level dirty flags:**

- `_dirtyLocal`: local TRS changed, local matrix needs recomputation
- `_dirtyWorld`: world matrix needs recomputation (self or ancestor changed)

```typescript
// Phase 1: Propagate dirty flags down the tree
const propagateDirty = (node: Node): void => {
  if (node._dirtyLocal) node._dirtyWorld = true
  if (node._dirtyWorld) {
    for (const child of node.children) {
      if (!child._dirtyWorld) {
        // Early exit: already marked
        child._dirtyWorld = true
        propagateDirty(child)
      }
    }
  }
}

// Phase 2: Recompute only dirty matrices (depth-first)
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
    // Write to contiguous matrix pool at this node's index
    worldMatrixPool.set(node._worldMatrix, node._poolIndex * 16)
    node._dirtyWorld = false
  }
  for (const child of node.children) updateWorldMatrices(child)
}
```

**Typical frame impact:** 2-5% of objects move → 95-98% of matrix work skipped. Cost: ~0.3ms for 1000 dirty nodes.

The early-exit optimization in `propagateDirty` prevents O(n²) cascading: if a child is already marked `_dirtyWorld`, its entire subtree was already processed.

## Render List: Rebuild Every Frame

Each frame, the visible mesh list is rebuilt from scratch:

```typescript
// Each frame:
const visibleList = frustumCull(scene, camera) // collect visible meshes
radixSort(visibleList, sortKeyOf) // sort by 32-bit composite key
buildDrawCommands(visibleList) // emit draw calls
```

For 2000 objects, culling + radix sort completes in <0.3ms. Rebuilding is simpler than maintaining a persistent sorted list (no invalidation logic, no stale entries, always correct). The marginal cost is negligible.

## Uniform Buffer Strategy

Three bind groups organized by update frequency:

| Slot | Name         | Update Frequency                    | Contents                                                                                 |
| ---- | ------------ | ----------------------------------- | ---------------------------------------------------------------------------------------- |
| 0    | Per-frame    | Once per frame                      | Camera VP matrix, directional light data, shadow VP matrix, shadow bias params, ambient light |
| 1    | Per-material | Per material switch                 | Material UBO (palette entries, base color, opacity), textures, samplers                  |
| 2    | Per-object   | Per draw call (dynamic offset only) | World matrix, bone matrices (skinned meshes)                                             |

Per-object data uses **dynamic offsets** into a shared buffer. One bind group is created for slot 2; each draw call just changes the offset. This avoids per-object bind group creation.

```typescript
// WebGPU: pass.setBindGroup(2, objectBindGroup, [byteOffset])
// WebGL2: gl.bindBufferRange(GL.UNIFORM_BUFFER, 2, buffer, byteOffset, size)
```

The shared buffer is pre-allocated for 4096 objects (4096 × 256 bytes = 1MB). If the scene exceeds this, the buffer grows geometrically (2x).

## Frame Lifecycle

Every frame executes this sequence:

```
 1. Time/clock update (deltaTime, elapsed)
 2. User callbacks (engine.onFrame / useFrame hooks)
 3. Animation system update (sample keyframes, blend, compute bone matrices)
 4. Dirty flag propagation (mark children of dirty nodes)
 5. World matrix recomputation (only dirty nodes, writes to contiguous pool)
 6. AABB update (recompute world-space AABBs for dirty nodes)
 7. Frustum culling (test AABBs against 6 camera planes)
 8. Build draw list (collect visible meshes into flat array)
 9. Sort draw list (radix sort by 32-bit key: layer > pipeline > material > depth)
10. Upload per-frame uniforms (camera, lights, shadow matrices → bind group 0)
11. Upload per-object uniforms (world matrices → shared buffer for bind group 2)
12. Execute render passes:
      a. Shadow pass (single shadow map, depth-only, light-space frustum cull)
      b. Opaque pass (sorted draws, 4x MSAA, MRT: color + emissive)
      c. Transparent pass (WBOIT accumulation into RGBA16F + R8, depth read-only)
13. MSAA resolve (multisample → single-sample for color + emissive)
14. OIT composite (blend transparent result over resolved opaque)
15. Bloom (emissive → downsample chain → upsample chain → composite)
16. Final blit to screen (gamma correction)
17. HTML overlay update (project 3D positions to screen coordinates)
18. Reset frame-scoped scratch pools
```

## GPU Abstraction Layer: `Device`

The abstraction follows WebGPU semantics. Both backends implement the same interface:

```typescript
interface Device {
  readonly backend: 'webgpu' | 'webgl2'

  createBuffer(desc: BufferDescriptor): Buffer
  createTexture(desc: TextureDescriptor): Texture
  createSampler(desc: SamplerDescriptor): Sampler
  createShaderModule(desc: ShaderDescriptor): ShaderModule
  createPipeline(desc: PipelineDescriptor): Pipeline
  createBindGroup(desc: BindGroupDescriptor): BindGroup

  beginFrame(): FrameEncoder
  submit(): void
}
```

**Key concepts:**

- **Immutable pipelines**: full render state (shader + blend + depth + cull + vertex layout + sample count) baked at creation. No per-draw state validation.
- **Explicit bind groups**: resources grouped by update frequency (per-frame, per-material, per-object).
- **Command recording**: explicit `beginRenderPass` / `endRenderPass`, even on WebGL2.

**WebGL2 translation:**

- Pipeline = `useProgram` + blend + depth + cull state applied as a bundle
- BindGroup = UBO bindings (`bindBufferRange`) + texture unit assignments (`activeTexture`/`bindTexture`)
- Full state cache: track current program, VAO, FBO, bound textures, UBOs, blend/depth/cull. Skip redundant GL calls — eliminates 40-60% of API calls.
- VAO caching: one VAO per geometry+pipeline pair, lazily created, destroyed on dispose.

## Module Communication

- **Direct function calls** between layers for hot paths (no event bus, no pub/sub for rendering)
- **Typed callbacks** for infrequent operations (asset loaded, resize, context lost)
- **Shared typed arrays** for renderer-facing data (world matrices, sort keys, bone matrices, AABBs)

## Memory Management

- **Pre-allocated pools** for math scratch variables at module scope (`_tempVec3`, `_tempMat4`, `_tempQuat`)
- **Frame-scoped scratch pool** with pointer reset at frame end (no deallocation, no GC)
- **Zero allocations in the render loop**: no `new`, no array creation, no closures in hot paths
- **GPU resources live until explicitly disposed**: no automatic GC of WebGL/WebGPU objects
- **Geometric buffer growth** for dynamic arrays: double capacity when exceeded, never shrink

## Threading Model

- **Main thread**: scene graph, render loop, user callbacks, GPU command submission
- **Web Workers**: optional BVH construction for large meshes
- **Communication**: `postMessage` with `Transferable` typed arrays (zero-copy transfer)

The render loop stays on the main thread. OffscreenCanvas would prevent HTML overlay compositing, and the synchronization overhead is not justified given the 16.6ms frame budget.

## Error Handling

- **Debug mode** (`createEngine(canvas, { debug: true })`): verbose validation errors, shader compilation diagnostics, missing attribute warnings, GPU validation layers
- **Production mode**: silent failures with `console.warn` for recoverable issues, throws for unrecoverable issues (no WebGL2/WebGPU support, context creation failure)
- **Backend fallback**: try WebGPU first, fall back to WebGL2 automatically, throw `UnsupportedBackendError` if neither available
