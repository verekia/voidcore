# VoidCore vs v1v2-engine Comparison

Both engines share the same author and core DNA. The math library, BVH, GLB loader, orbit controls, bloom pipeline, and vertex format are essentially identical code. The key differences are architectural choices and feature scope.

## Architecture: The Fundamental Difference

| Aspect                    | VoidCore                                                                     | v1v2-engine                                                                                 |
| ------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **Data model**            | OOP scene graph (Node/Mesh/Group/Camera classes, parent-child tree)          | Flat Structure-of-Arrays (SoA) — parallel typed arrays for positions, colors, scales, masks |
| **Transform propagation** | Dirty-flag world matrix tree traversal                                       | Flat `m4FromTRS` per-mesh each frame, bone attachment pass in second loop                   |
| **Mesh creation**         | `createMesh(geometry, material)` + add to scene graph                        | `new Mesh({geometry, position, color, ...})` + `scene.add(mesh)`                            |
| **Material system**       | Separate `Material` class with palette support, `type: 'basic' \| 'lambert'` | Inline per-mesh properties (`unlit`, `color`, `alpha`, `bloom`, `outline`)                  |
| **Frustum culling**       | AABB-based (transforms AABB to world, tests against frustum planes)          | Bounding sphere-based (radius \* maxScale, faster but less tight)                           |
| **Render loop ownership** | Engine owns the loop (`engine.start()`, `engine.onFrame()`)                  | User-controlled `Scheduler` class with priority callbacks + per-callback FPS throttling     |

v1v2's SoA approach is more cache-friendly for the renderer (data is already in the format the GPU upload loop needs). VoidCore's scene graph is more conventional and flexible for complex hierarchies but adds overhead copying to GPU-friendly layouts. v1v2's Scheduler is significantly more powerful than VoidCore's simple onFrame callback.

## What's Identical (or Nearly So)

These modules are essentially the same implementation with minor naming differences:

| Module                    | Notes                                                                                                                                                                                                                                                                                                                                  |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Math library**          | Both use Float32Array + offset pattern (`out, o, a, ao, ...`). Same functions: `v3Set`, `v3Normalize`, `m4Multiply`, `m4LookAt`, `m4Perspective`, `m4PerspectiveGL`, `m4Ortho`, `m4OrthoGL`, `m4Invert`, `m4FromTRS`, `m4FromQuatTRS`, `quatSlerp`, `frustumContainsSphere`, `m4ExtractFrustumPlanes`. Code is functionally identical. |
| **BVH construction**      | SAH-binned build, same constants (`MAX_LEAF_TRIS=4`, `SAH_BINS=12`, `NODE_FLOATS=8`), same flat-array layout, same partition logic.                                                                                                                                                                                                    |
| **BVH raycast**           | Slab test + Moller-Trumbore, shared `_stack` pre-allocated, same early termination.                                                                                                                                                                                                                                                    |
| **GLB/glTF loader**       | Same chunk parsing, same Draco integration, same `readAccessorFloat32`/`readAccessorIndices` helpers, same node hierarchy + animation parsing.                                                                                                                                                                                         |
| **KTX2 loader**           | Identical Basis Universal transcoder loading pattern, same `cTFRGBA32=13` constant.                                                                                                                                                                                                                                                    |
| **Orbit controls**        | Same spherical coordinates, same pointer events, same pinch-zoom + pan logic, same `updateEye()`.                                                                                                                                                                                                                                      |
| **Vertex format**         | Both use 10-float interleaved stride: `[px,py,pz, nx,ny,nz, cr,cg,cb, bloom]` = 40 bytes.                                                                                                                                                                                                                                              |
| **Coordinate system**     | Z-up, right-handed, +Y forward, +X right.                                                                                                                                                                                                                                                                                              |
| **Column-major matrices** | Same layout, same multiplication order.                                                                                                                                                                                                                                                                                                |
| **Bloom post-processing** | 5-mip downsample/upsample chain, 13-tap downsample, 9-tap tent upsample, Karis average on first pass.                                                                                                                                                                                                                                  |

## Where v1v2-engine Does Better

1. **Cascaded Shadow Mapping (CSM):** Full CSM with configurable cascade count (1-4), PSSM lambda blend (log/uniform), bounding-sphere-fit per cascade, and PCF9 sampling in the atlas. VoidCore uses a single shadow map instead.

2. **Outline rendering:** MRT-based outline system with per-group outlines, configurable thickness, color, and distance-based scaling. VoidCore doesn't have this.

3. **Scheduler:** Full priority-based rAF system with per-callback FPS throttling, global FPS cap with remainder tracking, pause/resume, and frame state injection. VoidCore has a simpler engine loop.

4. **HTML overlay system:** `HtmlOverlay`/`HtmlElement` that projects DOM elements to 3D world positions with distance-based scaling. VoidCore doesn't have this.

5. **Backend switching at runtime:** Can `scene.switchBackend(newCanvas, 'webgl')` at runtime, re-registering all geometries and textures on the new renderer. VoidCore doesn't expose this.

6. **Bone attachment system:** `scene.attachToBone(weaponMesh, characterMesh, 'Hand.R')` attaches meshes to skeleton bones with proper world matrix composition. VoidCore has skeleton support but the attachment API is less streamlined.

7. **Per-vertex bloom:** Carries a per-vertex `bloom` float in the vertex stride and a `bloomWhiten` uniform, allowing selective glow on specific vertices. VoidCore also has bloom but the per-vertex control is less explicit.

8. **Storage buffer for joints:** Uses `read-only-storage` buffer with dynamic offsets for joint matrices (128 joints max), which is more scalable than VoidCore's UBO approach (32 bones max).

9. **SoA cache efficiency:** The render loop iterates flat typed arrays directly — the `RenderScene` interface passes pre-packed parallel arrays to the renderer, avoiding per-mesh property access overhead.

## Where VoidCore Does Better

1. **Scene graph hierarchy:** Proper Node tree with parent-child relationships, dirty-flag propagation, and `lookAt()`. v1v2 is flat — no scene hierarchy, no parenting (except bone attachment).

2. **Material abstraction:** Separate `Material` class with palette support (32 palette entries with color + opacity + emissive per entry), `receiveShadow` flag, and explicit material types. v1v2 bakes everything into mesh properties.

3. **AABB frustum culling:** Computes per-geometry AABBs and transforms them for frustum testing (tighter culling). v1v2 uses bounding spheres (faster but more conservative — passes more meshes to the GPU).

4. **Geometry variety:** 7 procedural primitives (plane, box, sphere, cone, cylinder, capsule, circle). v1v2 has 2 (box, sphere) + merge utility.

5. **Animation system:** Full `AnimationMixer` with `AnimationAction` objects supporting `fadeIn`/`fadeOut`/`crossFadeTo`, loop modes (repeat/once/pingpong), and weight-based blending across multiple simultaneous clips. v1v2 has a simpler `transitionTo(skin, clipIndex, blendDuration)` — less flexible but sufficient for most cases.

6. **Raycaster API:** `setFromCamera(ndc, camera)` for screen-space picking with UV interpolation in hit results. v1v2's raycast takes raw world-space origin/direction only.

7. **Radix sort for draw order:** 4-pass LSD radix sort with composite sort keys (layer, pipeline, material, depth) for optimal GPU state change minimization. v1v2 draws in fixed order (unlit -> opaque static -> opaque textured -> opaque skinned -> transparent sorted) without sorting within each category.

8. **WeakMap GPU caching:** Caches GPU buffers in WeakMaps keyed by Geometry/Material objects, allowing automatic GC when objects are no longer referenced. v1v2 uses integer ID maps that require manual cleanup.

9. **Educational comments:** Detailed header comments in every source file explaining what each module does. v1v2 has one-line comments.

10. **TypeScript types:** Defines `Vec3`, `Vec4`, `Quat`, `Mat4`, `AABB` as named type aliases for Float32Array, making function signatures more readable. v1v2 uses raw `Float32Array` everywhere.

## Feature Parity Summary

| Feature                | VoidCore                | v1v2                             |
| ---------------------- | ----------------------- | -------------------------------- |
| WebGPU                 | Yes                     | Yes                              |
| WebGL2 fallback        | Yes                     | Yes                              |
| MSAA 4x                | Yes                     | Yes                              |
| Bloom                  | Yes                     | Yes                              |
| Shadow mapping         | Yes (single map)        | Yes (1-4 cascades, configurable) |
| Outline rendering      | No                      | Yes                              |
| Frustum culling        | AABB                    | Bounding sphere                  |
| BVH raycasting         | Yes                     | Yes                              |
| Skeletal animation     | Yes (mixer + crossfade) | Yes (simpler transition API)     |
| Bone attachment        | Yes                     | Yes                              |
| GLB/glTF loading       | Yes                     | Yes                              |
| Draco compression      | Yes                     | Yes                              |
| KTX2 textures          | No                      | Yes                              |
| Scene graph            | Yes (tree)              | No (flat)                        |
| Material system        | Yes (class)             | No (mesh props)                  |
| Procedural primitives  | 7 types                 | 2 types                          |
| HTML overlay           | No                      | Yes                              |
| Scheduler              | Simple loop             | Priority + FPS throttle          |
| Runtime backend switch | No                      | Yes                              |
| Draw order sorting     | Radix sort              | Fixed-order passes               |
| Merge geometries       | No                      | Yes                              |

## Performance Comparison

**v1v2 is faster in the hot path, VoidCore is smarter about what it sends to the GPU.** In practice, v1v2 likely wins overall for game-like workloads.

### v1v2 wins on raw throughput

- **SoA data layout** — The render loop iterates flat `Float32Array`/`Uint8Array` slices sequentially. This is ideal for CPU cache lines. VoidCore traverses a tree of objects, chasing pointers between Node instances, which causes more cache misses.
- **Zero per-frame copying** — Mesh properties _are_ the SoA arrays (positions, colors, scales are written directly). VoidCore has to copy from each Mesh object's properties into GPU-friendly buffers every frame in the `render()` sync loop.
- **Storage buffer for joints** — 128 joints via `read-only-storage` with dynamic offsets. VoidCore packs 32 bones into UBOs. Storage buffers have better throughput for this use case and support 4x more joints.
- **Simpler transform path** — Flat `m4FromTRS` per mesh, no tree walking, no dirty-flag checking. Less branching in the hot loop.

### VoidCore wins on avoiding unnecessary work

- **AABB frustum culling** is tighter than bounding spheres. A sphere around a long thin object (like a bridge or wall) wastes a lot of area. AABB culling sends fewer meshes to the GPU in practice, which can matter more than the culling test speed itself.
- **Radix sort** minimizes GPU pipeline/material state changes. v1v2 draws in fixed pass order (unlit -> opaque -> textured -> skinned -> transparent) without sorting within each pass. With many different materials, VoidCore causes fewer pipeline switches, which is one of the most expensive GPU operations.
- **Dirty-flag transforms** — In mostly-static scenes, VoidCore skips recomputing world matrices for unchanged nodes. v1v2 recomputes every mesh's `m4FromTRS` every frame regardless.

### Net assessment

For v1v2's target workload (hundreds of similar skinned entities, few material types, everything moving) — v1v2 is clearly faster. The SoA iteration and lack of tree traversal overhead dominate.

For a more complex scene (many different materials, mixed static/dynamic objects, large environments with lots of off-screen geometry) — VoidCore's tighter culling and sort-based batching could close the gap or win.

The architectural differences are unlikely to be the bottleneck in either engine — at web scale, the GPU draw call count and shader complexity matter far more than the CPU-side data layout. Both engines are well-optimized for zero allocation. The real performance difference comes from features: v1v2's configurable cascade count gives better shadow quality over large distances, while VoidCore's single shadow map is simpler and faster. VoidCore's radix sort prevents the pathological "thrashing between pipelines" case that v1v2 is vulnerable to with diverse scenes.
