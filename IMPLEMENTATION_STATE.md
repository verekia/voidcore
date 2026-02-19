# Implementation State

Audit of the Voidcore specs vs current implementation.

## Feature Completion Matrix

| Feature Category        | Status        |
| ----------------------- | ------------- |
| Math Library            | ✓ Implemented |
| Scene Graph             | ✓ Implemented |
| Materials               | ✓ Implemented |
| Geometry                | ✓ Implemented |
| Controls (Orbit)        | ✓ Implemented |
| glTF Loader (core)      | ✓ Implemented |
| Engine Core / Lifecycle | ✓ Implemented |
| Shadows (CSM)           | ✓ Implemented |
| Raycasting / BVH        | ✓ Implemented |
| Transparency (WBOIT)    | ✓ Implemented |
| HTML Overlay            | ✓ Implemented |
| React Bindings          | ✓ Implemented |
| Bloom                   | ✓ Implemented |
| Textures/KTX2           | ❌ Not done   |
| Animation               | ⚠️ Partial    |
| Renderer Core           | ⚠️ Partial    |

---

## Implemented as Specced

- **Architecture**: Layered design, engine init, WebGPU-first with WebGL2 fallback, frame lifecycle (`onFrame`, `start`, `stop`, `render`)
- **Scene Graph**: Node base class with position/rotation/scale, parent/children, `add/remove/traverse`, `lookAt`, visibility, shadow flags, `getByName` with Map cache
- **Math**: Full Z-up, column-major library — Vec3, Vec4, Quat, Mat4, AABB with all specified operations
- **Materials**: BasicMaterial, LambertMaterial, color/opacity/transparent/vertexColors, index palette system (32-entry max)
- **Geometry**: All parametric primitives (Plane, Box, Sphere, Cone, Cylinder, Capsule, Circle), Z-up, custom geometry factory, lazy GPU upload, `needsUpdate`
- **Animation**: Skeleton (32-bone limit), keyframe tracks, mixer with actions, fade in/out, crossfade, weight blending, loop modes, timeScale
- **Orbit Controls**: Target, damping, distance constraints, elevation constraints, mouse + touch input, callbacks, `dispose()`
- **glTF Loader**: GLB parsing, mesh/skeleton/animation extraction, flat result lists (`meshes`, `skeletons`, `animations`, `textures`), `dispose()`
- **Renderer**: WebGPU + WebGL2 backends, draw call sorting by pipeline→material→depth (32-bit radix sort), MSAA, frame stats API. All shader pipelines compiled eagerly at init (no first-frame hitching). Camera frustum culling via AABB-frustum testing in `collectMeshes()`.
- **Shaders**: Dual WGSL + GLSL maintained separately
- **Raycasting / BVH**: Two-level BVH (scene mesh + triangle level), binned SAH with 12 bins, slab ray-AABB, Möller-Trumbore ray-triangle, `Raycaster` class with `set`, `setFromCamera`, `intersectObject`, `intersectObjects`. Geometry BVH cached via `WeakMap`, invalidated on `needsUpdate`.
- **Shadows (CSM)**: 3-cascade shadow maps, depth pass, texel snapping, PCF filtering, cascade blending, shadow bias. Implemented in both WebGPU and WebGL2 backends.
- **Transparency (WBOIT)**: Two-pass OIT with accumulation (RGBA16F) and revealage buffers, McGuire weight function, MRT targets, composite pass. Full pipeline: Shadow → Opaque → WBOIT → Resolve → OIT Composite → Bloom → Blit. Implemented in both WebGPU and WebGL2 backends. WebGL2 falls back to sorted alpha blending on devices lacking `OES_draw_buffers_indexed` (some Android GPUs).
- **Bloom**: 5-level RGBA16F downsample/upsample chain driven by MRT emissive output. 13-tap Jimenez downsample with Karis average on the first mip (firefly suppression). 9-tap tent upsample with additive blending. Final blit merges bloom composite + gamma correction in a single pass. Implemented in both WebGPU (WGSL) and WebGL2 (GLSL) backends.
- **Backend error handling**: WebGPU-first with automatic WebGL2 fallback in `'auto'` mode. Descriptive errors when an explicitly requested backend is unavailable.

---

## Not Yet Implemented

1. **Cubic spline interpolation** (`ANIMATION.md`) — Only linear and step implemented; cubic spline missing. The glTF loader silently converts `CUBICSPLINE` to `LINEAR`. Needs: `'CUBICSPLINE'` added to `KeyframeTrack.interpolation`, tangent data preserved in loader, Hermite evaluation in mixer.

---

## Implemented but Differs from Spec

1. **Dirty flag propagation**: Spec describes a two-phase approach (propagate flags then recompute) with early-exit to skip clean subtrees. Implementation uses a single-phase traversal (`updateWorldMatrices` in `node.ts`) that visits all nodes but only recomputes dirty matrices. Both are correct; the spec visits fewer nodes but the single-pass approach has simpler code and better cache locality.

2. ~~**Animation scratch arrays**~~: Fixed — all scratch buffers including `_cachedIndices` are now pre-allocated at creation time. Zero allocations in the update loop.

3. **Vertex buffer layout**: WebGPU always allocates zero-filled UV and materialIndex buffers even when geometry lacks those attributes, and all shaders unconditionally declare all 4 attribute slots. WebGL2 is closer to spec (conditionally allocates buffers, disables unused attributes). Neither backend implements the feature-flag shader variant system from the spec. Skinned attributes (joints/weights) are correctly conditional in both backends.

4. **Bind group organization**: Main opaque/transparent passes correctly use 3 groups (per-frame, per-material, per-object with dynamic offsets). Shadow pass uses a separate `shadowBGL` layout instead of reusing the unified per-frame group. Post-processing passes use custom single-purpose layouts (reasonable deviation).

---

## Remaining Renderer Improvements

1. **Per-Cascade Shadow Frustum Culling** — Camera frustum culling correctly uses AABB-frustum testing, but shadow cascade culling uses a point-in-NDC check (mesh world center only). The math functions are available — the `collectMeshes()` pattern just needs to be applied to the shadow pass.

2. **Conditional Vertex Attribute Fetch** (`specs/RENDERER.md`) — Vertex buffer layouts are hardcoded. Spec calls for separate-per-attribute buffers where unused attributes are skipped rather than always bound. Would require the shader variant feature-flag system (significant refactor).

3. **Bind Group Organization** — Shadow pass uses separate bind group layouts rather than the unified approach the spec describes.

4. **WebGL2 State Cache** (`specs/RENDERER.md`) — Only `_lastMaterial` and `_lastProgram` are tracked. VAO bindings, UBO bindings, texture bindings, framebuffer bindings, and blend/depth/cull state are set redundantly. Spec estimates 40-60% reduction in redundant GL calls with a full `StateCache`.

5. **WebGL2 Pipeline as State Bundle** — WebGPU correctly uses `GPURenderPipeline` objects, but the WebGL2 backend sets blend/depth/cull state ad-hoc during rendering rather than using pre-built state bundles.

## Priority Order

| Priority | Feature                  | Impact                     |
| -------- | ------------------------ | -------------------------- |
| 1        | WebGL2 state cache       | Performance on WebGL2 path |
| 2        | Per-cascade AABB culling | Shadow accuracy at edges   |
| 3        | Conditional vertex fetch | Memory/bandwidth savings   |
