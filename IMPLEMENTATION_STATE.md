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
| KTX2 / Texture Compress | ✓ Implemented |
| Engine Core / Lifecycle | ✓ Implemented |
| Shadows                 | ✓ Implemented |
| Raycasting / BVH        | ✓ Implemented |
| Transparency (Sorted)   | ✓ Implemented |
| HTML Overlay            | ✓ Implemented |
| React Bindings          | ✓ Implemented |
| Bloom                   | ✓ Implemented |
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
- **Orbit Controls**: Target, damping, distance constraints, elevation constraints, mouse + touch input, `onChange` callback, `dispose()`
- **glTF Loader**: GLB parsing, mesh/skeleton/animation extraction, flat result lists (`meshes`, `skeletons`, `animations`, `textures`), `dispose()`
- **KTX2 / Basis Universal**: `loadKTX2` loader implemented in `loaders/ktx2.ts`. Selects best GPU-native format (ASTC 4×4 > BC7 > BC3 > ETC2 > RGBA8) from `engine.compressedTextureFormats`. Integrated into the glTF loader and exposed as `useKTX2` hook in React.
- **Renderer**: WebGPU + WebGL2 backends, draw call sorting by pipeline→material→depth (32-bit radix sort), MSAA, frame stats API. All shader pipelines compiled eagerly at init (no first-frame hitching). Camera frustum culling via AABB-frustum testing in `collectMeshes()`.
- **Shaders**: Dual WGSL + GLSL maintained separately
- **Raycasting / BVH**: Two-level BVH (scene mesh + triangle level), binned SAH with 12 bins, slab ray-AABB, Möller-Trumbore ray-triangle, `Raycaster` class with `set`, `setFromCamera`, `intersectObject`, `intersectObjects`. Geometry BVH cached via `WeakMap`, invalidated on `needsUpdate`.
- **Shadows**: Single shadow map, depth pass, texel snapping, PCF filtering, shadow bias. Implemented in both WebGPU and WebGL2 backends.
- **Transparency (Sorted Alpha Blend)**: Back-to-front sorted alpha blending drawn after opaques in the same render pass. Sort key layout puts depth in the most significant bits for transparent meshes (correct blending order) vs pipeline/material for opaques (state-change minimization). WebGPU uses premultiplied alpha (`one / one-minus-src-alpha`) to avoid `VK_ERROR_UNKNOWN` on Android Vulkan drivers. WebGL2 uses straight alpha (`src-alpha / one-minus-src-alpha`). Transparent pipelines disable depth writes and back-face culling. glTF `alphaMode: 'BLEND'` and `baseColorFactor` alpha are supported.
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

5. **Transparency approach changed**: The spec originally described Weighted Blended OIT (WBOIT) with a separate transparent pass and OIT composite step. The actual implementation uses sorted alpha-blend (back-to-front in the same render pass as opaques). `TRANSPARENCY.md` was updated to reflect this. `RENDERER.md` still contains stale WBOIT references (render target table, render pass pipeline steps).

6. **Project folder structure**: `ARCHITECTURE.md` described a separate `device/` GPU abstraction layer with its own `types.ts`, `webgpu.ts`, `webgl2.ts`. The actual implementation skips that indirection — WebGPU and WebGL2 renderers live directly in `renderer/webgpu.ts` and `renderer/webgl.ts` and own the GPU resource management themselves. The layered dependency flow is correct; only the physical module boundaries differ.

7. **Orbit controls — missing programmatic API**: `CONTROLS.md` specced `controls.setTarget([x,y,z], { animate: true, duration: 0.5 })`, `controls.setPosition(...)`, and `onStart`/`onEnd` event callbacks. The implementation only has `onChange`. The `target` property can be set directly (immediate, no animation). Animated transitions and the start/end callbacks were never built.

8. **Animation mixer blending**: Spec describes accumulating normalized weights per-bone across all active actions. Implementation uses incremental blending (`t = nw / accWeight` at each step) which is mathematically equivalent for two actions but avoids a separate accumulation pass. Result is identical; code path differs.

---

## Remaining Renderer Improvements

1. **Conditional Vertex Attribute Fetch** (`specs/RENDERER.md`) — Vertex buffer layouts are hardcoded. Spec calls for separate-per-attribute buffers where unused attributes are skipped rather than always bound. Would require the shader variant feature-flag system (significant refactor).

2. **Bind Group Organization** — Shadow pass uses separate bind group layouts rather than the unified approach the spec describes.

3. **WebGL2 Pipeline as State Bundle** — WebGPU correctly uses `GPURenderPipeline` objects, but the WebGL2 backend sets blend/depth/cull state ad-hoc during rendering rather than using pre-built state bundles.

4. **RENDERER.md cleanup** — The spec still describes the old WBOIT pipeline (OIT Accumulation/Revealage render targets, OIT composite pass, RGBA16F+R8 targets). These no longer exist. The spec needs a rewrite to match the sorted alpha-blend approach before it can be considered accurate.

## Priority Order

| Priority | Feature                  | Impact                   |
| -------- | ------------------------ | ------------------------ |
| 1        | Conditional vertex fetch | Memory/bandwidth savings |
