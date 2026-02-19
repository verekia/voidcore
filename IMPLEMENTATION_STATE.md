# Implementation State

Audit of the Voidcore specs vs current implementation.

## Feature Completion Matrix

| Feature Category        | Status             |
| ----------------------- | ------------------ |
| Math Library            | ✓ Implemented      |
| Scene Graph             | ✓ Implemented      |
| Materials               | ✓ Implemented      |
| Geometry                | ✓ Implemented      |
| Controls (Orbit)        | ✓ Implemented      |
| glTF Loader (core)      | ✓ Implemented      |
| Engine Core / Lifecycle | ✓ Implemented      |
| Shadows (CSM)           | ✓ Implemented      |
| Raycasting / BVH        | ✓ Implemented      |
| Animation               | ⚠️ Partial         |
| Renderer Core           | ⚠️ Partial         |
| Bloom                   | ⚠️ Partial         |
| Transparency (WBOIT)    | ❌ Not implemented |
| HTML Overlay            | ❌ Not implemented |
| React Bindings          | ❌ Not implemented |

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
- **Renderer**: WebGPU + WebGL2 backends, draw call sorting by pipeline→material→depth (32-bit radix sort), MSAA, frame stats API
- **Shaders**: Dual WGSL + GLSL maintained separately
- **Raycasting / BVH**: Two-level BVH (scene mesh + triangle level), binned SAH with 12 bins, slab ray-AABB, Möller-Trumbore ray-triangle, `Raycaster` class with `set`, `setFromCamera`, `intersectObject`, `intersectObjects`. Geometry BVH cached via `WeakMap`, invalidated on `needsUpdate`.
- **Shadows (CSM)**: 3-cascade shadow maps, depth pass, texel snapping, PCF filtering, cascade blending, shadow bias. Implemented in both WebGPU and WebGL2 backends.

---

## Not Yet Implemented

1. **HTML Overlay** (`HTML-OVERLAY.md`) — No `OverlayManager`, no world-to-screen projection, no CSS transform positioning, no occlusion testing.

2. **React Bindings** (`REACT.md`) — No custom reconciler, no `<Canvas>`, no JSX scene elements, no `useFrame`/`useEngine`/`useGLTF`/`useAnimations`, no `<Html>` component, no pointer events.

3. **WBOIT Transparency** (`TRANSPARENCY.md`) — Two-pass OIT (accumulation + revealage), McGuire weight function, MRT targets, composite pass — not implemented. The `transparent` flag exists but doesn't render correctly.

4. **Cubic spline interpolation** (`ANIMATION.md`) — Only linear and step implemented; cubic spline missing.

5. **Frustum culling in renderer** — Math functions (`frustumFromViewProjection`, `frustumContainsAABB`) exist in `math/index.ts` but are not hooked into the render loop.

6. **Shader warm-up** — Pre-compilation of common variants during loading not implemented.

7. **`UnsupportedBackendError`** — Mentioned in spec but not found in code.

---

## Implemented but Differs from Spec

1. **Bloom**: Shader uniforms, texture targets, and mip structure are present, but the actual downsample chain (13-tap Jimenez + Karis average) and upsample tent filter (9-tap) are not wired up. Framework exists without the implementation.

2. **Dirty flag propagation**: `_dirtyLocal`/`_dirtyWorld` flags exist in `node.ts` but the two-level propagation with early-exit described in `SCENE-GRAPH.md` appears incomplete.

3. **Animation scratch arrays**: Spec mandates zero-allocation update loop with pre-allocated scratch pools; `mixer.ts` has some pre-allocation but full scratch pool pattern is unclear.

4. **Vertex buffer layout**: Spec calls for separate-per-attribute buffers with conditional fetch (unused attributes skipped); `webgpu.ts` shows a fixed layout that doesn't appear conditional.

5. **Bind group organization**: Spec defines 3 bind groups by update frequency (per-frame, per-material, per-object with dynamic offsets); actual slot assignments in code don't fully match this layout.

---

Here's a breakdown of what's left for the renderer based on the specs:

## Critical Missing Features

1. **WBOIT Transparency** (`specs/TRANSPARENCY.md`) — The biggest gap. No OIT accumulation buffer (RGBA16F), no revealage buffer (R8), no McGuire weight function, no composite pass. The `transparent` flag on materials exists but doesn't route to a separate render pass. The entire pass pipeline should be Shadow → Opaque → **Transparent (WBOIT)** → Resolve → **OIT Composite** → Bloom → Blit, but currently there's no transparent pass or composite step.

2. **Bloom Post-Processing** (`specs/POST-PROCESSING.md`) — The framework exists (MRT emissive target, bloom texture chain) but the actual filtering is incomplete:
   - **Downsample**: Spec requires 13-tap Jimenez filter with Karis average on the first mip (firefly suppression via `1/(1+luma)` weighting)
   - **Upsample**: Spec requires 3×3 tent filter (9-tap)
   - Need to verify the shader files to confirm what's actually wired up

3. **Shader Warm-up** (`specs/RENDERER.md`) — No pre-compilation of common shader variants during loading. Spec lists 7 variants (lambert opaque+shadow+color texture, lambert+vertex colors, lambert+material index, lambert transparent, basic opaque, shadow depth-only, shadow depth-only+skinned).

## Partially Implemented / Differs from Spec

4. **Per-Cascade Shadow Frustum Culling** — Current code does a point-in-NDC check using mesh world center. Spec requires AABB-frustum intersection per cascade, which is more accurate at shadow map edges.

5. **Conditional Vertex Attribute Fetch** (`specs/RENDERER.md`) — Vertex buffer layouts are hardcoded. Spec calls for separate-per-attribute buffers where unused attributes are skipped rather than always bound.

6. **Bind Group Organization** — Mostly correct (group 0 per-frame, group 1 per-material, group 2 per-object with dynamic offsets), but shadow pass uses separate bind group layouts rather than the unified approach the spec describes.

7. **WebGL2 State Cache** (`specs/RENDERER.md`) — No `StateCache` object to track current GL state and skip redundant calls (`gl.useProgram`, `gl.bindVertexArray`, etc.). Spec estimates 40-60% reduction in redundant GL calls.

8. **WebGL2 Pipeline as State Bundle** — WebGPU correctly uses `GPURenderPipeline` objects, but the WebGL2 backend sets blend/depth/cull state ad-hoc during rendering rather than using pre-built state bundles.

## Priority Order

| Priority | Feature                                | Impact                                                                   |
| -------- | -------------------------------------- | ------------------------------------------------------------------------ |
| 1        | WBOIT Transparency                     | Transparency doesn't render correctly without it                         |
| 2        | Bloom filters (13-tap down / 9-tap up) | Visual quality — bloom framework exists but filters are wrong or missing |
| 3        | Shader warm-up                         | First-use frame hitches                                                  |
| 4        | WebGL2 state cache                     | Performance on WebGL2 path                                               |
| 5        | Per-cascade AABB culling               | Shadow accuracy at edges                                                 |
| 6        | Conditional vertex fetch               | Memory/bandwidth savings                                                 |
