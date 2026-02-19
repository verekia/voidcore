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
