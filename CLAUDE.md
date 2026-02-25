# VoidCore – Claude Code Guidelines

## Project Overview

VoidCore is a performant 3D graphics engine written in TypeScript. It supports both WebGPU and WebGL2 rendering backends. The engine is also designed to be educational — every source file has a high-level comment block explaining what it does.

## Architecture

- **Coordinate system**: Z-up, right-handed. Forward is +Y, right is +X, up is +Z.
- **Math**: All types are backed by Float32Array with a "write into output" pattern (zero allocation).
- **Rendering**: Dual-backend (WebGPU/WebGL2), MSAA, MRT (color + emissive), single shadow map with PCF filtering, sorted alpha-blend transparency (premultiplied alpha on WebGPU), bloom post-processing, configurable DPR limiting. Per-vertex colors via `bakePalette()` which resolves `materialIndices` + palette entries → per-vertex `colors` (unorm8x4) + `emissiveColors` (float16x4 HDR); results are cached by geometry+palette reference (WeakMap) so repeated calls with the same inputs return the same Geometry. The renderer detects `geometry.colors` and selects a vertex-color shader variant; meshes without vertex colors use `material.color` directly. VC shader variants also sample the AO map, so meshes using `bakePalette()` can still benefit from ambient occlusion textures. Per-material tiled AO textures repeat across surfaces using world-space XY coordinates — `PaletteEntry` accepts `tiledAo` (Texture), `tiledAoIntensity` (default 1.0, supports HDR), and `tiledAoScale` (world-space tiling frequency, default 1.0). `bakePalette()` packs the layer index into `colors.a` (unorm8) and intensity into `emissiveColors.a` (float16); unique textures are uploaded as a 2D array texture. Tiled AO multiplies the world AO: `finalAo = worldAo * tiledAo`. Max 255 unique tiled AO textures per palette. Per-material tiled normal maps add surface detail using world-space XY coordinates — `PaletteEntry` accepts `tiledNormal` (Texture), `tiledNormalIntensity` (default 1.0), and `tiledNormalScale` (world-space tiling frequency, default 1.0). `bakePalette()` stores normal data in a separate `tiledNormalData` (float16x4) vertex attribute at location 7: `[layerIndex/255, intensity, scale, 0]`. Unique normal textures are uploaded as a 2D array texture (binding 6). Normal mapping uses cotangent-frame (screen-space derivatives) for static meshes and triplanar projection for skinned meshes. Max 255 unique tiled normal textures per palette. Per-material noise color blending via `color2` and `noiseScale` on `PaletteEntry` — `bakePalette()` stores `[color2.r, color2.g, color2.b, noiseScale]` in `noiseColorData` (float16x4), interleaved with `tiledNormalData` into a single GPU buffer (stride 16, locations 7+8). The shader uses a procedural 3D dot noise function (3 octaves) evaluated in world space to blend between the primary and secondary color; when `noiseScale` is 0 (no `color2`), noise is skipped entirely. Texture maps (color map, AO map) on Lambert materials with KTX2/Basis Universal support — the KTX2 loader transcodes to GPU-native compressed formats (ASTC 4×4, BC7, BC3, ETC2 RGBA) when available, falling back to RGBA8. Both renderers detect supported compressed formats at init (`engine.compressedTextureFormats`). Shadow configuration (ortho box size, near/far, bias) lives on the DirectionalLight; the renderer only owns the texture resolution. Shadow baking freezes the shadow map for static scenes (`engine.shadowsBaked` / `<BakeShadows />`). Per-material face culling via `side` property (`'front'`/`'back'`/`'double'`). Single-draw-call inverted hull outlines on meshes via `outline` option (thickness + color, works with skinned meshes) — combined geometry (doubled vertices with smooth normals) drawn in one call with cullMode:none and front_facing discard in the fragment shader. Per-mesh distance culling via `maxDistance` (0 = disabled; hides mesh and its shadows when camera is farther than the specified distance; uses squared distance for zero-allocation performance). Custom shader materials via `customShader` option on BasicMaterial/LambertMaterial/SpriteMaterial — users provide WGSL/GLSL code snippets injected at hook points in the vertex stage (after worldPos/normal/uv, before clip projection) and fragment stage (after finalColor/alpha, before output). Per-material pipeline/program caching via WeakMap. Custom uniforms (`uniforms: Record<string, number>`) pass arbitrary float values from JS to the shader — accessible as `uniforms.xxx` in both WGSL and GLSL; update from JS each frame via `material.customShader.uniforms.xxx = value`. Backed by a GPU uniform buffer (group 3 / binding point 2) with per-material caching.
- **Raycasting**: BVH-accelerated ray-mesh intersection for collision detection, line-of-sight checks, ground placement, pseudo-physics, and pointer picking. `intersectObject`/`intersectObjects` return hits sorted by distance. For zero-allocation raycasting (e.g. per-frame collision tests or pointer raycasts), pass a pre-allocated target array: `const hits = [createRaycastHit()]; const count = raycaster.intersectObject(obj, false, hits);` — the raycaster writes into the pre-allocated `RaycastHit` objects and returns the hit count instead of a new array.
- **Sprites**: Billboard planes that always face the camera. `Sprite` extends `Mesh` with a shared 1×1 `PlaneGeometry` and `SpriteMaterial` (unlit, transparent by default, front-facing only). The renderer computes a billboard world matrix on the CPU in the batch fill phase by extracting camera right/up vectors from the view matrix and applying the sprite's position and scale. `SpriteMaterial` adds `rotation` (2D angle in radians around the view axis, default 0) and `sizeAttenuation` (when false, sprite maintains constant screen size regardless of distance; default true). Sprites do not cast shadows by default. React: `<sprite>` + `<spriteMaterial>`.
- **Scheduler**: Single rAF loop with priority-ordered callbacks, global/per-callback FPS caps. Both `maxFps` and `maxDpr` can be changed dynamically at runtime.
- **Scene graph**: Tree of Nodes with dirty-flag world matrix propagation. Transforms are set via `setPosition(x, y, z)`, `setRotation(x, y, z, w)`, `setScale(x, y, z)` / `setScale(s)` which automatically mark the node dirty. Per-component setters are also available: `setPositionX/Y/Z(v)`, `setScaleX/Y/Z(v)`. `markTransformDirty()` is available for code that writes directly to the underlying Float32Arrays (e.g., `quatFromAxisAngle`).
- **HTML overlay**: DOM elements tracked to 3D world positions via CSS transforms. Supports node tracking with offset, centering, dirty checking, depth-based z-index, distance scaling, and per-element pointer events.
- **React bindings**: Optional declarative layer based on React Three Fiber's API design, exported directly from `voidcore`. Uses a custom `react-reconciler` to map JSX elements (`<mesh>`, `<boxGeometry>`, `<lambertMaterial>`, `<directionalLight>`, `<ambientLight>`, etc.) to engine objects. Provides hooks (`useFrame`, `useEngine`, `useGLTF`, `useKTX2`, `useColoredGeometry`, `useColoredStaticGeometry`, `useAnimations`) and a `<Canvas>` root component. `useGLTF` supports `{ meshName }` option to return a single `Mesh` by name instead of the full `GLTFResult`. `useGLTF` with `{ meshName, clone: true }` returns a `ClonedMesh` with its own bone hierarchy, skeleton, and animations for independent instancing. `useColoredGeometry(geometry, palette)` is a memoized wrapper around `bakePalette` for vertex-colored geometry. `useColoredStaticGeometry(meshName, palette)` is a convenience hook combining `useGLTF` + `useColoredGeometry` for named meshes in a static bundle GLB; call `useColoredStaticGeometry.setStaticBundlePath(url)` at module level to configure the bundle path.
- **Web Worker**: Optional geometry worker offloads CPU-intensive operations (BVH construction, palette baking, mesh merging, smooth normals) to a background thread. Call `initGeometryWorker(worker)` once at startup with a Worker instance pointing at the built `geometry-worker` module. Worker-backed React hooks (`useWorkerColoredGeometry`, `useWorkerColoredStaticGeometry`, `useWorkerMergeStaticIntoSkinned`, `useWorkerPrebuildBVH`) use Suspense for async results. All async functions (`bakePaletteAsync`, `buildBVHAsync`, etc.) fall back to synchronous execution on the main thread if no worker is initialized — the worker is an optimization, not a requirement. The worker is self-contained (inlined math functions, no engine imports) for bundler portability.

## Project Structure

Monorepo with bun workspaces. Linting (oxlint) and formatting (oxfmt) are configured at the root.

```
library/       – The library package (published to npm as "voidcore")
  src/
    engine.ts            – Main entry point, owns the scheduler and renderer
    scheduler.ts         – Priority-based rAF loop with FPS throttling
    index.ts             – Public API barrel export
    animation/           – Skeletal animation (clips, mixer, skeleton)
    controls/            – Camera controls (orbit)
    geometry/            – Geometry data and procedural primitives
    loaders/             – Asset loaders (glTF/GLB, KTX2/Basis Universal)
    materials/           – Material definitions (basic, lambert, sprite) and textures
    math/                – Linear algebra (vec3, mat4, quat, AABB, frustum)
    helpers/             – Visual debug helpers (DirectionalLightHelper)
    raycasting/          – Ray-mesh intersection with BVH acceleration
    workers/             – Web worker for off-main-thread geometry processing
      geometry-worker.ts – Self-contained worker (BVH, bakePalette, merge, smooth normals)
      index.ts           – Worker manager + async API (initGeometryWorker, *Async functions)
    renderer/            – Rendering backends and shaders
      renderer.ts        – Interface + factory
      webgpu.ts          – WebGPU backend
      webgl.ts           – WebGL2 backend
      webgl-shaders.ts   – GLSL shaders
      webgpu-shaders.ts  – WGSL shaders
      shared.ts          – Shared traversal/culling utilities
      pack.ts            – Vertex attribute packing (snorm8, float16, unorm8)
      sort.ts            – Radix sort for draw order
    overlay.ts           – HTML overlay manager (DOM elements tracking 3D positions)
    scene/               – Scene graph nodes (Node, Scene, Mesh, Group, Camera, Light)
    react/               – React declarative bindings (exported from voidcore top-level)
      BakeShadows.tsx    – Freezes shadow map for static scenes
      Canvas.tsx         – Root component (engine init, reconciler mount, rAF loop)
      reconciler.ts      – Custom react-reconciler host config
      types.ts           – JSX catalogue, prop types, IntrinsicElements
      hooks.ts           – useFrame, useEngine, useLoader, useGLTF, useKTX2, useAnimations
      events.ts          – Pointer event system (raycast-based)
      context.ts         – React context for engine state
      Html.tsx           – DOM overlay projected to 3D coordinates
      index.ts           – Internal barrel (not a public entry point)
  package.json           – Library package (exports src/index.ts; tsup builds dist/ for npm)
  tsconfig.json          – Library tsconfig (jsx: react-jsx, noEmit: true)
  tsup.config.js         – tsup build config for npm publishing

example/         – Next.js demo app (imports voidcore from workspace)
  pages/                 – Next.js pages
  components/            – Example-specific React components
  public/                – Static assets (draco, basis, ktx2, glb files)
  package.json           – Example package (depends on workspace:voidcore)
  tsconfig.json          – Next.js tsconfig (jsx: preserve, plugins: [next])
  next.config.mjs        – Next.js config (transpilePackages: ['voidcore'])
  postcss.config.mjs     – PostCSS / Tailwind config
  tailwind.css           – Tailwind CSS entry
```

## Commands

All commands run from the monorepo root unless noted.

- `bun run dev` – Start the Next.js example dev server
- `bun run build` – Build all packages (library via tsup, example via next build)
- `bun run lint` – Lint with oxlint
- `bun run format` – Format with oxfmt
- `bun run buntest` – Run tests with bun
- `bun run typecheck` – Typecheck all packages with tsgo
- `bun run all` – Lint + format check + test + typecheck
- `bun run pub` – Build the library and publish to npm

## Conventions

- Constructors: `new Xxx()` pattern for all public API objects (Three.js-style). Engine uses `Engine.create()` since initialization is async.
- No runtime allocations in hot paths (render loop, animation updates). Pre-allocate buffers.
- All primitives are Z-up, centered at origin.
- Column-major matrices (Mat4), quaternions as [x,y,z,w].
- Educational comments at the top of every engine source file — keep them up to date.

## Important

Any significant change to the engine should be reflected in:

1. **This file (CLAUDE.md)** – Update architecture, structure, or conventions if they change.
2. **README.md** – Update user-facing documentation if the public API or usage changes.
3. **File header comments** – Update the educational comment at the top of any modified file.
