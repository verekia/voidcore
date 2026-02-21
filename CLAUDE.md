# VoidCore – Claude Code Guidelines

## Project Overview

VoidCore is a performant 3D graphics engine written in TypeScript. It supports both WebGPU and WebGL2 rendering backends. The engine is also designed to be educational — every source file has a high-level comment block explaining what it does.

## Architecture

- **Coordinate system**: Z-up, right-handed. Forward is +Y, right is +X, up is +Z.
- **Math**: All types are backed by typed float arrays (Float32Array by default, configurable to Float16Array via `floatPrecision` option) with a "write into output" pattern (zero allocation).
- **Rendering**: Dual-backend (WebGPU/WebGL2), MSAA, MRT (color + emissive), single shadow map with PCF filtering, sorted alpha-blend transparency (premultiplied alpha on WebGPU), bloom post-processing, configurable DPR limiting. Texture maps (color map, AO map) on Lambert materials with KTX2/Basis Universal support — the KTX2 loader transcodes to GPU-native compressed formats (ASTC 4×4, BC7, BC3, ETC2 RGBA) when available, falling back to RGBA8. Both renderers detect supported compressed formats at init (`engine.compressedTextureFormats`). Shadow configuration (ortho box size, near/far, bias) lives on the DirectionalLight; the renderer only owns the texture resolution. Shadow baking freezes the shadow map for static scenes (`engine.shadowsBaked` / `<BakeShadows />`). Per-material face culling via `side` property (`'front'`/`'back'`/`'double'`). Inverted hull outlines on meshes via `outline` option (thickness + color, works with skinned meshes). Per-mesh distance culling via `maxDistance` (0 = disabled; hides mesh and its shadows when camera is farther than the specified distance; uses squared distance for zero-allocation performance).
- **Scheduler**: Single rAF loop with priority-ordered callbacks, global/per-callback FPS caps. Both `maxFps` and `maxDpr` can be changed dynamically at runtime.
- **Scene graph**: Tree of Nodes with dirty-flag world matrix propagation. Transforms are set via `setPosition(x, y, z)`, `setRotation(x, y, z, w)`, `setScale(x, y, z)` / `setScale(s)` which automatically mark the node dirty. Per-component setters are also available: `setPositionX/Y/Z(v)`, `setScaleX/Y/Z(v)`. `markTransformDirty()` is available for code that writes directly to the underlying Float32Arrays (e.g., `quatFromAxisAngle`).
- **HTML overlay**: DOM elements tracked to 3D world positions via CSS transforms. Supports node tracking with offset, centering, dirty checking, depth-based z-index, distance scaling, and per-element pointer events.
- **React bindings**: Optional declarative layer via `voidcore/react` subpath. Uses a custom `react-reconciler` to map JSX elements (`<mesh>`, `<boxGeometry>`, `<lambertMaterial>`, `<directionalLight>`, `<ambientLight>`, etc.) to engine objects. Provides hooks (`useFrame`, `useEngine`, `useGLTF`, `useKTX2`, `useAnimations`) and a `<Canvas>` root component.

## Project Structure

```
src/
  engine.ts              – Main entry point, owns the scheduler and renderer
  float.ts               – Configurable float precision (Float32Array/Float16Array)
  scheduler.ts           – Priority-based rAF loop with FPS throttling
  index.ts               – Public API barrel export
  animation/             – Skeletal animation (clips, mixer, skeleton)
  controls/              – Camera controls (orbit)
  geometry/              – Geometry data and procedural primitives
  loaders/               – Asset loaders (glTF/GLB, KTX2/Basis Universal)
  materials/             – Material definitions (basic, lambert) and textures
  math/                  – Linear algebra (vec3, mat4, quat, AABB, frustum)
  helpers/               – Visual debug helpers (DirectionalLightHelper)
  raycasting/            – Ray-mesh intersection with BVH acceleration
  renderer/              – Rendering backends and shaders
    renderer.ts          – Interface + factory
    webgpu.ts            – WebGPU backend
    webgl.ts             – WebGL2 backend
    webgl-shaders.ts     – GLSL shaders
    webgpu-shaders.ts    – WGSL shaders
    shared.ts            – Shared traversal/culling utilities
    pack.ts              – Vertex attribute packing (snorm8, float16, unorm8)
    sort.ts              – Radix sort for draw order
  overlay.ts             – HTML overlay manager (DOM elements tracking 3D positions)
  scene/                 – Scene graph nodes (Node, Scene, Mesh, Group, Camera, Light)
  react/                 – React declarative bindings (voidcore/react subpath)
    BakeShadows.tsx      – Freezes shadow map for static scenes
    Canvas.tsx           – Root component (engine init, reconciler mount, rAF loop)
    reconciler.ts        – Custom react-reconciler host config
    types.ts             – JSX catalogue, prop types, IntrinsicElements
    hooks.ts             – useFrame, useEngine, useLoader, useGLTF, useKTX2, useAnimations
    events.ts            – Pointer event system (raycast-based)
    context.ts           – React context for engine state
    Html.tsx             – DOM overlay projected to 3D coordinates
    index.ts             – Barrel export
pages/                   – Next.js example app (uses React bindings)
```

## Commands

- `bun run dev` – Start dev server (Next.js example)
- `bun run build` – Build
- `bun run lint` – Lint with oxlint
- `bun run format` – Format with oxfmt
- `bun run buntest` – Run tests with bun
- `bun run typecheck` – Typecheck with tsgo
- `bun run all` – Lint + format check + test + typecheck

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
