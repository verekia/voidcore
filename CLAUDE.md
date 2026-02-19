# VoidCore – Claude Code Guidelines

## Project Overview

VoidCore is a performant 3D graphics engine written in TypeScript. It supports both WebGPU and WebGL2 rendering backends. The engine is also designed to be educational — every source file has a high-level comment block explaining what it does.

## Architecture

- **Coordinate system**: Z-up, right-handed. Forward is +Y, right is +X, up is +Z.
- **Math**: All types are Float32Array-backed with a "write into output" pattern (zero allocation).
- **Rendering**: Dual-backend (WebGPU/WebGL2), MSAA, MRT (color + emissive), 3-cascade shadow maps (CSM), bloom post-processing, configurable DPR limiting. Transparency uses WBOIT (Weighted Blended OIT) when `OES_draw_buffers_indexed` is available, with a sorted alpha-blend fallback for devices that lack the extension (some Android GPUs).
- **Scheduler**: Single rAF loop with priority-ordered callbacks, global/per-callback FPS caps. Both `maxFps` and `maxDpr` can be changed dynamically at runtime.
- **Scene graph**: Tree of Nodes with dirty-flag world matrix propagation.

## Project Structure

```
src/
  engine.ts              – Main entry point, owns the scheduler and renderer
  scheduler.ts           – Priority-based rAF loop with FPS throttling
  index.ts               – Public API barrel export
  animation/             – Skeletal animation (clips, mixer, skeleton)
  controls/              – Camera controls (orbit)
  geometry/              – Geometry data and procedural primitives
  loaders/               – Asset loaders (glTF/GLB)
  materials/             – Material definitions (basic, lambert)
  math/                  – Linear algebra (vec3, mat4, quat, AABB, frustum)
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
  scene/                 – Scene graph nodes (Node, Scene, Mesh, Group, Camera, Light)
pages/                   – Next.js example app (not part of the engine)
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
