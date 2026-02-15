# VoidCore

High-performance WebGPU/WebGL game engine with a C/WASM compute core. Zero dependencies in the C layer — no Emscripten, no libc. JS handles GPU API calls and scene management, C/WASM handles math, transforms, culling, and sorting. Shared memory via typed array views — no serialization overhead.

## Quick Start

```ts
import { Scene, Mesh, createBoxGeometry, createSphereGeometry } from 'voidcore'

const canvas = document.querySelector('canvas')!
const scene = await Scene.create(canvas)

// Register geometries
const boxGeo = createBoxGeometry(1, 1, 1)
const sphereGeo = createSphereGeometry(0.5, 16, 12)
const boxId = scene.registerGeometry(boxGeo)
const sphereId = scene.registerGeometry(sphereGeo)

// Add meshes
const cube = scene.add(
  new Mesh({
    geometryId: boxId,
    position: [0, 0, 1],
    color: [1, 0.3, 0.3, 1],
  }),
)

const ball = scene.add(
  new Mesh({
    geometryId: sphereId,
    position: [3, 0, 1],
    color: [0.3, 0.3, 1, 1],
    unlit: true,
  }),
)

// Lighting
scene.setDirectionalLight([0.5, 0.3, -1], [1, 0.95, 0.9])
scene.setAmbientLight([0.15, 0.15, 0.2])

// Camera
scene.camera.eye.set([0, -10, 5])
scene.camera.target.set([0, 0, 0])
scene.camera.fov = Math.PI / 4
scene.camera.near = 0.1
scene.camera.far = 500

// Render loop
const frame = () => {
  cube.rotation[2] += 0.02
  cube.setDirty()
  scene.render()
  requestAnimationFrame(frame)
}
requestAnimationFrame(frame)
```

## Features

- **WebGPU + WebGL2 fallback** — auto-detects GPU support, runtime backend switching
- **C/WASM compute core** — zero dependencies (no Emscripten, no libc), custom bump allocator
- **Structure-of-Arrays** — cache-friendly entity storage in shared WASM memory (50k max entities)
- **Frustum culling** — bounding sphere test, computed in WASM
- **Draw call sorting** — radix sort by geometry ID in WASM for batching
- **Lambert lighting** — directional + ambient, per-entity unlit flag
- **MSAA 4x** — multisample anti-aliasing (WebGPU), canvas antialiasing (WebGL)
- **Primitive geometry** — box and sphere generators with normals
- **Z-up right-handed** coordinate system (Blender convention)
- **Zero-copy JS↔WASM** — typed array views directly into WASM linear memory

## API

### Scene

```ts
const scene = await Scene.create(canvas, { backend: 'webgpu' }) // or 'webgl', or omit for auto
scene.render() // run the full render pipeline
scene.resize(width, height) // resize canvas and GPU resources
await scene.switchBackend(newCanvas, 'webgl') // hot-swap renderer at runtime
scene.destroy() // release all GPU resources
scene.drawCalls // draw call count (last frame)
scene.visibleCount // visible entity count (last frame)
```

### Geometry

```ts
import { createBoxGeometry, createSphereGeometry } from 'voidcore'

const box = createBoxGeometry(1, 1, 1) // width, height, depth
const sphere = createSphereGeometry(0.5, 16, 12) // radius, wSegs, hSegs
const geoId = scene.registerGeometry(box) // returns geometry ID
```

Vertex format: `[px, py, pz, nx, ny, nz]` (position + normal, 24 bytes/vertex).

### Mesh

```ts
const mesh = scene.add(
  new Mesh({
    geometryId: geoId,
    position: [0, 0, 0], // default [0, 0, 0]
    rotation: [0, 0, 0], // Euler ZXY, default [0, 0, 0]
    scale: [1, 1, 1], // default [1, 1, 1]
    color: [1, 1, 1, 1], // RGBA, default [1, 1, 1, 1]
    visible: true,
    unlit: false,
  }),
)

// Direct mutation of WASM memory (zero-copy Float32Array views)
mesh.position[0] += 1
mesh.rotation[2] = Math.PI / 4
mesh.scale.set([2, 2, 2])
mesh.color.set([1, 0, 0, 1])
mesh.setDirty() // flag for world matrix recomputation

mesh.visible = false // exclude from rendering
mesh.unlit = true // skip lighting calculation
scene.remove(mesh) // remove from scene (swap-with-last)
```

### Camera

```ts
scene.camera.eye.set([0, -10, 5]) // camera position
scene.camera.target.set([0, 0, 0]) // look-at target
scene.camera.up.set([0, 0, 1]) // up vector (Z-up default)
scene.camera.fov = Math.PI / 4 // vertical field of view
scene.camera.near = 0.1
scene.camera.far = 1000
```

### Lighting

```ts
scene.setDirectionalLight([0.5, 0.3, -1], [1, 0.95, 0.9]) // direction, color
scene.setAmbientLight([0.15, 0.15, 0.2]) // color
```

## Architecture

```
Game Code (JS) --> Scene/Mesh/Camera (JS) --> Computation (C/WASM) --> GPU Rendering (JS)
                                                                        |-- WebGPU (primary)
                                                                        \-- WebGL2 (fallback)
```

- **C/WASM owns**: SoA entity storage, TRS-to-world-matrix transforms, frustum culling, draw sorting, all math (sin/cos via minimax polynomials, WASM SIMD)
- **JS owns**: GPU device/pipelines/buffers/draw calls, Scene/Mesh/Camera classes, geometry generation
- **Matrix format**: column-major mat4, Euler ZXY rotation order

### Render Pipeline (`scene.render()`)

1. `vc_compute_world_matrices` — TRS to mat4 for dirty entities
2. `camera.update` — view/projection/VP matrices via WASM
3. Update bounding spheres from mesh positions + scales
4. `vc_extract_frustum_planes` + `vc_frustum_cull` — bounding sphere culling
5. `vc_build_sort_keys` + `vc_sort_draw_calls` — radix sort by geometry ID
6. Build draw entity list from visible indices
7. `renderer.draw` — submit GPU draw calls
8. `vc_frame_reset` — reset per-frame arena allocator

### WASM Memory (32 MB fixed)

```
[__heap_base..~7.5MB]   SoA arrays (50k max entities)
[after SoA..~8MB]       Scratch area (camera matrices, frustum planes)
[16MB..32MB]            Per-frame arena (bump allocator, reset each frame)
```

## Development

```bash
brew install llvm binaryen        # required: Apple clang lacks wasm32 target
bun install
bun run build:wasm                # compile C to WASM
bun run dev                       # start dev server
```

### Commands

| Command                | Description                         |
| ---------------------- | ----------------------------------- |
| `bun run build:wasm`   | Compile C to `public/voidcore.wasm` |
| `bun run dev`          | Start Next.js dev server            |
| `bun run typecheck`    | Type-check with tsgo                |
| `bun run lint`         | OXLint                              |
| `bun run format:check` | OXFmt                               |
| `bun run all`          | lint + format + test + typecheck    |

## Coordinate System

Right-handed Z-up (Blender convention): +X right, +Y forward, +Z up.
