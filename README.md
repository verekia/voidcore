# VoidCore

A performant 3D graphics engine written in TypeScript with WebGPU and WebGL2 support.

## Features

- **Dual rendering backends** – WebGPU (modern, fast) with automatic WebGL2 fallback
- **Scene graph** – Hierarchical node system with dirty-flag matrix propagation
- **Skeletal animation** – Clip-based animation with blending, crossfading, and loop modes
- **Procedural geometry** – Box, sphere, plane, cone, cylinder, capsule, circle
- **glTF/GLB loading** – Import 3D models with Draco compression support
- **Material system** – Basic (unlit) and Lambert (diffuse) shading with palette support
- **Cascaded shadow maps** – 3-cascade CSM with PCF filtering and cascade blending
- **Bloom post-processing** – Multi-level downsample/upsample with Karis average
- **Frustum culling** – AABB-based visibility culling with Gribb-Hartmann plane extraction
- **Raycasting** – BVH-accelerated ray-mesh intersection for mouse picking
- **DPR limiting** – Configurable max device pixel ratio (1.25 mobile / 1.5 desktop default)
- **Priority scheduler** – Single rAF loop with priority-ordered callbacks and FPS capping
- **Orbit controls** – Mouse/touch camera controls with damping and inertia
- **Zero-allocation math** – Float32Array-backed vectors, matrices, and quaternions
- **Z-up coordinate system** – Right-handed, Z-up convention throughout

## Quick Start

```ts
import {
  Engine,
  Scene,
  PerspectiveCamera,
  Mesh,
  BoxGeometry,
  LambertMaterial,
  DirectionalLight,
  OrbitControls,
} from 'voidcore'

const canvas = document.querySelector('canvas')!

const engine = await Engine.create(canvas)
const scene = new Scene()
const camera = new PerspectiveCamera({ fov: 60 })
camera.position[0] = 5
camera.position[1] = 5
camera.position[2] = 5

const box = new Mesh(new BoxGeometry(), new LambertMaterial({ color: [0.8, 0.2, 0.3] }))
scene.add(box)

const light = new DirectionalLight({ intensity: 1.5 })
light.position[0] = 5
light.position[1] = 5
light.position[2] = 10
scene.add(light)

const controls = new OrbitControls(camera, canvas)

engine.register(
  ({ dt }) => {
    controls.update(dt)
  },
  { priority: -1 },
)

engine.register(
  () => {
    engine.render(scene, camera)
  },
  { priority: 0 },
)

engine.maxFps = 60
engine.maxDpr = 1.5 // Cap resolution scaling (default: 1.25 mobile, 1.5 desktop, false to disable)
engine.start()
```

## Development

```bash
bun install
bun run dev        # Start dev server
bun run all        # Lint + format check + test + typecheck
```

## Architecture

The engine is organized into focused modules:

| Module         | Description                                          |
| -------------- | ---------------------------------------------------- |
| `engine.ts`    | Entry point, owns the scheduler and renderer         |
| `scheduler.ts` | Priority-based rAF loop with FPS throttling          |
| `scene/`       | Scene graph nodes (Node, Mesh, Group, Camera, Light) |
| `geometry/`    | Vertex data and procedural shape generators          |
| `materials/`   | Surface appearance definitions                       |
| `renderer/`    | WebGPU and WebGL2 backends with shaders              |
| `math/`        | Linear algebra primitives (vec3, mat4, quat, AABB)   |
| `animation/`   | Skeletal animation system                            |
| `raycasting/`  | BVH-accelerated ray intersection                     |
| `controls/`    | Camera interaction (orbit controls)                  |
| `loaders/`     | Asset importers (glTF/GLB)                           |

Every engine source file includes educational comments at the top explaining the high-level concepts for developers who may not be familiar with graphics programming.
