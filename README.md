# VoidCore

A performant 3D graphics engine written in TypeScript with WebGPU and WebGL2 support.

![Voidcore logo](/logo.webp)

> [!IMPORTANT]
> This engine is developed specifically for [**Mana Blade**](https://manablade.com/) and is not meant to be used by other projects at the moment.

## Features

- **Dual rendering backends** – WebGPU (modern, fast) with automatic WebGL2 fallback
- **Scene graph** – Hierarchical node system with dirty-flag matrix propagation and transform setters (`setPosition`, `setPositionX/Y/Z`, `setRotation`, `setScale`, `setScaleX/Y/Z`)
- **Skeletal animation** – Clip-based animation with blending, crossfading, and loop modes
- **Procedural geometry** – Box, sphere, plane, cone, cylinder, capsule, circle
- **glTF/GLB loading** – Import 3D models with Draco compression support
- **KTX2 textures** – Load Basis Universal compressed textures (ETC1S/UASTC) via WASM transcoder, with automatic transcoding to GPU-native formats (ASTC, BC7, BC3, ETC2) based on device support
- **Material system** – Basic (unlit) and Lambert (diffuse) shading with palette support, opacity, transparency, color maps, AO maps, and per-material face culling (`side: 'front'`/`'back'`/`'double'`)
- **Mesh outlines** – Inverted hull outlines via `outline` option (thickness + color), works with static and skinned meshes
- **Sorted alpha blending** – Back-to-front transparent mesh rendering with premultiplied alpha (WebGPU)
- **Shadow maps** – Single shadow map with PCF 3×3 filtering, light-owned configuration, shadow baking for static scenes
- **Bloom post-processing** – Multi-level downsample/upsample with Karis average
- **Frustum culling** – AABB-based visibility culling with Gribb-Hartmann plane extraction
- **Distance culling** – Per-mesh `maxDistance` to hide meshes beyond a camera distance threshold
- **Raycasting** – BVH-accelerated ray-mesh intersection for mouse picking
- **DPR limiting** – Configurable max device pixel ratio (1.25 mobile / 1.5 desktop default)
- **Priority scheduler** – Single rAF loop with priority-ordered callbacks and FPS capping
- **HTML overlay** – DOM elements tracking 3D world positions with dirty checking, depth z-index, distance scaling
- **Orbit controls** – Mouse/touch camera controls with damping and inertia
- **Zero-allocation math** – Float32Array-backed vectors, matrices, and quaternions (configurable to Float16Array via `floatPrecision` option)
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
camera.setPosition(5, 5, 5)

const box = new Mesh(new BoxGeometry(), new LambertMaterial({ color: [0.8, 0.2, 0.3] }))
scene.add(box)

const light = new DirectionalLight({ intensity: 1.5, castShadow: true, shadowMapSize: 200 })
light.setPosition(5, 5, 10)
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

## React Bindings

An optional declarative layer is available:

```tsx
import { Canvas, Html, useFrame, useGLTF } from 'voidcore'

const RotatingBox = () => {
  const ref = useRef(null)

  useFrame(({ elapsed }) => {
    const mesh = ref.current
    if (!mesh) return
    quatFromAxisAngle(mesh.rotation, [0, 0, 1], elapsed)
    mesh.markTransformDirty()
  })

  return (
    <mesh ref={ref} position={[0, 0, 1]} castShadow>
      <boxGeometry />
      <lambertMaterial args={[{ color: [0.8, 0.2, 0.3] }]} />
      <Html center>
        <div style={{ color: '#fff', background: 'rgba(0,0,0,0.6)', padding: '4px 8px' }}>Hello</div>
      </Html>
    </mesh>
  )
}

const App = () => (
  <Canvas
    shadows
    antialias
    camera={{ fov: 55, position: [0, -10, 5] }}
    ambientLight={{ color: [0.5, 0.5, 0.6], intensity: 0.4 }}
  >
    <directionalLight args={[{ intensity: 1.2 }]} position={[5, 5, 10]} castShadow shadowMapSize={200} />
    <RotatingBox />
  </Canvas>
)
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
| `overlay.ts`   | HTML overlay (DOM elements tracking 3D positions)    |
| `controls/`    | Camera interaction (orbit controls)                  |
| `loaders/`     | Asset importers (glTF/GLB, KTX2)                     |

Every engine source file includes educational comments at the top explaining the high-level concepts for developers who may not be familiar with graphics programming.
