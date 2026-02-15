# VoidCore

High-performance WebGPU/WebGL game engine with a C/WASM compute core. Zero dependencies in the C layer — no Emscripten, no libc. JS handles GPU API calls and scene management, C/WASM handles math, transforms, culling, and sorting. Shared memory via typed array views — no serialization overhead.

## Quick Start

```ts
import { Scene, Mesh, createBoxGeometry, createSphereGeometry, loadGLTF } from 'voidcore'

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
- **Skeletal animation** — glTF skinned meshes with linear blend skinning (up to 128 joints), crossfade transitions, bone attachments
- **glTF 2.0 / GLB import** — load external 3D models with skins, animations, and optional Draco mesh compression
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

### glTF / GLB Import

```ts
import { loadGLTF } from 'voidcore'

const gltf = await loadGLTF('/models/helmet.glb')
for (const mesh of gltf.meshes) {
  for (const prim of mesh.primitives) {
    const geoId = scene.registerGeometry(prim.geometry)
    scene.add(new Mesh({ geometryId: geoId, color: prim.color }))
  }
}

// With custom Draco decoder path
const gltf2 = await loadGLTF('/models/compressed.glb', {
  dracoDecoderPath: '/draco-1.5.7/',
})
```

Supports glTF 2.0 JSON + external `.bin` buffers, GLB binary, and `KHR_draco_mesh_compression`. Draco decoder is lazy-loaded only when a compressed primitive is encountered. Material `baseColorFactor` is extracted from `pbrMetallicRoughness`. Flat normals are computed when the `NORMAL` attribute is missing.

The result also includes `skins`, `animations`, and `nodeTransforms` for skeletal animation (see Skinned Meshes below).

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

### Skinned Meshes

```ts
import {
  loadGLTF,
  createSkeleton,
  createSkinInstance,
  updateSkinInstance,
  transitionTo,
  findBoneNodeIndex,
} from 'voidcore'

// Load a glTF with skins and animations
const gltf = await loadGLTF('/models/character.glb')
const bodyMesh = gltf.meshes.find(m => m.skinIndex !== undefined)
const prim = bodyMesh.primitives[0]
const skin = gltf.skins[bodyMesh.skinIndex]

// Create skeleton and animation instance
const skeleton = createSkeleton(skin, gltf.nodeTransforms)
const inst = createSkinInstance(skeleton, 0) // start with clip 0

// Register skinned geometry (includes joint indices + weights)
const geoId = scene.registerSkinnedGeometry(prim.geometry, prim.skinJoints, prim.skinWeights)

// Add skinned mesh to scene
scene.add(new Mesh({ geometryId: geoId, skinInstance: inst }))

// Attach a weapon to a bone
const handIdx = findBoneNodeIndex(skeleton, 'Hand.R')
scene.add(
  new Mesh({
    geometryId: weaponGeoId,
    boneAttachment: { skinInstance: inst, boneNodeIndex: handIdx },
  }),
)

// In your render loop — advance animation and crossfade between clips
updateSkinInstance(inst, gltf.animations, dt)
transitionTo(inst, 1, 0.2) // crossfade to clip 1 over 200ms
```

Linear blend skinning with up to 128 joints per skeleton. Joint matrices are computed in JS (quaternion SLERP + hierarchical traversal) and uploaded per-entity. Crossfade blending interpolates between two animation clips.

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
- **JS owns**: GPU device/pipelines/buffers/draw calls, Scene/Mesh/Camera classes, geometry generation, glTF/GLB import
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
