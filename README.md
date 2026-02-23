# voidcore

A minimal purpose-built 3D graphics engine written in TypeScript with WebGPU and WebGL2 support. The goal of this engine is to replace Three.js and React Three Fiber (love them both!) with a minimal subset of features that my game needs, and with better performance.

<p align="center">
  <img src="https://raw.githubusercontent.com/verekia/voidcore/refs/heads/main/logo.webp" width="160">
</p>

> [!IMPORTANT]
> This engine is developed specifically for [**Mana Blade**](https://manablade.com/). It is not meant to be used by other projects at the moment and its API can change at any time.

The [demo](https://engine.v1v2.io/) renders around 250 characters, positioned by 250 raycasts per frame, all independently animated, with no instancing. It has fairly good performance even on older phones. In practice I have at most 50 characters rendered at the same time in the game so this is good enough.

## Features

- **Dual rendering backends** – WebGPU (modern, fast) with automatic WebGL2 fallback
- **Scene graph** – Hierarchical node system with dirty-flag matrix propagation and transform setters (`setPosition`, `setPositionX/Y/Z`, `setRotation`, `setScale`, `setScaleX/Y/Z`)
- **Skeletal animation** – Clip-based animation with blending, crossfading, and loop modes
- **Procedural geometry** – Box, sphere, plane, cone, cylinder, capsule, circle
- **glTF/GLB loading** – Import 3D models with Draco compression support
- **KTX2 textures** – Load Basis Universal compressed textures (ETC1S/UASTC) via WASM transcoder, with automatic transcoding to GPU-native formats (ASTC, BC7, BC3, ETC2) based on device support
- **Material system** – Basic and Lambert shading with vertex colors, bloom, transparency, color and AO maps, per-material tiled normal maps
- **Mesh outlines** – Shader-based inverted hull outlines via `outline` option (thickness + color)
- **Sorted alpha blending** – Back-to-front transparent mesh rendering with premultiplied alpha (WebGPU)
- **Shadow maps** – Single shadow map with PCF 3×3 filtering, shadow baking for static scenes
- **Bloom post-processing** – Multi-level downsample/upsample with Karis average
- **Frustum culling** – AABB-based visibility culling with Gribb-Hartmann plane extraction
- **Distance culling** – Per-mesh `maxDistance` to hide meshes beyond a camera distance threshold
- **Raycasting** – BVH-accelerated raycasts
- **DPR limiting** – Configurable max device pixel ratio (1.25 mobile / 1.5 desktop default)
- **Priority scheduler** – Single rAF loop with priority-ordered callbacks and FPS capping
- **HTML overlay** – DOM elements tracking 3D world positions with depth z-index and distance scaling
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

## API Reference

### Engine

```ts
Engine.create(canvas: HTMLCanvasElement, config?: EngineConfig): Promise<Engine>
```

| Property / Method              | Description                                                        |
| ------------------------------ | ------------------------------------------------------------------ |
| `canvas`                       | The `HTMLCanvasElement`                                            |
| `renderer`                     | The active `Renderer` instance                                     |
| `backend`                      | `'webgpu'` or `'webgl2'`                                           |
| `scheduler`                    | The underlying `Scheduler`                                         |
| `maxFps`                       | Global FPS cap (0 = uncapped)                                      |
| `maxDpr`                       | Max device pixel ratio                                             |
| `compressedTextureFormats`     | GPU-compressed formats supported by the device (for KTX2)          |
| `shadowsBaked`                 | When true, the shadow map is frozen and not re-rendered each frame |
| `register(callback, options?)` | Register a per-frame callback; returns an unsubscribe function     |
| `start()`                      | Begin the scheduler loop                                           |
| `stop()`                       | Pause the scheduler loop                                           |
| `render(scene, camera)`        | Render a single frame (call inside a registered callback)          |
| `getStats(): FrameStats`       | Return current frame statistics                                    |
| `dispose()`                    | Clean up the scheduler and GPU resources                           |

```ts
interface EngineConfig {
  backend?: 'auto' | 'webgpu' | 'webgl2'
  antialias?: boolean
  bloom?: boolean | { intensity?: number; levels?: number }
  shadows?: boolean | ShadowConfig
  maxDpr?: number | false
}

interface ShadowConfig {
  enabled?: boolean
  resolution?: number
}

interface FrameStats {
  fps: number
  frameTime: number
  drawCalls: number
  shadowDrawCalls: number
  triangles: number
  visibleObjects: number
  culledObjects: number
}
```

### Scheduler

```ts
new Scheduler()
```

| Property / Method                          | Description                                     |
| ------------------------------------------ | ----------------------------------------------- |
| `maxFps`                                   | Global FPS cap (0 = uncapped)                   |
| `register(callback, options?): () => void` | Add a callback; returns an unsubscribe function |
| `start()`                                  | Begin the rAF loop                              |
| `stop()`                                   | Pause the rAF loop                              |
| `destroy()`                                | Stop and clear all callbacks                    |

```ts
interface SchedulerState {
  dt: number // delta time in seconds (capped at 0.1)
  elapsed: number // total elapsed seconds since start
  frame: number // frame counter
}

interface SchedulerCallbackOptions {
  priority?: number // execution order; lower runs first (default: 0, may be negative)
  fps?: number // per-callback FPS throttle (0 = every frame)
}
```

### Scene Graph

#### Node

Base class for all scene objects.

```ts
new Node()
```

| Property        | Type                                                                    | Description                                 |
| --------------- | ----------------------------------------------------------------------- | ------------------------------------------- |
| `name`          | `string`                                                                |                                             |
| `type`          | `'group' \| 'mesh' \| 'camera' \| 'directionalLight' \| 'ambientLight'` |                                             |
| `position`      | `Vec3`                                                                  | Local position (`Float32Array[3]`)          |
| `rotation`      | `Quat`                                                                  | Local rotation as quaternion `[x, y, z, w]` |
| `scale`         | `Vec3`                                                                  | Local scale                                 |
| `parent`        | `Node \| null`                                                          |                                             |
| `children`      | `Node[]`                                                                |                                             |
| `visible`       | `boolean`                                                               |                                             |
| `frustumCulled` | `boolean`                                                               |                                             |
| `castShadow`    | `boolean`                                                               |                                             |
| `receiveShadow` | `boolean`                                                               |                                             |
| `_localMatrix`  | `Mat4`                                                                  | Computed local transform matrix             |
| `_worldMatrix`  | `Mat4`                                                                  | Computed world transform matrix             |

| Method                                                    | Description                                                    |
| --------------------------------------------------------- | -------------------------------------------------------------- |
| `add(...nodes)`                                           | Attach child nodes (reparents if already attached)             |
| `remove(child)`                                           | Detach a child node                                            |
| `traverse(callback)`                                      | Walk the subtree depth-first                                   |
| `lookAt(target)`                                          | Orient to face a target point (Z-up convention)                |
| `setPosition(x, y, z)`                                    |                                                                |
| `setPositionX(v)` / `setPositionY(v)` / `setPositionZ(v)` |                                                                |
| `setRotation(x, y, z, w)`                                 | Set quaternion rotation                                        |
| `setScale(s)` / `setScale(x, y, z)`                       | Uniform or per-axis scale                                      |
| `setScaleX(v)` / `setScaleY(v)` / `setScaleZ(v)`          |                                                                |
| `markTransformDirty()`                                    | Mark for recalculation after writing directly to Float32Arrays |

Module-level: `updateWorldMatrices(node, parentDirty?)` — recursively recompute dirty world matrices top-down.

#### Scene

Extends `Node`. Root of the scene graph.

```ts
new Scene()
```

| Method                                       | Description                        |
| -------------------------------------------- | ---------------------------------- |
| `getByName(name: string): Node \| undefined` | O(1) lookup by name                |
| `updateGraph(): void`                        | Recompute all dirty world matrices |

#### Mesh

Extends `Node`. A renderable object.

```ts
new Mesh(geometry?: Geometry, material?: Material)
```

| Property      | Type                                 | Description                                           |
| ------------- | ------------------------------------ | ----------------------------------------------------- |
| `geometry`    | `Geometry`                           |                                                       |
| `material`    | `Material`                           |                                                       |
| `skeleton`    | `Skeleton \| undefined`              | For skinned meshes                                    |
| `outline`     | `MeshOutline \| number \| undefined` | Inverted hull outline                                 |
| `maxDistance` | `number`                             | Hide when camera exceeds this distance (0 = disabled) |

```ts
interface MeshOutline {
  thickness: number
  color?: [number, number, number] // default [0, 0, 0]
}
```

#### Sprite

Extends `Mesh`. A billboard plane that always faces the camera. Uses a shared 1×1 `PlaneGeometry` and defaults to `SpriteMaterial`. Does not cast shadows by default.

```ts
new Sprite(material?: SpriteMaterial)
```

#### Group

Extends `Node`. Empty container for grouping scene objects.

```ts
new Group()
```

#### PerspectiveCamera

Extends `Node`. Perspective projection camera.

```ts
new PerspectiveCamera(opts?: CameraOptions)

interface CameraOptions {
  fov?: number  // field of view in degrees (default: 60)
  near?: number // near clip plane (default: 0.1)
  far?: number  // far clip plane (default: 1000)
}
```

| Property                | Type     |
| ----------------------- | -------- |
| `fov`                   | `number` |
| `near`                  | `number` |
| `far`                   | `number` |
| `aspect`                | `number` |
| `_projectionMatrix`     | `Mat4`   |
| `_viewMatrix`           | `Mat4`   |
| `_viewProjectionMatrix` | `Mat4`   |

#### DirectionalLight

Extends `Node`. Parallel-ray light source with optional shadow casting.

```ts
new DirectionalLight(opts?: DirectionalLightOptions)

interface DirectionalLightOptions {
  color?: [number, number, number] // default [1, 1, 1]
  intensity?: number               // default 1
  castShadow?: boolean             // default false
  shadowMapSize?: number           // ortho box size (default: 200)
  shadowNear?: number              // default 1
  shadowFar?: number               // default 300
  shadowBias?: number              // default 0.001
  shadowSlopeBias?: number         // default 0.005
}
```

#### AmbientLight

Extends `Node`. Constant ambient illumination.

```ts
new AmbientLight(opts?: AmbientLightOptions)

interface AmbientLightOptions {
  color?: [number, number, number] // default [1, 1, 1]
  intensity?: number               // default 1
}
```

#### Scene Cloning

```ts
cloneScene(source: Node, skeletons?: Skeleton[], options?: CloneOptions): CloneResult

interface CloneOptions {
  meshFilter?: (mesh: Mesh) => boolean
}

interface CloneResult {
  root: Node
  skeletons: Skeleton[]
  nodeMap: Map<Node, Node>
}
```

### Geometry

#### Geometry

```ts
new Geometry(data: GeometryData)

interface GeometryData {
  positions: Float32Array
  normals: Float32Array
  indices: Uint16Array | Uint32Array
  uvs?: Float32Array            // 2 floats per vertex
  colors?: Float32Array         // unorm8x4 per vertex (baked palette)
  emissiveColors?: Float32Array // float16x4 HDR per vertex (baked palette)
  materialIndices?: Uint8Array  // per-vertex palette index
  joints?: Uint8Array | Uint16Array
  weights?: Float32Array
}
```

| Property          | Type                                     | Description                             |
| ----------------- | ---------------------------------------- | --------------------------------------- |
| `positions`       | `Float32Array`                           | 3 floats per vertex                     |
| `normals`         | `Float32Array`                           | 3 floats per vertex                     |
| `indices`         | `Uint16Array \| Uint32Array`             |                                         |
| `uvs`             | `Float32Array \| undefined`              | 2 floats per vertex                     |
| `colors`          | `Float32Array \| undefined`              | Per-vertex colors (unorm8x4)            |
| `emissiveColors`  | `Float32Array \| undefined`              | Per-vertex HDR emissive (float16x4)     |
| `materialIndices` | `Uint8Array \| undefined`                | Per-vertex palette index                |
| `joints`          | `Uint8Array \| Uint16Array \| undefined` | Bone indices for skinning               |
| `weights`         | `Float32Array \| undefined`              | Bone weights for skinning               |
| `aabb`            | `AABB`                                   | Axis-aligned bounding box               |
| `needsUpdate`     | `boolean`                                | Set to re-upload vertex data to the GPU |
| `dispose()`       |                                          | Release GPU buffers                     |

#### Procedural Primitives

All extend `Geometry`, Z-up, centered at origin.

| Class                         | Options (all optional, defaults in parentheses)                                                |
| ----------------------------- | ---------------------------------------------------------------------------------------------- |
| `new BoxGeometry(opts?)`      | `width` (1), `height` (1), `depth` (1)                                                         |
| `new SphereGeometry(opts?)`   | `radius` (1), `widthSegments` (32), `heightSegments` (16)                                      |
| `new PlaneGeometry(opts?)`    | `width` (1), `height` (1), `widthSegments` (1), `heightSegments` (1)                           |
| `new ConeGeometry(opts?)`     | `radius` (1), `height` (1), `radialSegments` (32)                                              |
| `new CylinderGeometry(opts?)` | `radiusTop` (1), `radiusBottom` (1), `height` (1), `radialSegments` (32), `heightSegments` (1) |
| `new CapsuleGeometry(opts?)`  | `radius` (1), `height` (1), `radialSegments` (16), `heightSegments` (1)                        |
| `new CircleGeometry(opts?)`   | `radius` (1), `segments` (32)                                                                  |

#### Geometry Utilities

```ts
// Bake palette into per-vertex colors/emissiveColors. Cached by geometry+palette reference.
bakePalette(geometry: Geometry, palette: PaletteEntry[]): Geometry

// Clear the bakePalette cache (all geometries, or a specific one).
clearColoredGeometryCache(geometry?: Geometry): void

mergeGeometries(geometries: Geometry[]): Geometry
mergeStaticIntoSkinned(skinned: Geometry, static: Geometry, boneIndex: number): Geometry
computeSmoothNormals(geometry: Geometry): void // position-averaged normals for outline meshes

interface PaletteEntry {
  color: [number, number, number]
  emissive?: [number, number, number]
  emissiveIntensity?: number
  tiledAo?: Texture              // per-material tiled AO texture (world-space XY repeat)
  tiledAoIntensity?: number      // default 1.0, supports HDR values
  tiledAoScale?: number          // default 1.0, world-space tiling frequency
  tiledNormal?: Texture          // per-material tiled normal map (world-space XY repeat)
  tiledNormalIntensity?: number  // default 1.0
  tiledNormalScale?: number      // default 1.0, world-space tiling frequency
}
```

### Materials

#### BasicMaterial / LambertMaterial

`BasicMaterial` is unlit (ignores lights). `LambertMaterial` is diffuse-shaded.

```ts
new BasicMaterial(opts?: MaterialOptions)
new LambertMaterial(opts?: MaterialOptions)

interface MaterialOptions {
  color?: [number, number, number]    // default [1, 1, 1]
  vertexColors?: boolean              // use baked per-vertex colors from geometry
  receiveShadow?: boolean             // default true (LambertMaterial only)
  palette?: PaletteEntry[]
  emissiveBrightness?: number         // 0–1; neon glow desaturation toward white (default: 1)
  opacity?: number                    // default 1
  transparent?: boolean               // enable sorted alpha blending
  side?: 'front' | 'back' | 'double' // face culling (default: 'front')
  colorMap?: Texture                  // diffuse/albedo texture
  aoMap?: Texture                     // ambient occlusion texture (red channel)
  aoIntensity?: number                // AO influence (default: 1)
  customShader?: CustomShader         // inject custom shader code snippets
}

interface CustomShader {
  vertexWGSL?: string    // WGSL code injected into vertex shader
  fragmentWGSL?: string  // WGSL code injected into fragment shader
  vertexGLSL?: string    // GLSL code injected into vertex shader
  fragmentGLSL?: string  // GLSL code injected into fragment shader
}
```

| Property             | Type                            |
| -------------------- | ------------------------------- |
| `color`              | `[number, number, number]`      |
| `vertexColors`       | `boolean`                       |
| `opacity`            | `number`                        |
| `transparent`        | `boolean`                       |
| `side`               | `'front' \| 'back' \| 'double'` |
| `receiveShadow`      | `boolean`                       |
| `emissiveBrightness` | `number`                        |
| `colorMap`           | `Texture \| undefined`          |
| `aoMap`              | `Texture \| undefined`          |
| `aoIntensity`        | `number`                        |
| `customShader`       | `CustomShader \| undefined`     |
| `needsUpdate`        | `boolean`                       |

**Custom Shader Hook Variables**

| Stage    | Variable                      | Type (WGSL / GLSL)   | Description                           |
| -------- | ----------------------------- | -------------------- | ------------------------------------- |
| Vertex   | `out.worldPos` / `v_worldPos` | `vec3<f32>` / `vec3` | World-space position (read/write)     |
| Vertex   | `out.normal` / `v_normal`     | `vec3<f32>` / `vec3` | World-space normal (read/write)       |
| Vertex   | `out.uv` / `v_uv`             | `vec2<f32>` / `vec2` | UV coordinates (read/write)           |
| Fragment | `finalColor`                  | `vec3<f32>` / `vec3` | Output color before alpha premultiply |
| Fragment | `alpha`                       | `f32` / `float`      | Output alpha                          |

#### SpriteMaterial

Unlit material with defaults tuned for sprites (transparent, double-sided).

```ts
new SpriteMaterial(opts?: SpriteMaterialOptions)

interface SpriteMaterialOptions extends MaterialOptions {
  rotation?: number         // 2D rotation in radians (default: 0)
  sizeAttenuation?: boolean // shrink with distance (default: true)
}
```

| Property          | Type      | Description                                   |
| ----------------- | --------- | --------------------------------------------- |
| `rotation`        | `number`  | 2D rotation around the view axis (radians)    |
| `sizeAttenuation` | `boolean` | When false, sprite keeps constant screen size |

Inherits all `MaterialOptions` properties. `transparent` defaults to `true`, `side` defaults to `'double'`.

#### Texture

```ts
new Texture(data: TextureData)

interface TextureData {
  width: number
  height: number
  data: Uint8Array
  format?: TextureFormat // default 'rgba8'
}

type TextureFormat = 'rgba8' | CompressedTextureFormat
type CompressedTextureFormat = 'astc-4x4' | 'bc7' | 'bc3' | 'etc2-rgba8'
```

### Animation

#### AnimationMixer

Plays and blends skeletal animations on a `Skeleton`.

```ts
new AnimationMixer(skeleton: Skeleton)
```

| Method                                             | Description                                                     |
| -------------------------------------------------- | --------------------------------------------------------------- |
| `clipAction(clip: AnimationClip): AnimationAction` | Create an action for the given clip                             |
| `update(dt: number): void`                         | Advance all playing actions and apply the blended pose to bones |

#### AnimationAction

Returned by `mixer.clipAction(clip)`.

| Property    | Type                               | Description                      |
| ----------- | ---------------------------------- | -------------------------------- |
| `clip`      | `AnimationClip`                    |                                  |
| `loop`      | `'repeat' \| 'once' \| 'pingpong'` | default `'repeat'`               |
| `timeScale` | `number`                           | default 1                        |
| `weight`    | `number`                           | blend weight (0–1)               |
| `time`      | `number`                           | current playback time in seconds |
| `paused`    | `boolean`                          |                                  |
| `playing`   | `boolean`                          |                                  |

| Method                          | Returns | Description                          |
| ------------------------------- | ------- | ------------------------------------ |
| `play()`                        | `this`  | Start playback from time 0           |
| `stop()`                        | `this`  | Stop and reset                       |
| `fadeIn(duration)`              | `this`  | Fade weight 0 → 1 over duration      |
| `fadeOut(duration)`             | `this`  | Fade weight to 0, then stop          |
| `crossFadeTo(target, duration)` | `this`  | Fade this out while fading target in |

#### Skeleton

```ts
new Skeleton(bones: Node[], inverseBindMatrices: Mat4[])
```

| Property       | Type           |
| -------------- | -------------- |
| `bones`        | `Node[]`       |
| `boneMatrices` | `Float32Array` |

| Method                                     | Description             |
| ------------------------------------------ | ----------------------- |
| `getBone(name: string): Node \| undefined` | Look up a bone by name  |
| `update(): void`                           | Recompute bone matrices |

#### AnimationClip / KeyframeTrack

```ts
interface AnimationClip {
  name: string
  duration: number
  tracks: KeyframeTrack[]
}

interface KeyframeTrack {
  boneIndex: number
  path: 'translation' | 'rotation' | 'scale'
  times: Float32Array
  values: Float32Array
  interpolation: 'LINEAR' | 'STEP'
}
```

### Controls

#### OrbitControls

Mouse and touch camera orbit around a target point with damping and inertia.

```ts
new OrbitControls(camera: PerspectiveCamera, canvas: HTMLCanvasElement, opts?: OrbitControlsOptions)

interface OrbitControlsOptions {
  target?: [number, number, number]
  dampingFactor?: number  // default 0.1
  minDistance?: number    // default 0.1
  maxDistance?: number    // default 1000
  minElevation?: number   // radians (default: -π/2 + 0.01)
  maxElevation?: number   // radians (default: π/2 - 0.01)
  enabled?: boolean       // default true
}
```

| Property        | Type      |
| --------------- | --------- |
| `target`        | `Vec3`    |
| `enabled`       | `boolean` |
| `dampingFactor` | `number`  |
| `azimuth`       | `number`  |
| `elevation`     | `number`  |
| `distance`      | `number`  |

| Method                                 | Description                                 |
| -------------------------------------- | ------------------------------------------- |
| `update(dt: number): void`             | Apply damping and recompute camera position |
| `onChange(callback: () => void): void` | Register a change listener                  |
| `dispose(): void`                      | Remove all event listeners                  |

### Raycasting

```ts
new Raycaster()
```

| Method                                                    | Description                                          |
| --------------------------------------------------------- | ---------------------------------------------------- |
| `set(origin, direction)`                                  | Set ray from world-space origin and direction        |
| `setFromCamera(coords: { x: number; y: number }, camera)` | Build ray from NDC screen coordinates (−1 to 1)      |
| `intersectObject(object, recursive?): RaycastHit[]`       | Raycast against a mesh (and optionally its children) |
| `intersectObjects(objects, recursive?): RaycastHit[]`     | Raycast against multiple meshes                      |

Results are sorted by ascending distance.

```ts
interface RaycastHit {
  distance: number // world-space distance from ray origin
  point: Vec3 // intersection point in world space
  normal: Vec3 // interpolated surface normal (world space)
  uv: Vec2 | null // interpolated UV (if geometry has UVs)
  triangleIndex: number
  object: Mesh
}
```

Module-level BVH helpers:

```ts
buildMeshBVH(mesh: Mesh): void       // build BVH acceleration structure (cached internally)
prebuildBVH(geometry: Geometry): void // pre-build to avoid a stall on the first raycast
```

### Loaders

```ts
loadGLTF(url: string, options?: LoadOptions): Promise<GLTFResult>

interface LoadOptions {
  draco?: { decoderPath: string }
  ktx2?: { transcoderPath: string }
}

interface GLTFResult {
  scene: Group
  scenes: Group[]
  meshes: Mesh[]
  skeletons: Skeleton[]
  animations: AnimationClip[]
  dispose: () => void
}
```

```ts
loadKTX2(
  url: string,
  transcoderPath: string,
  supportedFormats?: readonly CompressedTextureFormat[],
): Promise<Texture>
```

Transcodes KTX2/Basis Universal textures to the best supported GPU-native format (ASTC 4×4 › BC7 › ETC2 › BC3 › RGBA8 fallback).

### Overlay

```ts
createOverlayManager(canvas: HTMLCanvasElement): OverlayManager

interface OverlayOptions {
  element: HTMLElement
  position?: Vec3 | [number, number, number] // world position
  node?: Node                                // track a node's world position
  offset?: Vec3 | [number, number, number]
  center?: boolean                           // CSS translate(-50%, -50%)
  distanceScale?: boolean                    // scale element by camera distance
  pointerEvents?: boolean                    // enable pointer-events on element
}
```

| OverlayManager method                                 | Description                         |
| ----------------------------------------------------- | ----------------------------------- |
| `add(opts: OverlayOptions): OverlayHandle`            | Begin tracking a DOM element        |
| `remove(handle: OverlayHandle): void`                 | Stop tracking                       |
| `update(camera, width, height, scene?, frame?): void` | Call each frame to update positions |
| `dispose(): void`                                     | Remove all tracked elements         |

### Helpers

#### DirectionalLightHelper

Visualizes the shadow frustum of a `DirectionalLight`.

```ts
new DirectionalLightHelper(opts?: { color?: [number, number, number]; opacity?: number })
```

| Member                                  | Description                                       |
| --------------------------------------- | ------------------------------------------------- |
| `mesh: Mesh`                            | Add to the scene to display the frustum wireframe |
| `update(light: DirectionalLight): void` | Sync with the light each frame                    |
| `dispose(): void`                       | Release geometry                                  |

### Math

All types are `Float32Array` views. All functions use a "write into output" pattern for zero allocation.

```ts
type Vec2 = Float32Array // [x, y]
type Vec3 = Float32Array // [x, y, z]
type Vec4 = Float32Array // [x, y, z, w]
type Quat = Float32Array // [x, y, z, w]
type Mat4 = Float32Array // 16 elements, column-major
type AABB = Float32Array // [minX, minY, minZ, maxX, maxY, maxZ]
```

**Constants**: `VEC3_ZERO`, `VEC3_ONE`, `VEC3_UP`, `VEC3_FORWARD`, `VEC3_RIGHT`, `QUAT_IDENTITY`, `MAT4_IDENTITY`

**Vec3**

```ts
vec3Create(): Vec3
vec3Set(out, x, y, z): Vec3
vec3Copy(out, a): Vec3
vec3Add(out, a, b): Vec3
vec3Sub(out, a, b): Vec3
vec3Scale(out, a, scalar): Vec3
vec3Normalize(out, a): Vec3
vec3Cross(out, a, b): Vec3
vec3Dot(a, b): number
vec3Length(a): number
vec3Lerp(out, a, b, t): Vec3
vec3TransformMat4(out, a, m): Vec3
vec3TransformQuat(out, a, q): Vec3
```

**Vec4**

```ts
vec4Create(): Vec4
vec4Set(out, x, y, z, w): Vec4
```

**Mat4**

```ts
mat4Create(): Mat4
mat4Identity(out): Mat4
mat4Multiply(out, a, b): Mat4
mat4Invert(out, a): Mat4 | null
mat4Compose(out, position, rotation, scale): Mat4
mat4Perspective(out, fovY, aspect, near, far, depthRange): Mat4
mat4LookAt(out, eye, target, up?): Mat4
```

**Quat**

```ts
quatCreate(): Quat
quatNormalize(out, a): Quat
quatFromAxisAngle(out, axis, angleRadians): Quat
quatSlerp(out, a, b, t): Quat
```

### React

An optional declarative layer built with a custom `react-reconciler`.

#### Canvas

Root component. Creates the engine, scene, and camera on mount.

```tsx
<Canvas
  backend="auto" // 'auto' | 'webgpu' | 'webgl2'
  antialias={true}
  shadows={true} // or { resolution: 1024 }
  bloom={{ intensity: 1, levels: 5 }}
  maxFps={60}
  maxDpr={1.5}
  camera={{ fov: 60, near: 0.1, far: 1000, position: [0, -10, 5] }}
  onCreated={({ engine, scene, camera }) => {}}
  style={{}}
  className=""
>
  {/* scene graph */}
</Canvas>
```

#### JSX Elements

Scene objects are declared as lowercase JSX elements.

Node elements (`<mesh>`, `<group>`, `<directionalLight>`, `<ambientLight>`) accept these props in addition to element-specific ones:

| Prop                                                                                 | Type                                 |
| ------------------------------------------------------------------------------------ | ------------------------------------ |
| `position`                                                                           | `[number, number, number]`           |
| `rotation`                                                                           | `[number, number, number, number]`   |
| `scale`                                                                              | `[number, number, number] \| number` |
| `visible`                                                                            | `boolean`                            |
| `castShadow`                                                                         | `boolean`                            |
| `receiveShadow`                                                                      | `boolean`                            |
| `name`                                                                               | `string`                             |
| `ref`                                                                                | `Ref`                                |
| `onClick`                                                                            | `(event) => void`                    |
| `onPointerOver` / `onPointerOut` / `onPointerDown` / `onPointerUp` / `onPointerMove` | `(event) => void`                    |

Geometry and material elements are attached as children of `<mesh>` (the reconciler sets them on `mesh.geometry` / `mesh.material` automatically). Constructor options are passed via `args={[opts]}`.

| Element                    | Maps to                                           |
| -------------------------- | ------------------------------------------------- |
| `<mesh>`                   | `Mesh`                                            |
| `<sprite>`                 | `Sprite`                                          |
| `<group>`                  | `Group`                                           |
| `<directionalLight>`       | `DirectionalLight`                                |
| `<ambientLight>`           | `AmbientLight`                                    |
| `<boxGeometry>`            | `BoxGeometry`                                     |
| `<sphereGeometry>`         | `SphereGeometry`                                  |
| `<planeGeometry>`          | `PlaneGeometry`                                   |
| `<coneGeometry>`           | `ConeGeometry`                                    |
| `<cylinderGeometry>`       | `CylinderGeometry`                                |
| `<capsuleGeometry>`        | `CapsuleGeometry`                                 |
| `<circleGeometry>`         | `CircleGeometry`                                  |
| `<basicMaterial>`          | `BasicMaterial`                                   |
| `<lambertMaterial>`        | `LambertMaterial`                                 |
| `<spriteMaterial>`         | `SpriteMaterial`                                  |
| `<primitive object={...}>` | Insert any pre-built engine object into the scene |

#### Hooks

```ts
useEngine(): { engine: Engine; scene: Scene; camera: PerspectiveCamera; canvas: HTMLCanvasElement }
```

```ts
useFrame(callback: (state: FrameState) => void): void

interface FrameState {
  dt: number
  elapsed: number
  frame: number
  engine: Engine
  scene: Scene
  camera: PerspectiveCamera
}
```

```ts
// Suspense-compatible asset loader. Caches by URL.
useLoader<T>(loaderFn: (url: string, ...args) => Promise<T>, url: string, ...args): T
```

```ts
useGLTF(url: string, options?: UseGLTFOptions): GLTFResult
useGLTF(url: string, options: { meshName: string }): Mesh
useGLTF(url: string, options: { meshName: string; clone: true }): ClonedMesh

useGLTF.setDecoderPath(path: string): void // set global Draco + KTX2 decoder path
useGLTF.preload(url: string, options?: LoadOptions): void

interface ClonedMesh {
  root: Node
  mesh: Mesh
  skeleton: Skeleton
  nodeMap: Map<Node, Node>
  animations: AnimationClip[]
}
```

```ts
useKTX2(url: string, transcoderPath?: string): Texture
useKTX2.setTranscoderPath(path: string): void
```

```ts
// Memoized wrapper around bakePalette.
useColoredGeometry(geometry: Geometry, palette: PaletteEntry[]): Geometry
```

```ts
// Load a named mesh from the static bundle GLB and bake a palette onto its geometry.
useColoredStaticGeometry(meshName: string, palette: PaletteEntry[]): Geometry
useColoredStaticGeometry.setStaticBundlePath(path: string): void // set global bundle GLB path
```

```ts
// Creates an AnimationMixer and returns named actions. Calls mixer.update(dt) each frame.
useAnimations(
  animations: AnimationClip[],
  skeleton: Skeleton,
): { mixer: AnimationMixer; actions: Record<string, AnimationAction> }
```

#### Html

Render DOM content anchored to a 3D position. Inherits parent transforms.

```tsx
<Html
  position={[0, 0, 1]} // optional offset from parent node
  center={true} // CSS translate(-50%, -50%)
  style={{}}
  className=""
>
  <div>Hello</div>
</Html>
```

#### BakeShadows

Mount to freeze the shadow map for static scenes; unmount to resume real-time shadow rendering.

```tsx
<BakeShadows />
```
