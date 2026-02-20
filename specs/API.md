# Voidcore — API

## Code Style

- **TypeScript** strict mode throughout
- **No semicolons**
- **`const` arrow functions** preferred over `function` declarations
- **Options objects** for all configuration (no positional parameters beyond the essentials)
- **`async`/`await`** for all loading operations
- **`camelCase`** for functions and variables
- **`PascalCase`** for types and interfaces
- **`SCREAMING_SNAKE`** for constants

```typescript
const createMesh = (geometry: Geometry, material: Material): Mesh => {
  // ...
}
```

## Engine Initialization

Single async factory function:

```typescript
const engine = await createEngine(canvas, {
  backend: 'auto', // 'auto' | 'webgpu' | 'webgl2'
  antialias: true, // 4x MSAA (default true)
  shadows: true, // or { resolution: 2048, constantBias: 0.001, ... }
  bloom: true, // or { intensity: 0.5, levels: 5, ... }
  debug: false, // Verbose validation
})
```

Auto-detects WebGPU with graceful WebGL2 fallback. The `await` handles async adapter/device creation (WebGPU) or synchronous context creation (WebGL2).

```typescript
engine.backend // 'webgpu' | 'webgl2' — which backend was selected
engine.device // Low-level device access (for advanced users)
engine.renderer // Low-level renderer access (for advanced users)
```

### Configuration Pattern

Boolean shorthand or detailed config object — both accepted:

```typescript
// Simple — uses sensible defaults
createEngine(canvas, { shadows: true, bloom: true })

// Detailed — full control
createEngine(canvas, {
  shadows: { resolution: 2048, backExtend: 75, constantBias: 0.001, slopeBias: 0.005 },
  bloom: { intensity: 0.5, levels: 5 },
})
```

## Scene

```typescript
const scene = createScene()

// Global ambient light (not a node)
scene.ambientLight = { color: [0.4, 0.45, 0.5], intensity: 0.3 }

// Scene graph operations
scene.add(mesh)
scene.add(group)
scene.remove(mesh)

// Traversal
scene.traverse(node => {
  /* depth-first, visits all nodes */
})

// Lookup
scene.getByName('player') // O(1) via internal Map
```

Separate from engine — multiple scenes can exist simultaneously.

## Object Creation

All objects created via factory functions with options objects:

### Geometry

```typescript
const plane    = createPlaneGeometry({ width: 10, height: 10 })
const box      = createBoxGeometry({ width: 1, height: 1, depth: 1 })
const sphere   = createSphereGeometry({ radius: 0.5, widthSegments: 32 })
const cone     = createConeGeometry({ radius: 1, height: 2 })
const cylinder = createCylinderGeometry({ radiusTop: 1, radiusBottom: 1, height: 2 })
const capsule  = createCapsuleGeometry({ radius: 0.5, height: 2 })
const circle   = createCircleGeometry({ radius: 1, segments: 32 })

// Custom geometry from raw data
const custom = createGeometry({
  positions: new Float32Array([...]),
  normals: new Float32Array([...]),
  indices: new Uint16Array([...]),
  uvs: new Float32Array([...]),
  colors: new Float32Array([...]),
  materialIndices: new Uint8Array([...]),
})
```

### Materials

```typescript
const basic = createBasicMaterial({
  color: [1, 0, 0],
  colorTexture: texture,
  vertexColors: false,
  transparent: false,
  opacity: 1.0,
})

const lambert = createLambertMaterial({
  color: [0.8, 0.6, 0.4],
  colorTexture: texture,
  aoTexture: aoTexture,
  vertexColors: false,
  receiveShadow: true,
  transparent: false,
  opacity: 1.0,
  palette: [{ color: [1, 1, 1] }, { color: [0, 0, 0], emissive: [0, 1, 1], emissiveIntensity: 0.7 }],
})
```

### Scene Objects

```typescript
const mesh = createMesh(geometry, material)
const group = createGroup()
const camera = createPerspectiveCamera({ fov: 60, near: 0.1, far: 1000 })
const sun = createDirectionalLight({ color: [1, 1, 1], intensity: 1.0, castShadow: true })
```

## Transform API

```typescript
// Position, rotation, scale
mesh.position.set(1, 0, 3) // Vec3 (Float32Array view)
mesh.rotation = quatFromAxisAngle(out, [0, 0, 1], Math.PI / 4) // Quaternion
mesh.scale.set(2, 2, 2) // Vec3 (Float32Array view)

// Visibility and shadow
mesh.visible = false
mesh.frustumCulled = true
mesh.castShadow = true
mesh.receiveShadow = true

// Helpers
mesh.lookAt([10, 0, 0]) // Z-up aware lookAt
mesh.name = 'player' // For scene.getByName()
```

Setting any transform property marks the node dirty — no explicit `updateMatrix()` call needed.

## Scene Graph API

```typescript
// Parent/child
scene.add(mesh)
group.add(child1, child2) // Variadic
group.remove(child1)

// Traversal
scene.traverse(node => {
  /* visits every node depth-first */
})

// Lookup
scene.getByName('player')

// Bone attachment (bones are regular nodes)
const handBone = skeleton.getBone('hand_R')
handBone.add(swordMesh)
```

## Material Index System

First-class feature for stylized games:

```typescript
const material = createLambertMaterial({
  palette: [
    { color: [1, 1, 1], opacity: 1.0 }, // 0: white
    { color: [0.2, 0.2, 0.2] }, // 1: dark
    { color: [0, 1, 1], emissive: [0, 1, 1], emissiveIntensity: 0.7 }, // 2: glowing teal
    { color: [1, 0, 0], opacity: 0.5 }, // 3: semi-transparent red
  ],
})
```

Per-vertex `_materialindex` attribute selects a palette entry. A single draw call renders an entire character with different colors, glow, and transparency per surface region.

## Animation API

```typescript
// Create mixer for a skeleton
const mixer = createAnimationMixer(skeleton)

// Create actions from clips (loaded via glTF)
const idle = mixer.clipAction(idleClip)
const walk = mixer.clipAction(walkClip)

// Playback
idle.play()
walk.loop = 'repeat' // 'repeat' | 'once' | 'pingpong'
walk.timeScale = 1.5 // Playback speed

// Crossfade (smooth transition)
idle.crossFadeTo(walk, 0.3) // 300ms crossfade

// Per-frame update
mixer.update(deltaTime)
```

## Raycasting API

```typescript
const raycaster = createRaycaster()

// From camera + screen coordinates (NDC: -1 to +1)
raycaster.setFromCamera([mouseNdcX, mouseNdcY], camera)

// From arbitrary ray
raycaster.set(origin, direction)

// Query
const hits = raycaster.intersectObjects(scene.children, true) // true = recursive

// Hit result
hits[0].distance // World-space distance from camera
hits[0].point // World-space intersection point
hits[0].normal // Interpolated surface normal
hits[0].uv // Interpolated UV coordinates
hits[0].triangleIndex // Triangle index in the geometry
hits[0].object // The mesh that was hit
```

Results sorted by distance (nearest first).

## Controls API

```typescript
const controls = createOrbitControls(camera, canvas, {
  target: [0, 0, 0],
  dampingFactor: 0.1,
  minDistance: 1,
  maxDistance: 100,
})

// Per-frame update
controls.update(deltaTime)

// Programmatic
controls.setTarget([10, 0, 5], { animate: true, duration: 0.5 })
controls.setPosition([20, -15, 10], { animate: true, duration: 0.5 })

// Events
controls.onChange(() => {
  /* camera moved */
})
controls.onStart(() => {
  /* interaction started */
})
controls.onEnd(() => {
  /* interaction ended */
})

// Cleanup
controls.dispose()
```

## Asset Loading API

```typescript
const gltf = await loadGLTF('/model.glb', engine, {
  draco: { decoderPath: '/draco-1.5.7/' },
  ktx2: { transcoderPath: '/basis-1.50/' },
})

// Use loaded data
scene.add(gltf.scene) // Scene hierarchy
const mixer = createAnimationMixer(gltf.skeletons[0])
mixer.clipAction(gltf.animations[0]).play()

// Access flat lists
gltf.meshes // All meshes
gltf.skeletons // All skeletons
gltf.animations // All animation clips
gltf.textures // All textures

// Cleanup
gltf.dispose() // Release all GPU resources
```

Cached by URL — multiple loads of the same URL return the same Promise.

## HTML Overlay API

```typescript
const overlay = createOverlayManager(canvas)

const label = overlay.add({
  element: labelDiv, // Any DOM element
  node: characterMesh, // Track a scene node
  offset: [0, 0, 2.5], // Offset from node (Z-up)
  center: true, // Center on projected point
  occlude: true, // Hide when behind geometry
  distanceScale: false, // Perspective size scaling
  pointerEvents: false, // Allow DOM clicks
})

// Remove
overlay.remove(label)

// Cleanup
overlay.dispose()
```

Updated internally by the engine each frame.

## Render Loop

Two options — managed or manual:

```typescript
// Option 1: Managed loop (recommended)
engine.onFrame((deltaTime, elapsed) => {
  mixer.update(deltaTime)
  controls.update(deltaTime)
})
engine.start(scene, camera)
engine.stop() // Stop the loop

// Option 2: Manual loop
const animate = () => {
  requestAnimationFrame(animate)
  const dt = clock.getDelta()
  mixer.update(dt)
  controls.update(dt)
  engine.render(scene, camera)
}
animate()
```

The managed loop provides `deltaTime` (seconds since last frame) and `elapsed` (total seconds since start).

## Frame Statistics

```typescript
const stats = engine.getStats()
{
  fps: number,
  frameTime: number,            // Total frame time (ms)
  jsTime: number,               // CPU time (ms)
  gpuTime: number,              // GPU time if available (ms)
  drawCalls: number,
  triangles: number,
  stateChanges: number,
  pipelineSwitches: number,
  culledObjects: number,
  visibleObjects: number,
  shadowDrawCalls: number,
}
```

## Resource Disposal

```typescript
geometry.dispose() // Release GPU vertex/index buffers
material.dispose() // Release GPU pipeline, textures (if owned)
gltf.dispose() // Release all resources from a glTF load
controls.dispose() // Remove event listeners
overlay.dispose() // Remove overlay container and elements
engine.dispose() // Dispose everything — device, renderer, all resources
```

React bindings handle disposal automatically on component unmount.

## Error Handling

```typescript
try {
  const engine = await createEngine(canvas)
} catch (e) {
  if (e instanceof UnsupportedBackendError) {
    // Neither WebGPU nor WebGL2 available
    showFallbackUI()
  }
}
```

In debug mode (`{ debug: true }`):

- Verbose shader compilation errors
- Missing attribute warnings
- GPU validation layers enabled
- Performance warnings (e.g., too many shader variants)

## Complete Vanilla Example

```typescript
import {
  createEngine,
  createScene,
  createPerspectiveCamera,
  createDirectionalLight,
  createMesh,
  createPlaneGeometry,
  createLambertMaterial,
  createOrbitControls,
  createAnimationMixer,
  loadGLTF,
  quatFromAxisAngle,
} from 'voidcore'

const canvas = document.querySelector('canvas')!

const engine = await createEngine(canvas, {
  shadows: true,
  bloom: { intensity: 0.5 },
})

// Scene
const scene = createScene()
scene.ambientLight = { color: [0.4, 0.4, 0.5], intensity: 0.3 }

// Light
const sun = createDirectionalLight({ color: [1, 1, 0.95], intensity: 1.0, castShadow: true })
sun.position.set(50, 50, 100)
scene.add(sun)

// Ground
const ground = createMesh(
  createPlaneGeometry({ width: 200, height: 200 }),
  createLambertMaterial({ color: [0.3, 0.5, 0.2], receiveShadow: true }),
)
scene.add(ground)

// Character
const gltf = await loadGLTF('/character.glb', engine, {
  draco: { decoderPath: '/draco-1.5.7/' },
})
scene.add(gltf.scene)

const mixer = createAnimationMixer(gltf.skeletons[0])
mixer.clipAction(gltf.animations[0]).play()

// Sword attached to hand bone
const sword = await loadGLTF('/sword.glb', engine)
const handBone = gltf.skeletons[0].getBone('hand_R')!
handBone.add(sword.scene)

// Camera + Controls
const camera = createPerspectiveCamera({ fov: 60 })
camera.position.set(5, -10, 5)
camera.lookAt([0, 0, 1])

const controls = createOrbitControls(camera, canvas, {
  target: [0, 0, 1],
  minDistance: 2,
  maxDistance: 50,
})

// Render loop
engine.onFrame(dt => {
  mixer.update(dt)
  controls.update(dt)
})
engine.start(scene, camera)
```

## Complete React Example

```tsx
import { Canvas, useFrame, useGLTF, useAnimations, Html, OrbitControls } from 'voidcore/react'
import { Suspense, useState, useRef, useEffect } from 'react'

const Character = () => {
  const gltf = useGLTF('/character.glb', { draco: { decoderPath: '/draco-1.5.7/' } })
  const { actions } = useAnimations(gltf.animations, gltf.skeletons[0])

  useEffect(() => {
    actions.idle?.play()
  }, [])

  return (
    <group>
      <primitive object={gltf.scene} castShadow />
      <Html position={[0, 0, 2.5]} center occlude>
        <div className="label">Player 1</div>
      </Html>
    </group>
  )
}

const App = () => (
  <Canvas shadows bloom={{ intensity: 0.5 }} camera={{ position: [5, -10, 5], fov: 60 }}>
    <ambientLight intensity={0.3} color={[0.4, 0.45, 0.5]} />
    <directionalLight position={[50, 50, 100]} castShadow />

    <mesh receiveShadow>
      <planeGeometry args={{ width: 200, height: 200 }} />
      <lambertMaterial color={[0.3, 0.5, 0.2]} />
    </mesh>

    <Suspense fallback={null}>
      <Character />
    </Suspense>

    <OrbitControls target={[0, 0, 1]} minDistance={2} maxDistance={50} />
  </Canvas>
)
```
