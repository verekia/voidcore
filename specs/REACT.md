# Voidcore — React Bindings

## Package Location

React bindings live inside the main `voidcore` package, exposed via a `/react` subpath export:

```typescript
import { Canvas, useFrame, useGLTF, Html } from 'voidcore/react'
```

`react` and `react-dom` are **optional peer dependencies** — users who don't import `voidcore/react` pay zero cost. Tree-shaking eliminates all React code from non-React bundles.

```json
// package.json
{
  "name": "voidcore",
  "exports": {
    ".": "./dist/index.js",
    "./react": "./dist/react/index.js"
  },
  "peerDependencies": {
    "react": "^18.0.0 || ^19.0.0",
    "react-dom": "^18.0.0 || ^19.0.0"
  },
  "peerDependenciesMeta": {
    "react": { "optional": true },
    "react-dom": { "optional": true }
  }
}
```

## Custom Reconciler

A custom React reconciler via `react-reconciler` maps React's component tree directly to the engine's scene graph. This enables declarative scene construction with automatic lifecycle management.

```tsx
import { Canvas, useFrame, useGLTF } from 'voidcore/react'

const App = () => (
  <Canvas shadows bloom camera={{ position: [5, -10, 5], fov: 60 }}>
    <ambientLight intensity={0.3} />
    <directionalLight position={[10, -10, 10]} castShadow />
    <mesh position={[0, 0, 0.5]} castShadow>
      <boxGeometry args={{ width: 1, height: 1, depth: 1 }} />
      <lambertMaterial color={[0.8, 0.2, 0.2]} />
    </mesh>
    <Player />
  </Canvas>
)
```

### Reconciler Implementation

The reconciler implements the `react-reconciler` host config interface:

```typescript
const hostConfig = {
  createInstance(type, props) {
    // Map JSX element type to engine factory function
    // e.g., 'mesh' → createMesh(), 'group' → createGroup()
    return createElement(type, props)
  },

  appendChild(parent, child) {
    // Geometry/material → attach to parent mesh
    if (isGeometry(child)) {
      parent.geometry = child
      return
    }
    if (isMaterial(child)) {
      parent.material = child
      return
    }
    // Scene node → add to parent
    parent.add(child)
  },

  commitUpdate(instance, updatePayload, type, oldProps, newProps) {
    // Diff props and apply changes directly to engine objects
    applyProps(instance, newProps, oldProps)
  },

  removeChild(parent, child) {
    parent.remove(child)
    child.dispose?.() // Auto-dispose GPU resources on unmount
  },

  // ... other host config methods
}

const reconciler = createReconciler(hostConfig)
```

## Canvas Component

The `<Canvas>` component is the root. It:

1. Creates a `<canvas>` DOM element
2. Initializes the engine (`createEngine`)
3. Creates a default scene and camera
4. Starts the render loop via `requestAnimationFrame`
5. Provides engine context to child components via React context

```tsx
<Canvas
  shadows={true} // or { mapSize: 2048, ... }
  bloom={{ intensity: 0.5 }} // or true for defaults
  camera={{ position: [5, -10, 5], fov: 60 }}
  backend="auto" // 'auto' | 'webgpu' | 'webgl2'
  antialias={true} // 4x MSAA
  onCreated={engine => {}} // Called after initialization
  style={{ width: '100%', height: '100%' }}
>
  {children}
</Canvas>
```

### Canvas Internals

```typescript
const Canvas = ({ children, camera, shadows, bloom, ...rest }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<Engine>(null)

  useEffect(() => {
    const init = async () => {
      const engine = await createEngine(canvasRef.current!, {
        shadows, bloom, backend: rest.backend, ...
      })
      engineRef.current = engine

      // Create default camera from props
      const cam = createPerspectiveCamera(camera)

      // Start reconciler
      const container = reconciler.createContainer(engine.scene, ...)
      reconciler.updateContainer(children, container, ...)

      // Start render loop
      engine.onFrame((dt) => {
        // Execute useFrame callbacks
        frameCallbacks.forEach(cb => cb(dt, engine))
      })
      engine.start(engine.scene, cam)
    }
    init()

    return () => engineRef.current?.dispose()
  }, [])

  return <canvas ref={canvasRef} {...rest} />
}
```

## JSX Element Mapping

Lowercase JSX elements map to engine types:

| JSX Element           | Engine Factory              | Notes                                 |
| --------------------- | --------------------------- | ------------------------------------- |
| `<group>`             | `createGroup()`             | Transform-only container              |
| `<mesh>`              | `createMesh()`              | Requires geometry + material children |
| `<skinnedMesh>`       | `createSkinnedMesh()`       | Requires skeleton prop                |
| `<perspectiveCamera>` | `createPerspectiveCamera()` | Usually set via Canvas `camera` prop  |
| `<directionalLight>`  | `createDirectionalLight()`  |                                       |
| `<ambientLight>`      | Scene ambient light         | Sets `scene.ambientLight`             |
| `<boxGeometry>`       | `createBoxGeometry()`       | Attached to parent mesh               |
| `<sphereGeometry>`    | `createSphereGeometry()`    | Attached to parent mesh               |
| `<planeGeometry>`     | `createPlaneGeometry()`     | Attached to parent mesh               |
| `<coneGeometry>`      | `createConeGeometry()`      | Attached to parent mesh               |
| `<cylinderGeometry>`  | `createCylinderGeometry()`  | Attached to parent mesh               |
| `<capsuleGeometry>`   | `createCapsuleGeometry()`   | Attached to parent mesh               |
| `<circleGeometry>`    | `createCircleGeometry()`    | Attached to parent mesh               |
| `<lambertMaterial>`   | `createLambertMaterial()`   | Attached to parent mesh               |
| `<basicMaterial>`     | `createBasicMaterial()`     | Attached to parent mesh               |

## Geometry and Material Attachment

When a geometry or material is a child of a mesh in JSX, the reconciler attaches it to the parent mesh rather than adding it to the scene graph:

```tsx
<mesh>
  <boxGeometry args={{ width: 1, height: 1, depth: 1 }} /> {/* → mesh.geometry = ... */}
  <lambertMaterial color={[1, 0, 0]} /> {/* → mesh.material = ... */}
</mesh>
```

The `args` prop passes construction options to the factory function. Other props are applied as property updates after creation.

## Hooks

### useFrame

Per-frame callback. Runs every `requestAnimationFrame`, outside of React's render cycle:

```typescript
useFrame((deltaTime, engine) => {
  meshRef.current.rotation = quatFromAxisAngle(out, [0, 0, 1], elapsed * 0.5)
})
```

Callbacks execute in registration order. No priority system — simplicity over flexibility.

**Important:** `useFrame` callbacks run outside React. They modify engine objects directly. This ensures 60fps rendering is never blocked by React's reconciliation.

### useEngine

Access the engine, scene, and camera:

```typescript
const { engine, scene, camera } = useEngine()
```

### useLoader

Generic Suspense-compatible asset loader:

```typescript
const texture = useLoader(loadTexture, '/textures/diffuse.ktx2')
```

Throws a Promise during loading (React Suspense catches it and shows a fallback). Cached by URL.

### useGLTF

Convenience hook for glTF loading:

```typescript
const gltf = useGLTF('/character.glb', {
  draco: { decoderPath: '/draco-1.5.7/' },
})

// Access scene, meshes, skeletons, animations
scene.add(gltf.scene)
```

Suspense-compatible. Cached by URL. Disposes automatically on unmount.

### useAnimations

Convenience hook for animation setup:

```typescript
const { actions, mixer } = useAnimations(gltf.animations, gltf.skeletons[0])

useEffect(() => {
  actions.idle?.play()
}, [])

// To crossfade:
actions.idle?.crossFadeTo(actions.walk!, 0.3)
```

Returns a map of animation names to `AnimationAction` objects. Automatically:

- Creates an `AnimationMixer`
- Registers `mixer.update(dt)` in the frame loop via `useFrame`
- Disposes the mixer on unmount

## Pointer Events

Raycast-based pointer events on meshes:

```tsx
<mesh
  onClick={e => console.log('clicked', e.point)}
  onPointerOver={e => setHovered(true)}
  onPointerOut={e => setHovered(false)}
  onPointerMove={e => {}}
>
  <boxGeometry args={{ width: 1, height: 1, depth: 1 }} />
  <lambertMaterial color={hovered ? [1, 0, 0] : [0.5, 0.5, 0.5]} />
</mesh>
```

### Event System

The reconciler tracks meshes with pointer event handlers. On each DOM pointer event on the canvas:

1. Convert mouse/touch position to NDC
2. Cast a ray from the camera through the pointer position
3. Intersect against all registered meshes (using the raycaster)
4. Dispatch events to hit meshes (closest first)
5. Track enter/leave state for `onPointerOver`/`onPointerOut`

### Event Interface

```typescript
interface PointerEvent3D {
  object: Mesh // The mesh that was hit
  point: Vec3 // World-space intersection point
  normal: Vec3 // Interpolated surface normal
  distance: number // Distance from camera
  uv: Vec2 | null // UV coordinates at hit point
  nativeEvent: MouseEvent | TouchEvent // Original DOM event
  stopPropagation(): void // Stop event from reaching meshes behind
}
```

`stopPropagation()` prevents the event from being dispatched to meshes further from the camera.

## Primitive Component

For pre-built scene graphs (e.g., loaded glTF models):

```tsx
const gltf = useGLTF('/model.glb')
return <primitive object={gltf.scene} />
```

The `primitive` component adds the object directly to the scene graph without wrapping it. Useful for glTF scenes that have their own hierarchy.

## Html Component

DOM overlays positioned in 3D space:

```tsx
import { Html } from 'voidcore/react'

<Html position={[0, 0, 2.5]} center occlude>
  <div className="player-label">Player Name</div>
</Html>

// With node tracking
<Html node={meshRef.current} offset={[0, 0, 2]} center>
  <div className="health-bar" />
</Html>
```

Implemented via `ReactDOM.createPortal` into the engine's overlay container div. Props map directly to the overlay manager API.

## Automatic Disposal

All GPU resources created by React components are disposed automatically on unmount:

- Geometries created via JSX `<boxGeometry>`, etc.
- Materials created via JSX `<lambertMaterial>`, etc.
- Textures loaded via `useLoader`
- glTF resources loaded via `useGLTF`
- Animation mixers from `useAnimations`
- Overlay elements from `<Html>`
- Controls from `<OrbitControls>`

## Render Loop Independence

The render loop runs via `requestAnimationFrame`, outside of React's render cycle. React state changes trigger prop updates that are applied to engine objects directly (not through React re-renders of the canvas).

```
React re-render → reconciler diffs props → applies changes to engine objects
                                                       ↓
requestAnimationFrame → engine.render() reads engine objects → GPU draw
```

This ensures 60fps rendering is never blocked by React's reconciliation, even during heavy component trees or frequent state updates.

## Prop Updates

When React re-renders a component, the reconciler diffs old and new props:

```tsx
// When position changes, reconciler calls: mesh.position.set(newX, newY, newZ)
<mesh position={[x, y, z]} />

// When color changes, reconciler calls: material.color = newColor
<lambertMaterial color={color} />
```

Array props (position, color, scale) are compared element-by-element to avoid unnecessary updates.

## TypeScript Support

Full JSX type definitions for all engine elements:

```typescript
// In voidcore/react
declare global {
  namespace JSX {
    interface IntrinsicElements {
      group: GroupProps
      mesh: MeshProps
      skinnedMesh: SkinnedMeshProps
      perspectiveCamera: CameraProps
      directionalLight: DirectionalLightProps
      ambientLight: AmbientLightProps
      boxGeometry: BoxGeometryProps
      sphereGeometry: SphereGeometryProps
      planeGeometry: PlaneGeometryProps
      coneGeometry: ConeGeometryProps
      cylinderGeometry: CylinderGeometryProps
      capsuleGeometry: CapsuleGeometryProps
      circleGeometry: CircleGeometryProps
      lambertMaterial: LambertMaterialProps
      basicMaterial: BasicMaterialProps
    }
  }
}
```

All props are fully typed. IDE autocompletion works for all element attributes.

## Complete React Example

```tsx
import { Canvas, useFrame, useGLTF, useAnimations, Html, OrbitControls } from 'voidcore/react'
import { quatFromAxisAngle } from 'voidcore'
import { useRef, useState, Suspense } from 'react'

const Character = () => {
  const gltf = useGLTF('/character.glb', { draco: { decoderPath: '/draco-1.5.7/' } })
  const { actions } = useAnimations(gltf.animations, gltf.skeletons[0])
  const [hovered, setHovered] = useState(false)

  useEffect(() => {
    actions.idle?.play()
  }, [])

  return (
    <group>
      <primitive object={gltf.scene} onPointerOver={() => setHovered(true)} onPointerOut={() => setHovered(false)} />
      <Html position={[0, 0, 2.5]} center>
        <div className="name-tag">Player 1</div>
      </Html>
    </group>
  )
}

const SpinningBox = () => {
  const ref = useRef()
  let elapsed = 0

  useFrame(dt => {
    elapsed += dt
    ref.current.rotation = quatFromAxisAngle(out, [0, 0, 1], elapsed)
  })

  return (
    <mesh ref={ref} position={[3, 0, 0.5]} castShadow>
      <boxGeometry args={{ width: 1, height: 1, depth: 1 }} />
      <lambertMaterial color={[0.2, 0.5, 1.0]} />
    </mesh>
  )
}

const App = () => (
  <Canvas shadows bloom={{ intensity: 0.4 }} camera={{ position: [5, -10, 5], fov: 60 }}>
    <ambientLight intensity={0.3} color={[0.4, 0.45, 0.5]} />
    <directionalLight position={[50, 50, 100]} castShadow />

    <mesh receiveShadow>
      <planeGeometry args={{ width: 200, height: 200 }} />
      <lambertMaterial color={[0.3, 0.5, 0.2]} />
    </mesh>

    <Suspense fallback={null}>
      <Character />
    </Suspense>

    <SpinningBox />

    <OrbitControls target={[0, 0, 1]} minDistance={2} maxDistance={50} />
  </Canvas>
)
```
