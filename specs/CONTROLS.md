# Voidcore — Controls

## Orbit Controls

The only camera interaction model in v1. Three operations:

- **Orbit**: rotate the camera around a target point
- **Pan**: move the target point in the camera's screen plane
- **Zoom**: change the distance from target (dolly in/out)

## API

Factory function, consistent with the rest of the engine:

```typescript
const controls = createOrbitControls(camera, canvas, {
  target: [0, 0, 0], // Point to orbit around
  dampingFactor: 0.1, // Deceleration rate (0 = instant stop, 1 = no deceleration)
  minDistance: 0.1, // Minimum zoom distance
  maxDistance: 1000, // Maximum zoom distance
  minElevation: -Math.PI / 2 + 0.01, // How far down (radians from horizon)
  maxElevation: Math.PI / 2 - 0.01, // How far up (small epsilon prevents gimbal lock)
  enabled: true, // Enable/disable all interaction
})
```

### Per-Frame Update

```typescript
// In the frame loop:
controls.update(deltaTime)
```

Updates camera position and orientation based on current velocity and damping. Must be called every frame for smooth damping.

### Programmatic Control

```typescript
// Immediate (no animation)
controls.target = [10, 0, 5]

// Animated transition
controls.setTarget([10, 0, 5], { animate: true, duration: 0.5 })
controls.setPosition([20, -15, 10], { animate: true, duration: 0.5 })
```

Animated transitions use ease-out interpolation for natural deceleration.

### State Access

```typescript
controls.target // Current target position (Vec3, readonly)
controls.enabled // Enable/disable interaction
controls.azimuth // Current azimuth angle (readonly)
controls.elevation // Current elevation angle (readonly)
controls.distance // Current distance from target (readonly)
```

### Events

```typescript
controls.onChange(() => {
  /* camera moved (orbit, pan, zoom, or animation) */
})
controls.onStart(() => {
  /* user interaction started (mousedown, touchstart) */
})
controls.onEnd(() => {
  /* user interaction ended (mouseup, touchend) */
})
```

### Cleanup

```typescript
controls.dispose() // Removes all mouse, touch, and wheel event listeners
```

Essential for preventing memory leaks when controls are created/destroyed dynamically (e.g., in React components).

## Angle Convention

**Elevation angle from XY plane** (Z-up convention):

- Elevation 0 = horizontal (looking at the horizon)
- Elevation +π/2 = looking straight down from above
- Elevation -π/2 = looking straight up from below
- Azimuth increases counter-clockwise when viewed from above

This is more intuitive than polar-from-Z-axis: positive elevation means "looking down," which matches the mental model for games.

### Spherical to Cartesian (Z-up)

```typescript
const x = target[0] + distance * Math.cos(elevation) * Math.cos(azimuth)
const y = target[1] + distance * Math.cos(elevation) * Math.sin(azimuth)
const z = target[2] + distance * Math.sin(elevation)
```

The camera position is computed from spherical coordinates, then `mat4LookAt` generates the view matrix pointing from the camera position to the target.

## Damping

Velocity-based damping with exponential decay:

```typescript
const dampingFactor = 0.1 // Default

// Each frame:
velocity.azimuth *= Math.pow(1 - dampingFactor, deltaTime * 60)
velocity.elevation *= Math.pow(1 - dampingFactor, deltaTime * 60)
velocity.distance *= Math.pow(1 - dampingFactor, deltaTime * 60)

azimuth += velocity.azimuth * deltaTime
elevation += velocity.elevation * deltaTime
distance += velocity.distance * deltaTime
```

The `deltaTime * 60` normalization ensures consistent feel regardless of frame rate (the decay is expressed per-60fps-frame, but applied continuously).

Damping factor of 0.1 provides smooth deceleration without feeling sluggish. At 0 there is no damping (instant stop on release). At 1.0 there would be no deceleration (infinite coast).

## Input Mapping

### Mouse

| Input              | Action                       |
| ------------------ | ---------------------------- |
| Left button drag   | Orbit (rotate around target) |
| Right button drag  | Pan (move target)            |
| Middle button drag | Pan (alternative)            |
| Scroll wheel       | Zoom (change distance)       |

### Touch

| Input                           | Action |
| ------------------------------- | ------ |
| 1-finger drag                   | Orbit  |
| 2-finger drag                   | Pan    |
| Pinch (2-finger spread/squeeze) | Zoom   |

Touch gesture recognition uses pointer count to distinguish orbit from pan/zoom. The switch happens on pointer down/up.

## Pan Implementation

Screen-space pan scaled by distance from target:

```typescript
const panSpeed = (distance * Math.tan(fov / 2) * 2) / canvasHeight

// Camera's local right and up vectors
const right = getCameraRight(camera) // X axis of camera's world matrix
const up = getCameraUp(camera) // Z axis in Z-up (up direction in screen space)

// Move target in screen plane
target[0] += (right[0] * -dx + up[0] * dy) * panSpeed
target[1] += (right[1] * -dx + up[1] * dy) * panSpeed
target[2] += (right[2] * -dx + up[2] * dy) * panSpeed
```

Scaling by `distance` ensures pan speed feels consistent regardless of zoom level: close zoom → small pan, far zoom → large pan. This matches the "1 pixel of mouse movement = 1 pixel of world movement" expectation.

## Constraints

```typescript
const controls = createOrbitControls(camera, canvas, {
  // Distance constraints
  minDistance: 0.1, // Can't zoom closer than this
  maxDistance: 1000, // Can't zoom further than this

  // Elevation constraints
  minElevation: -Math.PI / 2 + 0.01, // Can't look up past vertical (small epsilon)
  maxElevation: Math.PI / 2 - 0.01, // Can't look down past vertical

  // Azimuth constraints (optional, unrestricted by default)
  // minAzimuth: -Math.PI,
  // maxAzimuth: Math.PI,
})
```

Elevation is clamped with a small epsilon buffer (0.01 radians ≈ 0.57°) to prevent gimbal lock at the poles where the up vector becomes parallel to the view direction.

Azimuth is unrestricted by default (full 360° rotation). Optional `minAzimuth`/`maxAzimuth` for constrained rotation (e.g., architectural walkthroughs).

## Animated Transitions

```typescript
controls.setTarget([10, 0, 5], { animate: true, duration: 0.5 })
controls.setPosition([20, -15, 10], { animate: true, duration: 0.5 })
```

Implementation:

```typescript
// Interpolation state
let animationStart: number
let animationDuration: number
let fromTarget: Vec3
let toTarget: Vec3
let animating = false

// In update():
if (animating) {
  const t = Math.min(1, (elapsed - animationStart) / animationDuration)
  const eased = 1 - Math.pow(1 - t, 3) // Ease-out cubic
  vec3Lerp(currentTarget, fromTarget, toTarget, eased)
  if (t >= 1) animating = false
}
```

Animated transitions are interruptible — starting a new animation or beginning user interaction cancels the current animation.

## React Integration

```tsx
import { OrbitControls } from 'voidcore/react'
;<Canvas>
  <OrbitControls target={[0, 0, 0]} minDistance={1} maxDistance={100} dampingFactor={0.1} />
  {/* ... scene contents ... */}
</Canvas>
```

The React component:

- Creates controls on mount with the current camera and canvas
- Updates props reactively (changing `target` prop updates `controls.target`)
- Registers `controls.update(dt)` in the frame loop
- Disposes on unmount (removes event listeners)
