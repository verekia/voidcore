# Voidcore — HTML Overlay

## Architecture

An absolute-positioned container div sits directly above the canvas. Individual overlay elements are positioned using GPU-accelerated CSS transforms based on their projected 3D world positions.

```html
<div style="position: relative;">
  <canvas />
  <div id="voidcore-overlay" style="position: absolute; inset: 0; pointer-events: none; overflow: hidden;">
    <!-- Individual overlay elements positioned via CSS transform -->
    <div style="transform: translate(450px, 280px); pointer-events: none;">Player Label</div>
  </div>
</div>
```

The overlay container is created automatically when `createOverlayManager` is called (or when `<Html>` is used in React). It has `pointer-events: none` and `overflow: hidden` to prevent intercepting canvas clicks and bleeding outside the canvas area.

## Projection Pipeline

Standard world-to-screen projection, executed once per overlay per frame:

```typescript
const projectToScreen = (
  worldPos: Vec3,
  vpMatrix: Mat4,
  canvasWidth: number,
  canvasHeight: number,
): { x: number; y: number; visible: boolean } => {
  // 1. World → clip space
  const clipPos = vec4TransformMat4(_tempVec4, [worldPos[0], worldPos[1], worldPos[2], 1], vpMatrix)

  // 2. Behind camera check
  if (clipPos[3] <= 0) return { x: 0, y: 0, visible: false }

  // 3. Perspective divide → NDC
  const ndcX = clipPos[0] / clipPos[3]
  const ndcY = clipPos[1] / clipPos[3]

  // 4. Frustum check
  if (Math.abs(ndcX) > 1.1 || Math.abs(ndcY) > 1.1) return { x: 0, y: 0, visible: false }

  // 5. NDC → screen pixels
  const screenX = (ndcX * 0.5 + 0.5) * canvasWidth
  const screenY = (1 - (ndcY * 0.5 + 0.5)) * canvasHeight // Y flipped for DOM coordinates

  return { x: screenX, y: screenY, visible: true }
}
```

The 1.1 frustum margin (instead of 1.0) prevents elements from popping out abruptly at the screen edges.

## CSS Transform Positioning

GPU-accelerated positioning via `transform`:

```typescript
element.style.transform = `translate(${screenX}px, ${screenY}px)`
element.style.willChange = 'transform' // Hint for GPU layer promotion
```

CSS transforms avoid triggering layout/reflow, making them the cheapest way to move DOM elements.

## Centering

```typescript
if (center) {
  // Center the element on the projected point
  element.style.transform = `translate(${screenX}px, ${screenY}px) translate(-50%, -50%)`
} else {
  // Top-left corner at the projected point
  element.style.transform = `translate(${screenX}px, ${screenY}px)`
}
```

- `center: true` — element visually centered on the 3D position (default for labels, health bars)
- `center: false` — element's top-left corner at the 3D position

Using CSS `translate(-50%, -50%)` for centering avoids needing to measure the element's dimensions in JavaScript.

## Tracking Modes

### Static Position

Fixed world-space position:

```typescript
overlay.add({
  element: labelDiv,
  position: [10, 5, 3],
  center: true,
})
```

The overlay stays at the given world position. Useful for location markers, waypoints.

### Node Tracking

Follow a scene node with an optional offset:

```typescript
overlay.add({
  element: healthBar,
  node: characterMesh,
  offset: [0, 0, 2.5], // 2.5 units above the character (Z-up)
  center: true,
})
```

Each frame, the overlay position is computed as:

```typescript
vec3Add(worldPos, node.worldPosition, offset)
```

The overlay automatically moves as the node moves, animates, or is reparented.

## Occlusion Testing

Optional raycast-based occlusion — when enabled, overlays behind opaque geometry are hidden:

```typescript
overlay.add({
  element: labelDiv,
  node: characterMesh,
  offset: [0, 0, 2.5],
  occlude: true, // Enable occlusion testing
})
```

When `occlude: true`, a ray is cast from the camera to the overlay's world position each frame. If the ray hits an opaque object closer than the overlay, the element is hidden (`display: none`).

### Staggered Updates

Not all overlays are occlusion-tested every frame. Tests are distributed across frames:

```typescript
if ((frameCount + overlayIndex) % 3 === 0) {
  overlay._occluded = testOcclusion(overlay, camera, scene)
}
```

This reduces the per-frame cost from N rays to N/3 rays. For 20 overlays, that's ~7 rays per frame instead of 20. At ~0.02ms per ray, this costs ~0.14ms instead of ~0.4ms.

The 3-frame latency is imperceptible for slow-moving objects. For fast-moving scenes, the stagger interval can be reduced to 1 (every frame).

## Distance Scaling

Optional perspective-consistent size scaling:

```typescript
overlay.add({
  element: labelDiv,
  node: mesh,
  distanceScale: true, // Scale element based on camera distance
})
```

When enabled, the element's CSS `scale` is adjusted:

```typescript
const distance = vec3Distance(camera.worldPosition, overlayWorldPos)
const baseDistance = 10 // Distance at which scale = 1.0
const scale = baseDistance / distance
element.style.transform = `translate(${x}px, ${y}px) scale(${scale})`
```

Closer overlays appear larger, farther ones smaller — matching the perspective of the 3D scene.

## Depth-Based Z-Index

Overlays closer to the camera automatically receive higher CSS `z-index`:

```typescript
const normalizedDepth = clipPos[3] / camera.far // 0 = near, 1 = far
element.style.zIndex = String(Math.floor((1 - normalizedDepth) * 10000))
```

This ensures proper stacking when multiple overlays overlap on screen (e.g., labels for objects at different distances).

## Pointer Events

The overlay container has `pointer-events: none`. Individual elements can opt in:

```typescript
overlay.add({
  element: buttonDiv,
  node: mesh,
  pointerEvents: true, // Sets pointer-events: auto on this element
})
```

Elements with `pointerEvents: true` can receive clicks, hovers, etc. This is useful for interactive UI (buttons, links) while keeping non-interactive labels from blocking canvas interaction.

## Dirty Checking

Skip DOM updates when the position change is sub-pixel:

```typescript
const dx = Math.abs(newX - overlay._lastScreenX)
const dy = Math.abs(newY - overlay._lastScreenY)

if (dx < 0.5 && dy < 0.5) return // No visible change, skip DOM write

overlay._lastScreenX = newX
overlay._lastScreenY = newY
element.style.transform = `translate(${newX}px, ${newY}px)`
```

This minimizes DOM writes. Even CSS transform updates trigger browser compositing work, so skipping unnecessary updates saves ~0.01ms per overlay per frame.

## Visibility Handling

Elements are hidden (`display: none`) when any of:

- Behind the camera (`clipPos[3] <= 0`)
- Outside the frustum (NDC > 1.1)
- Occluded by opaque geometry (when `occlude: true`)

```typescript
if (!projected.visible || overlay._occluded) {
  if (element.style.display !== 'none') {
    element.style.display = 'none'
  }
} else {
  if (element.style.display === 'none') {
    element.style.display = ''
  }
}
```

The display check prevents redundant style writes.

## Vanilla API

```typescript
const overlay = createOverlayManager(canvas)

// Add an overlay
const handle = overlay.add({
  element: document.createElement('div'), // Any DOM element
  position: [0, 0, 2.5], // OR node + offset
  node: characterMesh, // Optional: track a scene node
  offset: [0, 0, 2.5], // Offset from tracked node
  center: true, // Center on projected point
  occlude: false, // Raycast occlusion testing
  distanceScale: false, // Perspective size scaling
  pointerEvents: false, // Allow DOM pointer events
})

// Remove an overlay
overlay.remove(handle)

// Update is called internally by the engine each frame:
// overlay.update(camera, canvasWidth, canvasHeight)

// Cleanup
overlay.dispose() // Removes container div and all event listeners
```

## React Integration

```tsx
import { Html } from 'voidcore/react'

// Static position
<Html position={[0, 0, 2.5]} center occlude>
  <div className="player-label">Player Name</div>
</Html>

// Track a node
<Html node={meshRef.current} offset={[0, 0, 2]} center occlude>
  <div className="health-bar" style={{ width: `${hp}%` }} />
</Html>

// Interactive overlay
<Html position={[5, 0, 1]} center pointerEvents>
  <button onClick={() => alert('clicked!')}>Click Me</button>
</Html>
```

Implemented via `ReactDOM.createPortal` into the overlay container div. Props map directly to overlay manager options. The overlay is automatically added on mount and removed on unmount.

## Performance Budget

```
Position projection (20 overlays):           ~0.05ms
DOM updates (20 overlays, with dirty check): ~0.10ms
Occlusion rays (7 per frame, staggered):     ~0.10ms
───────────────────────────────────────────────────────
Total overlay overhead:                      ~0.25ms
```

For scenes with fewer overlays (<10), the overhead is negligible (<0.1ms).
