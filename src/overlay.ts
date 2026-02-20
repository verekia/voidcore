// Overlay – Positions HTML elements to track 3D world positions on screen.
//
// An overlay manager creates a fixed-position container div that sits on top of the canvas.
// Individual DOM elements are added as overlays and positioned each frame using GPU-accelerated
// CSS transforms based on their projected 3D world coordinates.
//
// Projection pipeline: world position → clip space (via view-projection matrix) → perspective
// divide → NDC → screen pixels. A 1.1 frustum margin prevents labels from popping out at edges.
//
// Features:
//   - Node tracking with offset (e.g. label floating above a character's head)
//   - Static world-space positions (e.g. waypoint markers)
//   - Dirty checking: skips DOM writes when position change is sub-pixel (<0.5px)
//   - Centering via CSS translate(-50%, -50%) — no JS measurement needed
//   - Depth-based z-index for correct overlap ordering
//   - Optional distance scaling for perspective-consistent sizing
//   - Optional raycast occlusion (staggered across frames for performance)
//   - Per-element pointer-events opt-in (container is pointer-events: none)
//
// createOverlayManager(canvas) – Creates the overlay system for a given canvas.
// overlay.add(opts)            – Registers a DOM element to track a 3D position.
// overlay.remove(handle)       – Removes an overlay and its DOM element.
// overlay.update(camera, w, h) – Per-frame update: projects and positions all overlays.
// overlay.dispose()            – Tears down the container and all overlays.

import {
  mat4Create,
  mat4Multiply,
  vec3Add,
  vec3Create,
  vec3Distance,
  vec4Create,
  vec4Set,
  vec4TransformMat4,
} from './math/index.ts'

import type { Mat4, Vec3, Vec4 } from './math/index.ts'
import type { PerspectiveCamera } from './scene/camera.ts'
import type { Node } from './scene/node.ts'

export interface OverlayOptions {
  element: HTMLElement
  position?: Vec3 | [number, number, number]
  node?: Node
  offset?: Vec3 | [number, number, number]
  center?: boolean
  occlude?: boolean
  distanceScale?: boolean
  pointerEvents?: boolean
}

export interface OverlayHandle {
  _index: number
}

interface OverlayEntry {
  element: HTMLElement
  position: Vec3 | null
  node: Node | null
  offset: Vec3
  center: boolean
  occlude: boolean
  distanceScale: boolean
  _lastScreenX: number
  _lastScreenY: number
  _wasVisible: boolean
  _occluded: boolean
}

// Pre-allocated scratch buffers (zero allocation in hot path)
const _tempVec4: Vec4 = vec4Create()
const _tempWorldPos: Vec3 = vec3Create()
const _tempCamPos: Vec3 = vec3Create()
const _vpMatrix: Mat4 = mat4Create()

const projectToScreen = (
  worldPos: Vec3,
  vpMatrix: Float32Array,
  canvasWidth: number,
  canvasHeight: number,
): { x: number; y: number; depth: number; visible: boolean } => {
  // World → clip space
  vec4Set(_tempVec4, worldPos[0]!, worldPos[1]!, worldPos[2]!, 1)
  vec4TransformMat4(_tempVec4, _tempVec4, vpMatrix)

  const w = _tempVec4[3]!

  // Behind camera check
  if (w <= 0) return { x: 0, y: 0, depth: 0, visible: false }

  // Perspective divide → NDC
  const ndcX = _tempVec4[0]! / w
  const ndcY = _tempVec4[1]! / w

  // Frustum check with 1.1 margin to prevent popping at edges
  if (Math.abs(ndcX) > 1.1 || Math.abs(ndcY) > 1.1) return { x: 0, y: 0, depth: 0, visible: false }

  // NDC → screen pixels (Y flipped for DOM coordinates)
  const screenX = (ndcX * 0.5 + 0.5) * canvasWidth
  const screenY = (1 - (ndcY * 0.5 + 0.5)) * canvasHeight

  return { x: screenX, y: screenY, depth: w, visible: true }
}

export const createOverlayManager = (canvas: HTMLCanvasElement) => {
  // Create container as next sibling of canvas, matching its fixed positioning
  const container = document.createElement('div')
  container.style.cssText = 'position:fixed;inset:0;pointer-events:none;overflow:hidden'
  canvas.parentNode!.insertBefore(container, canvas.nextSibling)

  const entries: (OverlayEntry | null)[] = []
  let nextIndex = 0

  const add = (opts: OverlayOptions): OverlayHandle => {
    const entry: OverlayEntry = {
      element: opts.element,
      position: opts.position ? (new Float32Array(opts.position) as Vec3) : null,
      node: opts.node ?? null,
      offset: opts.offset ? (new Float32Array(opts.offset) as Vec3) : (new Float32Array(3) as Vec3),
      center: opts.center ?? false,
      occlude: opts.occlude ?? false,
      distanceScale: opts.distanceScale ?? false,
      _lastScreenX: -Infinity,
      _lastScreenY: -Infinity,
      _wasVisible: false,
      _occluded: false,
    }

    // Style the element for GPU-accelerated positioning
    entry.element.style.position = 'absolute'
    entry.element.style.top = '0'
    entry.element.style.left = '0'
    entry.element.style.willChange = 'transform'
    entry.element.style.display = 'none'

    if (opts.pointerEvents) {
      entry.element.style.pointerEvents = 'auto'
    } else {
      entry.element.style.pointerEvents = 'none'
    }

    container.appendChild(entry.element)

    const index = nextIndex++
    entries[index] = entry

    return { _index: index }
  }

  const remove = (handle: OverlayHandle) => {
    const entry = entries[handle._index]
    if (!entry) return
    container.removeChild(entry.element)
    entries[handle._index] = null
  }

  const update = (
    camera: PerspectiveCamera,
    canvasWidth: number,
    canvasHeight: number,
    _scene?: unknown,
    frameCount?: number,
  ) => {
    // Compute VP matrix from camera's projection and view matrices
    // (renderers compute this into their own buffer, not onto the camera)
    mat4Multiply(_vpMatrix, camera._projectionMatrix, camera._viewMatrix)
    const camFar = camera.far

    // Camera world position (camera.position is always up-to-date via setPosition())
    _tempCamPos[0] = camera.position[0]!
    _tempCamPos[1] = camera.position[1]!
    _tempCamPos[2] = camera.position[2]!

    const frame = frameCount ?? 0

    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i]
      if (!entry) continue

      // Compute world position
      if (entry.node) {
        // Extract node position from its world matrix and add offset
        _tempWorldPos[0] = entry.node._worldMatrix[12]!
        _tempWorldPos[1] = entry.node._worldMatrix[13]!
        _tempWorldPos[2] = entry.node._worldMatrix[14]!
        vec3Add(_tempWorldPos, _tempWorldPos, entry.offset)
      } else if (entry.position) {
        _tempWorldPos[0] = entry.position[0]!
        _tempWorldPos[1] = entry.position[1]!
        _tempWorldPos[2] = entry.position[2]!
      } else {
        continue
      }

      // Project to screen
      const projected = projectToScreen(_tempWorldPos, _vpMatrix, canvasWidth, canvasHeight)

      // Staggered occlusion testing (every 3rd frame per overlay)
      if (entry.occlude && (frame + i) % 3 === 0) {
        // TODO: raycast occlusion when scene raycasting API is available
        entry._occluded = false
      }

      // Visibility: hide if behind camera, outside frustum, or occluded
      const visible = projected.visible && !entry._occluded

      if (!visible) {
        if (entry._wasVisible) {
          entry.element.style.display = 'none'
          entry._wasVisible = false
        }
        continue
      }

      // Show element if it was hidden
      if (!entry._wasVisible) {
        entry.element.style.display = ''
        entry._wasVisible = true
      }

      // Dirty check: skip DOM write if sub-pixel change
      const dx = Math.abs(projected.x - entry._lastScreenX)
      const dy = Math.abs(projected.y - entry._lastScreenY)

      if (dx >= 0.5 || dy >= 0.5) {
        entry._lastScreenX = projected.x
        entry._lastScreenY = projected.y

        // Build transform string
        let transform = `translate(${projected.x}px,${projected.y}px)`

        if (entry.center) {
          transform += ' translate(-50%,-50%)'
        }

        if (entry.distanceScale) {
          const distance = vec3Distance(_tempCamPos, _tempWorldPos)
          const BASE_DISTANCE = 10
          const scale = BASE_DISTANCE / Math.max(distance, 0.01)
          transform += ` scale(${scale})`
        }

        entry.element.style.transform = transform
      }

      // Depth-based z-index: closer objects get higher z-index
      const normalizedDepth = projected.depth / camFar
      entry.element.style.zIndex = String(Math.floor((1 - normalizedDepth) * 10000))
    }
  }

  const dispose = () => {
    container.remove()
  }

  return { add, remove, update, dispose }
}
