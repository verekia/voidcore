// Orbit Controls – Mouse/touch camera controls that orbit around a target point.
//
// The camera's position is defined in spherical coordinates (azimuth, elevation, distance)
// relative to a target point. User input (drag, scroll, pinch) adds velocity to these
// coordinates, and damping smoothly decays the velocity each frame for inertia.
//
// All input is handled through Pointer Events, which unify mouse and touch into a single
// code path. Multi-touch (pinch zoom + two-finger pan) is supported by tracking active
// pointers in a Map. The canvas sets `touch-action: none` to prevent the browser from
// intercepting touch gestures.
//
// Left drag / 1-finger  → orbit (rotate azimuth and elevation)
// Right drag / 2-finger → pan (shift the target point)
// Scroll / pinch        → zoom (change distance)
//
// OrbitControls.update(dt)  – Applies damping and recomputes the camera's position/view matrix.
// OrbitControls.dispose()   – Removes all event listeners.
// new OrbitControls()       – Creates controls for a camera + canvas pair.

import { mat4LookAt, VEC3_UP, vec3Create, vec3Set } from '../math/index'

import type { Vec3 } from '../math/index'
import type { PerspectiveCamera } from '../scene/camera'

export interface OrbitControlsOptions {
  target?: [number, number, number]
  dampingFactor?: number
  minDistance?: number
  maxDistance?: number
  minElevation?: number
  maxElevation?: number
  enabled?: boolean
}

export class OrbitControls {
  camera: PerspectiveCamera
  canvas: HTMLCanvasElement
  target: Vec3
  enabled: boolean

  dampingFactor: number
  minDistance: number
  maxDistance: number
  minElevation: number
  maxElevation: number

  azimuth = 0
  elevation = 0.5
  distance = 10

  private _velocityAz = 0
  private _velocityEl = 0
  private _velocityDist = 0
  private _velocityPanX = 0
  private _velocityPanY = 0

  private _pointers = new Map<number, { x: number; y: number }>()
  private _pointerButton = 0
  private _lastX = 0
  private _lastY = 0
  private _lastPinchDist = 0

  private _onChange: (() => void)[] = []

  constructor(camera: PerspectiveCamera, canvas: HTMLCanvasElement, opts: OrbitControlsOptions = {}) {
    this.camera = camera
    this.canvas = canvas
    this.target = vec3Create()
    if (opts.target) vec3Set(this.target, opts.target[0], opts.target[1], opts.target[2])
    this.dampingFactor = opts.dampingFactor ?? 0.1
    this.minDistance = opts.minDistance ?? 0.1
    this.maxDistance = opts.maxDistance ?? 1000
    this.minElevation = opts.minElevation ?? -Math.PI / 2 + 0.01
    this.maxElevation = opts.maxElevation ?? Math.PI / 2 - 0.01
    this.enabled = opts.enabled ?? true

    // Prevent browser touch gestures (scroll, pinch-zoom, pull-to-refresh)
    canvas.style.touchAction = 'none'

    // Compute initial spherical from camera position
    const dx = camera.position[0]! - this.target[0]!
    const dy = camera.position[1]! - this.target[1]!
    const dz = camera.position[2]! - this.target[2]!
    this.distance = Math.sqrt(dx * dx + dy * dy + dz * dz)
    if (this.distance > 0.01) {
      this.elevation = Math.asin(Math.max(-1, Math.min(1, dz / this.distance)))
      this.azimuth = Math.atan2(dy, dx)
    }

    this._bindEvents()
  }

  update(dt: number) {
    if (!this.enabled) return

    const damping = Math.pow(1 - this.dampingFactor, dt * 60)
    this._velocityAz *= damping
    this._velocityEl *= damping
    this._velocityDist *= damping
    this._velocityPanX *= damping
    this._velocityPanY *= damping

    this.azimuth += this._velocityAz
    this.elevation += this._velocityEl
    this.elevation = Math.max(this.minElevation, Math.min(this.maxElevation, this.elevation))
    this.distance += this._velocityDist
    this.distance = Math.max(this.minDistance, Math.min(this.maxDistance, this.distance))

    // Pan
    if (Math.abs(this._velocityPanX) > 1e-6 || Math.abs(this._velocityPanY) > 1e-6) {
      // Camera right and up vectors
      const cosEl = Math.cos(this.elevation)
      const cosAz = Math.cos(this.azimuth)
      const sinAz = Math.sin(this.azimuth)

      // Right vector (perpendicular to forward in XY plane)
      const rx = -sinAz,
        ry = cosAz

      // Up vector (Z-up)
      const sinEl = Math.sin(this.elevation)
      const ux = -sinEl * cosAz
      const uy = -sinEl * sinAz
      const uz = cosEl

      const panSpeed = (this.distance * Math.tan((this.camera.fov * Math.PI) / 180 / 2) * 2) / this.canvas.clientHeight

      this.target[0] = this.target[0]! + (rx * this._velocityPanX + ux * this._velocityPanY) * panSpeed
      this.target[1] = this.target[1]! + (ry * this._velocityPanX + uy * this._velocityPanY) * panSpeed
      this.target[2] = this.target[2]! + uz * this._velocityPanY * panSpeed
    }

    // Compute camera position from spherical coordinates (Z-up)
    const cosEl = Math.cos(this.elevation)
    this.camera.setPosition(
      this.target[0]! + this.distance * cosEl * Math.cos(this.azimuth),
      this.target[1]! + this.distance * cosEl * Math.sin(this.azimuth),
      this.target[2]! + this.distance * Math.sin(this.elevation),
    )

    // Compute view matrix directly (bypasses Node.lookAt quaternion convention)
    mat4LookAt(this.camera._viewMatrix, this.camera.position, this.target, VEC3_UP)

    for (const cb of this._onChange) cb()
  }

  onChange(callback: () => void) {
    this._onChange.push(callback)
  }

  dispose() {
    this.canvas.removeEventListener('pointerdown', this._onPointerDown)
    this.canvas.removeEventListener('pointermove', this._onPointerMove)
    this.canvas.removeEventListener('pointerup', this._onPointerUp)
    this.canvas.removeEventListener('pointerleave', this._onPointerUp)
    this.canvas.removeEventListener('pointercancel', this._onPointerUp)
    this.canvas.removeEventListener('wheel', this._onWheel)
    this.canvas.removeEventListener('contextmenu', this._onContextMenu)
  }

  private _bindEvents() {
    this.canvas.addEventListener('pointerdown', this._onPointerDown)
    this.canvas.addEventListener('pointermove', this._onPointerMove)
    this.canvas.addEventListener('pointerup', this._onPointerUp)
    this.canvas.addEventListener('pointerleave', this._onPointerUp)
    this.canvas.addEventListener('pointercancel', this._onPointerUp)
    this.canvas.addEventListener('wheel', this._onWheel, { passive: false })
    this.canvas.addEventListener('contextmenu', this._onContextMenu)
  }

  private _onPointerDown = (e: PointerEvent) => {
    if (!this.enabled) return
    this._pointers.set(e.pointerId, { x: e.clientX, y: e.clientY })
    this.canvas.setPointerCapture(e.pointerId)

    if (this._pointers.size === 1) {
      this._pointerButton = e.button
      this._lastX = e.clientX
      this._lastY = e.clientY
    } else if (this._pointers.size === 2) {
      const pts = [...this._pointers.values()]
      const dx = pts[1]!.x - pts[0]!.x
      const dy = pts[1]!.y - pts[0]!.y
      this._lastPinchDist = Math.sqrt(dx * dx + dy * dy)
      this._lastX = (pts[0]!.x + pts[1]!.x) / 2
      this._lastY = (pts[0]!.y + pts[1]!.y) / 2
    }
  }

  private _onPointerMove = (e: PointerEvent) => {
    if (!this.enabled || !this._pointers.has(e.pointerId)) return
    this._pointers.set(e.pointerId, { x: e.clientX, y: e.clientY })

    if (this._pointers.size === 1) {
      const dx = e.clientX - this._lastX
      const dy = e.clientY - this._lastY
      this._lastX = e.clientX
      this._lastY = e.clientY

      if (this._pointerButton === 0) {
        // Left button / single touch: orbit
        this._velocityAz -= dx * 0.002
        this._velocityEl += dy * 0.002
      } else if (this._pointerButton === 2 || this._pointerButton === 1) {
        // Right/middle button: pan
        this._velocityPanX -= dx * 0.2
        this._velocityPanY += dy * 0.2
      }
    } else if (this._pointers.size === 2) {
      const pts = [...this._pointers.values()]

      // Pinch zoom
      const pdx = pts[1]!.x - pts[0]!.x
      const pdy = pts[1]!.y - pts[0]!.y
      const dist = Math.sqrt(pdx * pdx + pdy * pdy)
      const delta = this._lastPinchDist - dist
      this._lastPinchDist = dist
      this._velocityDist += delta * 0.002 * this.distance

      // Two-finger pan
      const mx = (pts[0]!.x + pts[1]!.x) / 2
      const my = (pts[0]!.y + pts[1]!.y) / 2
      const panDx = mx - this._lastX
      const panDy = my - this._lastY
      this._lastX = mx
      this._lastY = my
      this._velocityPanX -= panDx * 0.5
      this._velocityPanY += panDy * 0.5
    }
  }

  private _onPointerUp = (e: PointerEvent) => {
    this._pointers.delete(e.pointerId)
    try {
      this.canvas.releasePointerCapture(e.pointerId)
    } catch {}

    // When going from 2 pointers to 1, reset tracking to avoid a jump
    if (this._pointers.size === 1) {
      const remaining = [...this._pointers.values()][0]!
      this._lastX = remaining.x
      this._lastY = remaining.y
      this._pointerButton = 0
    }
  }

  private _onWheel = (e: WheelEvent) => {
    if (!this.enabled) return
    e.preventDefault()

    // Normalize delta across deltaMode (pixel / line / page)
    let delta = e.deltaY
    if (e.deltaMode === 1) delta *= 16
    else if (e.deltaMode === 2) delta *= 100

    // Clamp per-event magnitude so mouse wheel (~100) and trackpad (~1-10) feel consistent
    const clamped = Math.sign(delta) * Math.min(Math.abs(delta), 50)
    this._velocityDist += clamped * 0.0002 * this.distance
  }

  private _onContextMenu = (e: Event) => {
    e.preventDefault()
  }
}
