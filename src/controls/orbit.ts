import { mat4LookAt, VEC3_UP, vec3Create, vec3Set } from '../math/index.ts'

import type { Vec3 } from '../math/index.ts'
import type { PerspectiveCamera } from '../scene/camera.ts'

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

  private _pointerDown = false
  private _pointerButton = 0
  private _lastX = 0
  private _lastY = 0
  private _touchCount = 0
  private _lastPinchDist = 0

  private _onChange: (() => void)[] = []
  private _disposed = false

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

      this.target[0]! += (rx * this._velocityPanX + ux * this._velocityPanY) * panSpeed
      this.target[1]! += (ry * this._velocityPanX + uy * this._velocityPanY) * panSpeed
      this.target[2]! += uz * this._velocityPanY * panSpeed
    }

    // Compute camera position from spherical coordinates (Z-up)
    const cosEl = Math.cos(this.elevation)
    this.camera.position[0] = this.target[0]! + this.distance * cosEl * Math.cos(this.azimuth)
    this.camera.position[1] = this.target[1]! + this.distance * cosEl * Math.sin(this.azimuth)
    this.camera.position[2] = this.target[2]! + this.distance * Math.sin(this.elevation)
    this.camera._dirtyLocal = true
    this.camera._dirtyWorld = true

    // Compute view matrix directly (bypasses Node.lookAt quaternion convention)
    mat4LookAt(this.camera._viewMatrix, this.camera.position, this.target, VEC3_UP)

    for (const cb of this._onChange) cb()
  }

  onChange(callback: () => void) {
    this._onChange.push(callback)
  }

  dispose() {
    this._disposed = true
    this.canvas.removeEventListener('pointerdown', this._onPointerDown)
    this.canvas.removeEventListener('pointermove', this._onPointerMove)
    this.canvas.removeEventListener('pointerup', this._onPointerUp)
    this.canvas.removeEventListener('wheel', this._onWheel)
    this.canvas.removeEventListener('contextmenu', this._onContextMenu)
    this.canvas.removeEventListener('touchstart', this._onTouchStart)
    this.canvas.removeEventListener('touchmove', this._onTouchMove)
    this.canvas.removeEventListener('touchend', this._onTouchEnd)
  }

  private _bindEvents() {
    this.canvas.addEventListener('pointerdown', this._onPointerDown)
    this.canvas.addEventListener('pointermove', this._onPointerMove)
    this.canvas.addEventListener('pointerup', this._onPointerUp)
    this.canvas.addEventListener('pointerleave', this._onPointerUp)
    this.canvas.addEventListener('wheel', this._onWheel, { passive: false })
    this.canvas.addEventListener('contextmenu', this._onContextMenu)
    this.canvas.addEventListener('touchstart', this._onTouchStart, { passive: false })
    this.canvas.addEventListener('touchmove', this._onTouchMove, { passive: false })
    this.canvas.addEventListener('touchend', this._onTouchEnd)
  }

  private _onPointerDown = (e: PointerEvent) => {
    if (!this.enabled) return
    this._pointerDown = true
    this._pointerButton = e.button
    this._lastX = e.clientX
    this._lastY = e.clientY
    this.canvas.setPointerCapture(e.pointerId)
  }

  private _onPointerMove = (e: PointerEvent) => {
    if (!this.enabled || !this._pointerDown) return
    const dx = e.clientX - this._lastX
    const dy = e.clientY - this._lastY
    this._lastX = e.clientX
    this._lastY = e.clientY

    if (this._pointerButton === 0) {
      // Left button: orbit
      this._velocityAz -= dx * 0.005
      this._velocityEl += dy * 0.005
    } else if (this._pointerButton === 2 || this._pointerButton === 1) {
      // Right/middle button: pan
      this._velocityPanX += dx * 2
      this._velocityPanY += dy * 2
    }
  }

  private _onPointerUp = (e: PointerEvent) => {
    this._pointerDown = false
    try {
      this.canvas.releasePointerCapture(e.pointerId)
    } catch {}
  }

  private _onWheel = (e: WheelEvent) => {
    if (!this.enabled) return
    e.preventDefault()
    this._velocityDist += e.deltaY * 0.01 * (this.distance * 0.1)
  }

  private _onContextMenu = (e: Event) => {
    e.preventDefault()
  }

  private _onTouchStart = (e: TouchEvent) => {
    if (!this.enabled) return
    e.preventDefault()
    this._touchCount = e.touches.length
    if (e.touches.length === 1) {
      this._lastX = e.touches[0]!.clientX
      this._lastY = e.touches[0]!.clientY
    } else if (e.touches.length === 2) {
      const dx = e.touches[1]!.clientX - e.touches[0]!.clientX
      const dy = e.touches[1]!.clientY - e.touches[0]!.clientY
      this._lastPinchDist = Math.sqrt(dx * dx + dy * dy)
      this._lastX = (e.touches[0]!.clientX + e.touches[1]!.clientX) / 2
      this._lastY = (e.touches[0]!.clientY + e.touches[1]!.clientY) / 2
    }
  }

  private _onTouchMove = (e: TouchEvent) => {
    if (!this.enabled) return
    e.preventDefault()

    if (e.touches.length === 1 && this._touchCount === 1) {
      // Orbit
      const dx = e.touches[0]!.clientX - this._lastX
      const dy = e.touches[0]!.clientY - this._lastY
      this._lastX = e.touches[0]!.clientX
      this._lastY = e.touches[0]!.clientY
      this._velocityAz -= dx * 0.005
      this._velocityEl += dy * 0.005
    } else if (e.touches.length === 2) {
      // Pinch zoom
      const dx = e.touches[1]!.clientX - e.touches[0]!.clientX
      const dy = e.touches[1]!.clientY - e.touches[0]!.clientY
      const dist = Math.sqrt(dx * dx + dy * dy)
      const delta = this._lastPinchDist - dist
      this._lastPinchDist = dist
      this._velocityDist += delta * 0.02 * (this.distance * 0.1)

      // Pan
      const mx = (e.touches[0]!.clientX + e.touches[1]!.clientX) / 2
      const my = (e.touches[0]!.clientY + e.touches[1]!.clientY) / 2
      const panDx = mx - this._lastX
      const panDy = my - this._lastY
      this._lastX = mx
      this._lastY = my
      this._velocityPanX += panDx * 2
      this._velocityPanY += panDy * 2
    }
  }

  private _onTouchEnd = () => {
    this._touchCount = 0
  }
}

export const createOrbitControls = (
  camera: PerspectiveCamera,
  canvas: HTMLCanvasElement,
  opts?: OrbitControlsOptions,
): OrbitControls => new OrbitControls(camera, canvas, opts)
