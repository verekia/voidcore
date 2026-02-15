import type { Camera } from './camera.ts'

export class OrbitControls {
  azimuth: number
  elevation: number
  distance: number

  target = new Float32Array(3)

  rotateSensitivity = 0.005
  zoomSensitivity = 0.001
  panSensitivity = 0.001

  minElevation = -Math.PI / 2 + 0.05
  maxElevation = Math.PI / 2 - 0.05
  minDistance = 1
  maxDistance = 1000

  private camera: Camera
  private canvas: HTMLCanvasElement
  private isDragging = false
  private isPanning = false
  private lastX = 0
  private lastY = 0

  private pointers = new Map<number, { x: number; y: number }>()
  private lastPinchDist = 0

  constructor(camera: Camera, canvas: HTMLCanvasElement) {
    this.camera = camera
    this.canvas = canvas

    // Derive spherical coords from current camera position
    this.target[0] = camera.target[0]!
    this.target[1] = camera.target[1]!
    this.target[2] = camera.target[2]!

    const dx = camera.eye[0]! - this.target[0]!
    const dy = camera.eye[1]! - this.target[1]!
    const dz = camera.eye[2]! - this.target[2]!

    this.distance = Math.sqrt(dx * dx + dy * dy + dz * dz)
    const horizontal = Math.sqrt(dx * dx + dy * dy)
    this.elevation = Math.atan2(dz, horizontal)
    this.azimuth = Math.atan2(dx, dy)

    this.setupEvents()
    this.update()
  }

  private setupEvents() {
    this.canvas.addEventListener('pointerdown', this.onPointerDown)
    this.canvas.addEventListener('pointermove', this.onPointerMove)
    this.canvas.addEventListener('pointerup', this.onPointerUp)
    this.canvas.addEventListener('pointercancel', this.onPointerUp)
    this.canvas.addEventListener('wheel', this.onWheel, { passive: false })
    this.canvas.addEventListener('contextmenu', this.onContextMenu)
    this.canvas.style.touchAction = 'none'
  }

  private onContextMenu = (e: Event) => e.preventDefault()

  private onPointerDown = (e: PointerEvent) => {
    this.canvas.setPointerCapture(e.pointerId)
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY })

    if (this.pointers.size === 1) {
      this.isPanning = e.button === 2
      this.isDragging = true
      this.lastX = e.clientX
      this.lastY = e.clientY
    } else if (this.pointers.size === 2) {
      this.isDragging = false
      this.isPanning = true
      const pts = [...this.pointers.values()]
      const pdx = pts[1]!.x - pts[0]!.x
      const pdy = pts[1]!.y - pts[0]!.y
      this.lastPinchDist = Math.sqrt(pdx * pdx + pdy * pdy)
      this.lastX = (pts[0]!.x + pts[1]!.x) / 2
      this.lastY = (pts[0]!.y + pts[1]!.y) / 2
    }
  }

  private onPointerMove = (e: PointerEvent) => {
    if (!this.pointers.has(e.pointerId)) return
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY })

    if (this.pointers.size === 2) {
      const pts = [...this.pointers.values()]
      const pdx = pts[1]!.x - pts[0]!.x
      const pdy = pts[1]!.y - pts[0]!.y
      const pinchDist = Math.sqrt(pdx * pdx + pdy * pdy)

      if (this.lastPinchDist > 0) {
        const ratio = this.lastPinchDist / pinchDist
        this.distance = Math.max(this.minDistance, Math.min(this.maxDistance, this.distance * ratio))
      }
      this.lastPinchDist = pinchDist

      const midX = (pts[0]!.x + pts[1]!.x) / 2
      const midY = (pts[0]!.y + pts[1]!.y) / 2
      this.pan(midX - this.lastX, midY - this.lastY)
      this.lastX = midX
      this.lastY = midY
      this.update()
      return
    }

    if (!this.isDragging) return

    const movX = e.clientX - this.lastX
    const movY = e.clientY - this.lastY
    this.lastX = e.clientX
    this.lastY = e.clientY

    if (this.isPanning) {
      this.pan(movX, movY)
    } else {
      this.azimuth += movX * this.rotateSensitivity
      this.elevation += movY * this.rotateSensitivity
      this.elevation = Math.max(this.minElevation, Math.min(this.maxElevation, this.elevation))
    }

    this.update()
  }

  private onPointerUp = (e: PointerEvent) => {
    this.pointers.delete(e.pointerId)

    if (this.pointers.size === 0) {
      this.isDragging = false
      this.isPanning = false
    } else if (this.pointers.size === 1) {
      const remaining = [...this.pointers.values()][0]!
      this.lastX = remaining.x
      this.lastY = remaining.y
      this.isDragging = true
      this.isPanning = false
      this.lastPinchDist = 0
    }
  }

  private onWheel = (e: WheelEvent) => {
    e.preventDefault()
    const zoomFactor = 1 + e.deltaY * this.zoomSensitivity
    this.distance = Math.max(this.minDistance, Math.min(this.maxDistance, this.distance * zoomFactor))
    this.update()
  }

  private pan(screenDx: number, screenDy: number) {
    const panScale = this.distance * this.panSensitivity

    const cosAz = Math.cos(this.azimuth)
    const sinAz = Math.sin(this.azimuth)
    const cosEl = Math.cos(this.elevation)
    const sinEl = Math.sin(this.elevation)

    // Right vector (perpendicular to look direction in XY plane)
    const rightX = cosAz
    const rightY = -sinAz

    // Up vector (screen up mapped to world space)
    const upX = -sinAz * sinEl
    const upY = -cosAz * sinEl
    const upZ = cosEl

    const dx = screenDx * panScale
    const dy = screenDy * panScale

    this.target[0] = this.target[0]! + rightX * dx + upX * dy
    this.target[1] = this.target[1]! + rightY * dx + upY * dy
    this.target[2] = this.target[2]! + upZ * dy
  }

  update() {
    const cosEl = Math.cos(this.elevation)
    const sinEl = Math.sin(this.elevation)
    const cosAz = Math.cos(this.azimuth)
    const sinAz = Math.sin(this.azimuth)

    this.camera.eye[0] = this.target[0]! + this.distance * cosEl * sinAz
    this.camera.eye[1] = this.target[1]! + this.distance * cosEl * cosAz
    this.camera.eye[2] = this.target[2]! + this.distance * sinEl

    this.camera.target[0] = this.target[0]!
    this.camera.target[1] = this.target[1]!
    this.camera.target[2] = this.target[2]!
  }

  dispose() {
    this.canvas.removeEventListener('pointerdown', this.onPointerDown)
    this.canvas.removeEventListener('pointermove', this.onPointerMove)
    this.canvas.removeEventListener('pointerup', this.onPointerUp)
    this.canvas.removeEventListener('pointercancel', this.onPointerUp)
    this.canvas.removeEventListener('wheel', this.onWheel)
    this.canvas.removeEventListener('contextmenu', this.onContextMenu)
  }
}
