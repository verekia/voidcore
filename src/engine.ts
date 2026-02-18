import { createRenderer, type Renderer, type RendererConfig, type FrameStats } from './renderer/renderer.ts'

import type { PerspectiveCamera } from './scene/camera.ts'
import type { Scene } from './scene/scene.ts'

export interface EngineConfig extends RendererConfig {
  backend?: 'auto' | 'webgpu' | 'webgl2'
  shadows?: boolean | object
  debug?: boolean
}

export class Engine {
  renderer: Renderer
  backend: 'webgpu' | 'webgl2'
  canvas: HTMLCanvasElement

  private _frameCallbacks: ((dt: number, elapsed: number) => void)[] = []
  private _running = false
  private _rafId = 0
  private _startTime = 0
  private _lastTime = 0

  constructor(canvas: HTMLCanvasElement, renderer: Renderer) {
    this.canvas = canvas
    this.renderer = renderer
    this.backend = renderer.backend
  }

  onFrame(callback: (deltaTime: number, elapsed: number) => void) {
    this._frameCallbacks.push(callback)
  }

  start(scene: Scene, camera: PerspectiveCamera) {
    if (this._running) return
    this._running = true
    this._startTime = performance.now() / 1000
    this._lastTime = this._startTime

    const loop = () => {
      if (!this._running) return

      const now = performance.now() / 1000
      const dt = Math.min(now - this._lastTime, 0.1) // Cap dt at 100ms
      const elapsed = now - this._startTime
      this._lastTime = now

      for (const cb of this._frameCallbacks) cb(dt, elapsed)

      this.renderer.render(scene, camera)

      this._rafId = requestAnimationFrame(loop)
    }
    this._rafId = requestAnimationFrame(loop)
  }

  stop() {
    this._running = false
    cancelAnimationFrame(this._rafId)
  }

  render(scene: Scene, camera: PerspectiveCamera) {
    this.renderer.render(scene, camera)
  }

  getStats(): FrameStats {
    return this.renderer.stats
  }

  dispose() {
    this.stop()
    this.renderer.dispose()
  }
}

export const createEngine = async (canvas: HTMLCanvasElement, config: EngineConfig = {}): Promise<Engine> => {
  const renderer = await createRenderer(canvas, config)
  return new Engine(canvas, renderer)
}
