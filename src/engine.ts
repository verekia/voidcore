// Engine – The main entry point that ties everything together.
//
// The Engine owns the scheduler: a single requestAnimationFrame loop that drives the
// entire application. Callbacks are registered with priorities (lower runs first) and
// optional per-callback FPS throttling. A global FPS cap can limit the overall tick rate.
//
// createEngine()      – Async factory that picks the best available renderer (WebGPU or
//                       WebGL2) and returns an Engine instance.
// engine.register()   – Adds a callback to the scheduler with priority and FPS options.
//                       Returns an unsubscribe function.
// engine.start()      – Begins the scheduler loop.
// engine.stop()       – Pauses the scheduler loop.
// engine.maxFps       – Global FPS cap (0 = uncapped).
// engine.maxDpr       – Max device pixel ratio (default 1.25 on mobile, 1.5 on desktop).
// engine.render()     – Renders a single frame (call this inside a registered callback).
// engine.dispose()    – Cleans up the scheduler and GPU resources.

import { createRenderer, type Renderer, type RendererConfig, type FrameStats } from './renderer/renderer.ts'
import { Scheduler, type SchedulerCallback, type SchedulerCallbackOptions } from './scheduler.ts'

import type { ShadowConfig } from './renderer/renderer.ts'
import type { PerspectiveCamera } from './scene/camera.ts'
import type { Scene } from './scene/scene.ts'

export interface EngineConfig extends RendererConfig {
  backend?: 'auto' | 'webgpu' | 'webgl2'
  shadows?: boolean | ShadowConfig
  debug?: boolean
}

export class Engine {
  renderer: Renderer
  backend: 'webgpu' | 'webgl2'
  canvas: HTMLCanvasElement
  scheduler: Scheduler

  constructor(canvas: HTMLCanvasElement, renderer: Renderer) {
    this.canvas = canvas
    this.renderer = renderer
    this.backend = renderer.backend
    this.scheduler = new Scheduler()
  }

  /** Global FPS cap applied to the entire loop. 0 = uncapped (run every rAF). */
  get maxFps(): number {
    return this.scheduler.maxFps
  }

  set maxFps(fps: number) {
    this.scheduler.maxFps = fps
  }

  /** Maximum device pixel ratio. Caps resolution scaling to save GPU on high-density displays. */
  get maxDpr(): number {
    return this.renderer.maxDpr
  }

  set maxDpr(dpr: number) {
    this.renderer.maxDpr = dpr
  }

  /**
   * Register a callback to run each frame (or throttled via `fps`).
   * Callbacks execute in priority order (lower first, can be negative).
   * Returns an unsubscribe function.
   */
  register(callback: SchedulerCallback, options?: SchedulerCallbackOptions): () => void {
    return this.scheduler.register(callback, options)
  }

  /** Start the scheduler loop. No-op if already running. */
  start() {
    this.scheduler.start()
  }

  /** Stop the scheduler loop. Can be resumed with start(). */
  stop() {
    this.scheduler.stop()
  }

  /** Render a single frame. Call this inside a registered callback. */
  render(scene: Scene, camera: PerspectiveCamera) {
    this.renderer.render(scene, camera)
  }

  getStats(): FrameStats {
    return this.renderer.stats
  }

  dispose() {
    this.scheduler.destroy()
    this.renderer.dispose()
  }
}

export const createEngine = async (canvas: HTMLCanvasElement, config: EngineConfig = {}): Promise<Engine> => {
  const renderer = await createRenderer(canvas, config)
  return new Engine(canvas, renderer)
}
