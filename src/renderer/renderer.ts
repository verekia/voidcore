import type { PerspectiveCamera } from '../scene/camera.ts'
import type { Scene } from '../scene/scene.ts'

export interface RendererConfig {
  antialias?: boolean
  bloom?: boolean | { intensity?: number; levels?: number }
}

export interface FrameStats {
  fps: number
  frameTime: number
  drawCalls: number
  triangles: number
  visibleObjects: number
  culledObjects: number
}

export interface Renderer {
  readonly backend: 'webgpu' | 'webgl2'
  render(scene: Scene, camera: PerspectiveCamera): void
  dispose(): void
  stats: FrameStats
}

export const createRenderer = async (
  canvas: HTMLCanvasElement,
  config: RendererConfig & { backend?: 'auto' | 'webgpu' | 'webgl2' },
): Promise<Renderer> => {
  const backend = config.backend ?? 'auto'

  if (backend === 'webgpu' || backend === 'auto') {
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
      try {
        const { WebGPURenderer } = await import('./webgpu.ts')
        const renderer = await WebGPURenderer.create(canvas, config)
        console.log('[voidcore] Using WebGPU renderer')
        return renderer
      } catch (e) {
        if (backend === 'webgpu') throw e
        console.warn('[voidcore] WebGPU init failed, falling back to WebGL2:', (e as Error).message)
      }
    } else if (backend === 'webgpu') {
      throw new Error('WebGPU not available in this browser')
    }
  }

  const { WebGL2Renderer } = await import('./webgl2.ts')
  console.log('[voidcore] Using WebGL2 renderer')
  return new WebGL2Renderer(canvas, config)
}
