// Renderer – Interface and factory for the rendering backend (WebGPU or WebGL2).
//
// The renderer is responsible for taking a scene and camera and producing pixels on screen.
// Voidcore supports two backends: WebGPU (modern, faster) and WebGL2 (wider compatibility).
// The `createRenderer` factory tries WebGPU first, falls back to WebGL2 if unavailable.
//
// Both backends implement the same Renderer interface so the rest of the engine doesn't
// need to know which one is active.
//
// createRenderer() – Async factory: tries WebGPU, falls back to WebGL2.
// renderer.render() – Draws a single frame.
// renderer.dispose() – Releases GPU resources.

import type { PerspectiveCamera } from '../scene/camera.ts'
import type { Scene } from '../scene/scene.ts'

export interface ShadowConfig {
  enabled?: boolean
  resolution?: number
  lambda?: number
  backExtend?: number
  constantBias?: number
  slopeBias?: number
  blendRange?: number
}

export interface RendererConfig {
  antialias?: boolean
  bloom?: boolean | { intensity?: number; levels?: number }
  shadows?: boolean | ShadowConfig
  maxDpr?: number | false
}

export interface FrameStats {
  fps: number
  frameTime: number
  drawCalls: number
  shadowDrawCalls: number
  triangles: number
  visibleObjects: number
  culledObjects: number
}

export interface Renderer {
  readonly backend: 'webgpu' | 'webgl2'
  maxDpr: number
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
