import { createWebGPURenderer } from './renderer.ts'
import { createWebGLRenderer } from './webgl-renderer.ts'

import type { Renderer } from './renderer.ts'

export type Backend = 'webgpu' | 'webgl'

export async function createRenderer(canvas: HTMLCanvasElement, backend?: Backend): Promise<Renderer> {
  if (backend === 'webgl') {
    return createWebGLRenderer(canvas)
  }

  if (backend === 'webgpu' || navigator.gpu) {
    try {
      return await createWebGPURenderer(canvas)
    } catch {
      // Fall through to WebGL
    }
  }

  return createWebGLRenderer(canvas)
}

export type { Renderer }
