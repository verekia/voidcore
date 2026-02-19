// Context – React context for sharing VoidCore engine state across the component tree.
//
// Holds the engine, scene, camera, canvas ref, raycaster, pointer state, and the set
// of per-frame callbacks registered by useFrame hooks.

import { createContext } from 'react'

import type { Engine } from '../engine.ts'
import type { Raycaster } from '../raycasting/index.ts'
import type { PerspectiveCamera } from '../scene/camera.ts'
import type { Scene } from '../scene/scene.ts'
import type { Tunnel } from './tunnel.ts'

export interface FrameCallback {
  (state: { dt: number; elapsed: number; frame: number; engine: Engine; scene: Scene; camera: PerspectiveCamera }): void
}

export interface VoidStore {
  engine: Engine
  scene: Scene
  camera: PerspectiveCamera
  canvas: HTMLCanvasElement
  raycaster: Raycaster
  pointer: { x: number; y: number }
  frameCallbacks: Set<FrameCallback>
  tunnel: Tunnel
}

export const VoidContext = createContext<VoidStore | null>(null)
