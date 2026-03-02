// Context – React context for sharing VoidCore engine state across the component tree.
//
// Holds the engine, scene, camera, canvas ref, raycaster, pointer state, and the set
// of per-frame callbacks registered by useFrame hooks. Also exposes a module-level
// global store reference so hooks like useFrame and useEngine can work outside <Canvas>.

import { createContext } from 'react'

import type { Engine } from '../engine'
import type { Raycaster } from '../raycasting/index'
import type { PerspectiveCamera } from '../scene/camera'
import type { Scene } from '../scene/scene'
import type { Tunnel } from './tunnel'

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

// Module-level store reference so hooks like useFrame can work outside <Canvas>.
// First Canvas to mount claims the global slot; later Canvases don't overwrite it.
// Only the Canvas that owns the slot clears it on unmount.
let _globalStore: VoidStore | null = null
export const claimGlobalStore = (store: VoidStore): boolean => {
  if (_globalStore) return false
  _globalStore = store
  return true
}
export const releaseGlobalStore = (store: VoidStore) => {
  if (_globalStore === store) _globalStore = null
}
export const getGlobalStore = () => _globalStore
