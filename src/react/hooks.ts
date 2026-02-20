// Hooks – Public React hooks for VoidCore's declarative API.
//
// useEngine()       – Access engine, scene, camera, and canvas from context.
// useFrame(cb)      – Register a callback that runs every frame (before render).
// useLoader(fn,url) – Suspense-compatible asset loader with caching.
// useGLTF(url)      – Load a glTF/GLB model (wrapper around useLoader).
// useAnimations()   – Create an AnimationMixer and return action map.

import { useContext, useEffect, useRef, useMemo } from 'react'

import { AnimationMixer } from '../animation/index'
import { loadGLTF } from '../loaders/gltf'
import { VoidContext } from './context'

import type { AnimationClip } from '../animation/index'
import type { Skeleton } from '../animation/skeleton'
import type { GLTFResult, LoadOptions } from '../loaders/gltf'
import type { FrameCallback } from './context'

// ─── useEngine ────────────────────────────────────────────────────────────────

export const useEngine = () => {
  const store = useContext(VoidContext)
  if (!store) throw new Error('useEngine must be used inside <Canvas>')
  return { engine: store.engine, scene: store.scene, camera: store.camera, canvas: store.canvas }
}

// ─── useFrame ─────────────────────────────────────────────────────────────────

export const useFrame = (callback: FrameCallback) => {
  const store = useContext(VoidContext)
  if (!store) throw new Error('useFrame must be used inside <Canvas>')

  const callbackRef = useRef(callback)
  callbackRef.current = callback

  useEffect(() => {
    const wrapper: FrameCallback = state => callbackRef.current(state)
    store.frameCallbacks.add(wrapper)
    return () => {
      store.frameCallbacks.delete(wrapper)
    }
  }, [store])
}

// ─── useLoader ────────────────────────────────────────────────────────────────

interface CacheEntry<T> {
  promise?: Promise<T>
  result?: T
  error?: unknown
}

const loaderCache = new Map<string, CacheEntry<any>>()

export function useLoader<T>(loaderFn: (url: string, ...args: any[]) => Promise<T>, url: string, ...args: any[]): T {
  const cacheKey = url

  let entry = loaderCache.get(cacheKey) as CacheEntry<T> | undefined

  if (!entry) {
    entry = {} as CacheEntry<T>
    loaderCache.set(cacheKey, entry)
    const promise = loaderFn(url, ...args)
      .then(result => {
        entry!.result = result
      })
      .catch(error => {
        entry!.error = error
      })
    entry.promise = promise as unknown as Promise<T>
  }

  if (entry.error) throw entry.error
  if (entry.result !== undefined) return entry.result
  throw entry.promise
}

// ─── useGLTF ──────────────────────────────────────────────────────────────────

export const useGLTF = (url: string, options?: LoadOptions): GLTFResult => {
  return useLoader((u: string, opts?: LoadOptions) => loadGLTF(u, opts), url, options)
}

// Preload a glTF/GLB into the cache so useGLTF won't suspend.
// Call at module level to start parallel downloads before components render.
useGLTF.preload = (url: string, options?: LoadOptions) => {
  if (typeof window === 'undefined') return
  const cacheKey = url
  if (loaderCache.has(cacheKey)) return
  const entry: CacheEntry<GLTFResult> = {}
  loaderCache.set(cacheKey, entry)
  const promise = loadGLTF(url, options)
    .then(result => {
      entry.result = result
    })
    .catch(error => {
      entry.error = error
    })
  entry.promise = promise as unknown as Promise<GLTFResult>
}

// ─── useAnimations ────────────────────────────────────────────────────────────

export const useAnimations = (animations: AnimationClip[], skeleton: Skeleton) => {
  const mixer = useMemo(() => new AnimationMixer(skeleton), [skeleton])
  const actions = useMemo(() => {
    const map: Record<string, ReturnType<typeof mixer.clipAction>> = {}
    for (const clip of animations) {
      map[clip.name] = mixer.clipAction(clip)
    }
    return map
  }, [mixer, animations])

  useFrame(({ dt }) => {
    mixer.update(dt)
  })

  return { mixer, actions }
}
