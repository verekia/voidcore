// Hooks – Public React hooks for VoidCore's declarative API.
//
// useEngine()       – Access engine, scene, camera, and canvas from context.
// useFrame(cb)      – Register a callback that runs every frame (before render).
// useLoader(fn,url) – Suspense-compatible asset loader with caching.
// useGLTF(url)      – Load a glTF/GLB model (wrapper around useLoader).
//   .setDecoderPath  – Set global default Draco/KTX2 decoder path.
// useKTX2(url)      – Load a KTX2 texture (wrapper around useLoader).
//   .setTranscoderPath – Set global default Basis transcoder path.
// useAnimations()   – Create an AnimationMixer and return action map.

import { useContext, useEffect, useRef, useMemo } from 'react'

import { AnimationMixer } from '../animation/index'
import { loadGLTF } from '../loaders/gltf'
import { loadKTX2 } from '../loaders/ktx2'
import { VoidContext } from './context'

import type { AnimationClip } from '../animation/index'
import type { Skeleton } from '../animation/skeleton'
import type { GLTFResult, LoadOptions } from '../loaders/gltf'
import type { CompressedTextureFormat } from '../materials/texture'
import type { Texture } from '../materials/texture'
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

let _defaultGLTFOptions: LoadOptions | undefined

interface UseGLTF {
  (url: string, options?: LoadOptions): GLTFResult
  /** Set global default decoder path (Draco + KTX2) for all useGLTF calls. */
  setDecoderPath: (decoderPath: string) => void
  /** Preload a glTF/GLB into the cache so useGLTF won't suspend. */
  preload: (url: string, options?: LoadOptions) => void
}

export const useGLTF: UseGLTF = (url: string, options?: LoadOptions): GLTFResult => {
  const resolved = options ?? _defaultGLTFOptions
  return useLoader((u: string, opts?: LoadOptions) => loadGLTF(u, opts), url, resolved)
}

useGLTF.setDecoderPath = (decoderPath: string) => {
  _defaultGLTFOptions = { draco: { decoderPath }, ktx2: { transcoderPath: decoderPath } }
}

useGLTF.preload = (url: string, options?: LoadOptions) => {
  if (typeof window === 'undefined') return
  const resolved = options ?? _defaultGLTFOptions
  const cacheKey = url
  if (loaderCache.has(cacheKey)) return
  const entry: CacheEntry<GLTFResult> = {}
  loaderCache.set(cacheKey, entry)
  const promise = loadGLTF(url, resolved)
    .then(result => {
      entry.result = result
    })
    .catch(error => {
      entry.error = error
    })
  entry.promise = promise as unknown as Promise<GLTFResult>
}

// ─── useKTX2 ─────────────────────────────────────────────────────────────────

let _defaultTranscoderPath: string | undefined

interface UseKTX2 {
  (url: string, transcoderPath?: string): Texture
  /** Set global default Basis transcoder path for all useKTX2 calls. */
  setTranscoderPath: (transcoderPath: string) => void
}

export const useKTX2: UseKTX2 = (url: string, transcoderPath?: string): Texture => {
  const resolved = transcoderPath ?? _defaultTranscoderPath
  if (!resolved)
    throw new Error('useKTX2 requires a transcoderPath — pass it directly or call useKTX2.setTranscoderPath()')
  const store = useContext(VoidContext)
  if (!store) throw new Error('useKTX2 must be used inside <Canvas>')
  const formats = store.engine.compressedTextureFormats
  return useLoader(
    (u: string, tp: string, fmts: readonly CompressedTextureFormat[]) => loadKTX2(u, tp, fmts),
    url,
    resolved,
    formats,
  )
}

useKTX2.setTranscoderPath = (transcoderPath: string) => {
  _defaultTranscoderPath = transcoderPath
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
