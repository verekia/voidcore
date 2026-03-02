// Hooks – Public React hooks for VoidCore's declarative API.
//
// useEngine()       – Access engine, scene, camera, and canvas (works outside <Canvas>).
// useFrame(cb)      – Register a callback that runs every frame (works outside <Canvas>).
// useLoader(fn,url) – Suspense-compatible asset loader with caching.
// useGLTF(url)      – Load a glTF/GLB model (wrapper around useLoader).
//   With { meshName } – Returns a single Mesh by name instead of the full result.
//   With { meshName, clone: true } – Returns a ClonedMesh with its own bone hierarchy.
//   .setDecoderPath  – Set global default Draco/KTX2 decoder path.
// useKTX2(url)      – Load a KTX2 texture (wrapper around useLoader).
//   .setTranscoderPath – Set global default Basis transcoder path.
// useColoredGeometry(geometry, palette) – Memoized bakePalette for vertex-colored geometry.
// useColoredStaticGeometry(name, palette) – Load a named mesh from the static bundle and bake palette.
//   .setStaticBundlePath – Set global static bundle GLB path.
// useAnimations()   – Create an AnimationMixer and return action map.
// useGrass(geo, palette, opts) – Generate grass blades on grass-marked terrain faces.

import { useContext, useEffect, useRef, useMemo } from 'react'

import { AnimationMixer } from '../animation/index'
import { bakePalette } from '../geometry/geometry'
import { generateGrass, createGrassMaterial } from '../geometry/grass'
import { loadGLTF } from '../loaders/gltf'
import { loadKTX2 } from '../loaders/ktx2'
import { cloneScene } from '../scene/clone'
import { Mesh } from '../scene/mesh'
import { VoidContext, getGlobalStore } from './context'

import type { AnimationClip } from '../animation/index'
import type { Skeleton } from '../animation/skeleton'
import type { Geometry } from '../geometry/geometry'
import type { GrassOptions } from '../geometry/grass'
import type { GLTFResult, LoadOptions } from '../loaders/gltf'
import type { BasicMaterial, PaletteEntry } from '../materials/material'
import type { CompressedTextureFormat } from '../materials/texture'
import type { Texture } from '../materials/texture'
import type { Node } from '../scene/node'
import type { FrameCallback } from './context'

// ─── useEngine ────────────────────────────────────────────────────────────────

export const useEngine = () => {
  const contextStore = useContext(VoidContext)
  const store = contextStore ?? getGlobalStore()
  if (!store) throw new Error('useEngine: no VoidCore engine found — use inside <Canvas> or ensure <Canvas> is mounted')
  return { engine: store.engine, scene: store.scene, camera: store.camera, canvas: store.canvas }
}

// ─── useFrame ─────────────────────────────────────────────────────────────────

export const useFrame = (callback: FrameCallback) => {
  const contextStore = useContext(VoidContext)
  const store = contextStore ?? getGlobalStore()
  if (!store) throw new Error('useFrame: no VoidCore engine found — use inside <Canvas> or ensure <Canvas> is mounted')

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

export interface UseGLTFOptions extends LoadOptions {
  /** When set, useGLTF returns the first Mesh whose name matches instead of the full GLTFResult. */
  meshName?: string
  /** When set together with meshName, returns a ClonedMesh with its own bone hierarchy and skeleton. */
  clone?: boolean
}

export interface ClonedMesh {
  root: Node
  mesh: Mesh
  skeleton: Skeleton
  nodeMap: Map<Node, Node>
  animations: AnimationClip[]
}

interface UseGLTF {
  (url: string, options: UseGLTFOptions & { meshName: string; clone: true }): ClonedMesh
  (url: string, options: UseGLTFOptions & { meshName: string }): Mesh
  (url: string, options?: UseGLTFOptions): GLTFResult
  /** Set global default decoder path (Draco + KTX2) for all useGLTF calls. */
  setDecoderPath: (decoderPath: string) => void
  /** Preload a glTF/GLB into the cache so useGLTF won't suspend. */
  preload: (url: string, options?: LoadOptions) => void
}

export const useGLTF: UseGLTF = ((url: string, options?: UseGLTFOptions): GLTFResult | Mesh | ClonedMesh => {
  const resolved: UseGLTFOptions = { ..._defaultGLTFOptions, ...options }
  const gltf = useLoader((u: string, opts?: LoadOptions) => loadGLTF(u, opts), url, resolved)

  const cloned = useMemo(() => {
    if (!resolved.clone || !resolved.meshName) return null
    const { root, skeletons, nodeMap } = cloneScene(gltf.scene, gltf.skeletons, {
      meshFilter: m => m.name === resolved.meshName,
    })
    const mesh = Array.from(nodeMap.values()).find((n): n is Mesh => n instanceof Mesh)
    if (!mesh) throw new Error(`useGLTF: mesh "${resolved.meshName}" not found in clone of ${url}`)
    return { root, mesh, skeleton: skeletons[0]!, nodeMap, animations: gltf.animations } as ClonedMesh
  }, [gltf, resolved.clone, resolved.meshName])

  if (cloned) return cloned

  if (resolved.meshName) {
    const mesh = gltf.meshes.find(m => m.name === resolved.meshName)
    if (!mesh) throw new Error(`useGLTF: mesh "${resolved.meshName}" not found in ${url}`)
    return mesh
  }

  return gltf
}) as UseGLTF

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

// ─── useColoredGeometry ───────────────────────────────────────────────────────

export const useColoredGeometry = (geometry: Geometry, palette: PaletteEntry[]): Geometry => {
  return useMemo(() => bakePalette(geometry, palette), [geometry, palette])
}

// ─── useColoredStaticGeometry ─────────────────────────────────────────────────────

let _staticBundlePath: string | undefined

interface UseColoredStaticMesh {
  (meshName: string, palette: PaletteEntry[]): Geometry
  /** Set the global static bundle GLB path for all useColoredStaticGeometry calls. */
  setStaticBundlePath: (path: string) => void
}

export const useColoredStaticGeometry: UseColoredStaticMesh = ((
  meshName: string,
  palette: PaletteEntry[],
): Geometry => {
  if (!_staticBundlePath)
    throw new Error(
      'useColoredStaticGeometry requires a static bundle path — call useColoredStaticGeometry.setStaticBundlePath() first',
    )
  const mesh = useGLTF(_staticBundlePath, { meshName }) as Mesh
  return useColoredGeometry(mesh.geometry, palette)
}) as UseColoredStaticMesh

useColoredStaticGeometry.setStaticBundlePath = (path: string) => {
  _staticBundlePath = path
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

// ─── useGrass ─────────────────────────────────────────────────────────────────

export const useGrass = (
  geometry: Geometry,
  palette: PaletteEntry[],
  options?: GrassOptions,
): { geometry: Geometry; material: BasicMaterial } =>
  useMemo(() => {
    const grassGeo = generateGrass(geometry, palette, options)
    return { geometry: grassGeo, material: createGrassMaterial(palette, options) }
  }, [geometry, palette, options])
