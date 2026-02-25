// Geometry Worker Manager – Main-thread API for offloading geometry work to a web worker.
//
// Provides async versions of CPU-intensive geometry operations (BVH construction, palette
// baking, mesh merging, smooth normals). When a worker is initialized via initGeometryWorker(),
// operations run off the main thread. Without a worker, all operations fall back to synchronous
// execution on the main thread — the library always works, the worker is an optimization.
//
// initGeometryWorker(worker)     – Set the worker instance (call once at startup).
// terminateGeometryWorker()      – Terminate the worker and release resources.
// buildBVHAsync(pos, idx)        – Build a BVH for raycasting (off main thread).
// prebuildBVHAsync(geometry)     – Pre-build and cache a BVH for a geometry.
// bakePaletteAsync(geo, palette) – Bake palette entries into vertex colors (off main thread).
// mergeStaticIntoSkinnedAsync()  – Merge static geometry into skinned mesh (off main thread).
// computeSmoothNormalsAsync(geo) – Compute averaged normals for outlines (off main thread).
// mergeGeometriesAsync(geos)     – Merge multiple geometries into one (off main thread).

import {
  Geometry,
  bakePalette,
  mergeStaticIntoSkinned,
  computeSmoothNormals,
  mergeGeometries,
} from '../geometry/geometry'
import { buildMeshBVH, prebuildBVH, setBVH } from '../raycasting/index'

import type { PaletteEntry } from '../materials/material'
import type { Texture } from '../materials/texture'
import type { Mat4 } from '../math/index'
import type { MeshBVH } from '../raycasting/index'

// ─── Worker State ────────────────────────────────────────────────────────────

let _worker: Worker | null = null
let _nextId = 0
const _pending = new Map<number, { resolve: (value: any) => void; reject: (error: any) => void }>()

/** Initialize the geometry worker. Call once at startup with a Worker instance. */
export const initGeometryWorker = (worker: Worker): void => {
  _worker = worker
  _worker.onmessage = (e: MessageEvent) => {
    const { id } = e.data
    const entry = _pending.get(id)
    if (entry) {
      _pending.delete(id)
      entry.resolve(e.data)
    }
  }
  _worker.onerror = () => {
    // Worker failed to load — null it out so future calls fall back to synchronous execution
    _worker = null
    // Reject pending operations (Suspense cache will retry with sync fallback on next render)
    for (const [, entry] of _pending) {
      entry.reject(new Error('Geometry worker failed'))
    }
    _pending.clear()
  }
}

/** Terminate the geometry worker and release resources. */
export const terminateGeometryWorker = (): void => {
  if (_worker) {
    _worker.terminate()
    _worker = null
    for (const [, entry] of _pending) {
      entry.reject(new Error('Geometry worker terminated'))
    }
    _pending.clear()
  }
}

/** Returns true if the geometry worker has been initialized. */
export const hasGeometryWorker = (): boolean => _worker !== null

function postToWorker(msg: Record<string, unknown>): Promise<any> {
  return new Promise((resolve, reject) => {
    const id = _nextId++
    _pending.set(id, { resolve, reject })
    _worker!.postMessage({ id, ...msg })
  })
}

// ─── Serialization Helpers ───────────────────────────────────────────────────

interface SerializedPaletteEntry {
  color: [number, number, number]
  emissive?: [number, number, number]
  emissiveIntensity?: number
  tiledAoLayerIndex: number
  tiledAoIntensity: number
  tiledNormalLayerIndex: number
  tiledNormalIntensity: number
  tiledNormalScale: number
  color2?: [number, number, number]
  noiseScale?: number
}

/** Pre-resolve texture layer indices on the main thread, return serialized palette + texture arrays. */
function serializePalette(palette: PaletteEntry[]): {
  entries: SerializedPaletteEntry[]
  hasTiledNormals: boolean
  hasNoiseColor: boolean
  tiledAoTextures: Texture[]
  tiledAoScales: number[]
  tiledNormalTextures: Texture[]
} {
  const tiledAoTextureMap = new Map<Texture, number>()
  const tiledAoTextures: Texture[] = []
  const tiledAoScales: number[] = []
  for (const entry of palette) {
    if (entry.tiledAo && !tiledAoTextureMap.has(entry.tiledAo)) {
      const layerIndex = tiledAoTextures.length + 1
      tiledAoTextureMap.set(entry.tiledAo, layerIndex)
      tiledAoTextures.push(entry.tiledAo)
      tiledAoScales.push(entry.tiledAoScale ?? 1.0)
    }
  }

  const tiledNormalTextureMap = new Map<Texture, number>()
  const tiledNormalTextures: Texture[] = []
  for (const entry of palette) {
    if (entry.tiledNormal && !tiledNormalTextureMap.has(entry.tiledNormal)) {
      const layerIndex = tiledNormalTextures.length + 1
      tiledNormalTextureMap.set(entry.tiledNormal, layerIndex)
      tiledNormalTextures.push(entry.tiledNormal)
    }
  }

  let hasNoiseColor = false
  for (const entry of palette) {
    if (entry.color2) {
      hasNoiseColor = true
      break
    }
  }

  const entries: SerializedPaletteEntry[] = palette.map(entry => ({
    color: entry.color,
    emissive: entry.emissive,
    emissiveIntensity: entry.emissiveIntensity,
    tiledAoLayerIndex: entry.tiledAo ? (tiledAoTextureMap.get(entry.tiledAo) ?? 0) : 0,
    tiledAoIntensity: entry.tiledAoIntensity ?? 1.0,
    tiledNormalLayerIndex: entry.tiledNormal ? (tiledNormalTextureMap.get(entry.tiledNormal) ?? 0) : 0,
    tiledNormalIntensity: entry.tiledNormalIntensity ?? 1.0,
    tiledNormalScale: entry.tiledNormalScale ?? 1.0,
    color2: entry.color2,
    noiseScale: entry.noiseScale,
  }))

  return {
    entries,
    hasTiledNormals: tiledNormalTextures.length > 0,
    hasNoiseColor,
    tiledAoTextures,
    tiledAoScales,
    tiledNormalTextures,
  }
}

interface GeometryArrays {
  positions: Float32Array
  normals: Float32Array
  indices: Uint16Array | Uint32Array
  uvs: Float32Array | null
  colors: Float32Array | null
  emissiveColors: Float32Array | null
  materialIndices: Uint8Array | null
  joints: Uint8Array | Uint16Array | null
  weights: Float32Array | null
  tiledNormalData: Float32Array | null
  noiseColorData: Float32Array | null
  vertexCount: number
  indexCount: number
}

function serializeGeometry(geo: Geometry): GeometryArrays {
  return {
    positions: geo.positions,
    normals: geo.normals,
    indices: geo.indices,
    uvs: geo.uvs ?? null,
    colors: geo.colors ?? null,
    emissiveColors: geo.emissiveColors ?? null,
    materialIndices: geo.materialIndices ?? null,
    joints: geo.joints ?? null,
    weights: geo.weights ?? null,
    tiledNormalData: geo.tiledNormalData ?? null,
    noiseColorData: geo.noiseColorData ?? null,
    vertexCount: geo.vertexCount,
    indexCount: geo.indexCount,
  }
}

// ─── Async API ───────────────────────────────────────────────────────────────

/**
 * Build a BVH (bounding volume hierarchy) for raycasting.
 * With a worker: runs off main thread. Without: runs synchronously.
 */
export const buildBVHAsync = async (positions: Float32Array, indices: Uint16Array | Uint32Array): Promise<MeshBVH> => {
  if (!_worker) {
    return buildMeshBVH(positions, indices)
  }

  const result = await postToWorker({ type: 'buildBVH', positions, indices })
  const floatNodes = new Float32Array(result.bvhBuffer)
  const intNodes = new Int32Array(result.bvhBuffer)
  return { floatNodes, intNodes, triOrder: result.triOrder }
}

/**
 * Pre-build and cache a BVH for a geometry (avoids first-raycast stall).
 * With a worker: runs off main thread. Without: runs synchronously.
 */
export const prebuildBVHAsync = async (geometry: Geometry): Promise<void> => {
  if (!_worker) {
    prebuildBVH(geometry)
    return
  }

  const bvh = await buildBVHAsync(geometry.positions, geometry.indices)
  setBVH(geometry, bvh)
}

/**
 * Bake palette entries into per-vertex colors and emissive attributes.
 * With a worker: runs off main thread. Without: runs synchronously.
 */
export const bakePaletteAsync = async (geometry: Geometry, palette: PaletteEntry[]): Promise<Geometry> => {
  if (!_worker) {
    return bakePalette(geometry, palette)
  }

  const { entries, hasTiledNormals, hasNoiseColor, tiledAoTextures, tiledAoScales, tiledNormalTextures } =
    serializePalette(palette)

  const result = await postToWorker({
    type: 'bakePalette',
    materialIndices: geometry.materialIndices ?? null,
    vertexCount: geometry.vertexCount,
    palette: entries,
    hasTiledNormals,
    hasNoiseColor,
  })

  // Reconstruct Geometry on main thread with worker-computed vertex data
  const bakedGeo = new Geometry({
    positions: geometry.positions,
    normals: geometry.normals,
    indices: geometry.indices,
    uvs: geometry.uvs,
    colors: result.colors,
    emissiveColors: result.emissiveColors,
    joints: geometry.joints,
    weights: geometry.weights,
  })

  // Attach texture references (kept on main thread, not sent to worker)
  if (tiledAoTextures.length > 0) {
    bakedGeo.tiledAoTextures = tiledAoTextures
    bakedGeo.tiledAoScales = new Float32Array(tiledAoScales)
  }
  if (tiledNormalTextures.length > 0 && result.tiledNormalData) {
    bakedGeo.tiledNormalTextures = tiledNormalTextures
    bakedGeo.tiledNormalData = result.tiledNormalData
  }
  if (result.noiseColorData) {
    bakedGeo.noiseColorData = result.noiseColorData
  }

  return bakedGeo
}

/**
 * Merge a static geometry into a skinned geometry at a bone.
 * With a worker: runs off main thread. Without: runs synchronously.
 */
export const mergeStaticIntoSkinnedAsync = async (
  skinned: Geometry,
  staticGeo: Geometry,
  boneIndex: number,
  inverseBindMatrix: Mat4,
  localTransform?: Mat4,
): Promise<Geometry> => {
  if (!_worker) {
    return mergeStaticIntoSkinned(skinned, staticGeo, boneIndex, inverseBindMatrix, localTransform)
  }

  const result = await postToWorker({
    type: 'mergeStaticIntoSkinned',
    skinned: serializeGeometry(skinned),
    staticGeo: serializeGeometry(staticGeo),
    boneIndex,
    inverseBindMatrix,
    localTransform: localTransform ?? null,
  })

  const merged = new Geometry({
    positions: result.positions,
    normals: result.normals,
    indices: result.indices,
    uvs: result.uvs ?? undefined,
    colors: result.colors ?? undefined,
    emissiveColors: result.emissiveColors ?? undefined,
    joints: result.joints ?? undefined,
    weights: result.weights ?? undefined,
  })

  // Propagate texture references from sources
  const tiledSrc = skinned.tiledAoTextures ? skinned : staticGeo.tiledAoTextures ? staticGeo : null
  if (tiledSrc) {
    merged.tiledAoTextures = tiledSrc.tiledAoTextures
    merged.tiledAoScales = tiledSrc.tiledAoScales
  }
  if (result.tiledNormalData) {
    merged.tiledNormalData = result.tiledNormalData
    const normalSrc = skinned.tiledNormalTextures ? skinned : staticGeo.tiledNormalTextures ? staticGeo : null
    if (normalSrc) merged.tiledNormalTextures = normalSrc.tiledNormalTextures
  }
  if (result.noiseColorData) {
    merged.noiseColorData = result.noiseColorData
  }

  return merged
}

/**
 * Compute position-averaged smooth normals for gap-free inverted hull outlines.
 * With a worker: runs off main thread. Without: runs synchronously.
 */
export const computeSmoothNormalsAsync = async (geometry: Geometry): Promise<Float32Array> => {
  if (!_worker) {
    return computeSmoothNormals(geometry)
  }

  const result = await postToWorker({
    type: 'computeSmoothNormals',
    positions: geometry.positions,
    normals: geometry.normals,
    vertexCount: geometry.vertexCount,
  })

  geometry._smoothNormals = result.smoothNormals
  return result.smoothNormals
}

/**
 * Merge multiple geometries into one.
 * With a worker: runs off main thread. Without: runs synchronously.
 */
export const mergeGeometriesAsync = async (geometries: Geometry[]): Promise<Geometry> => {
  if (!_worker) {
    return mergeGeometries(geometries)
  }

  const result = await postToWorker({
    type: 'mergeGeometries',
    geometries: geometries.map(serializeGeometry),
  })

  const merged = new Geometry({
    positions: result.positions,
    normals: result.normals,
    indices: result.indices,
    uvs: result.uvs ?? undefined,
    colors: result.colors ?? undefined,
    emissiveColors: result.emissiveColors ?? undefined,
    materialIndices: result.materialIndices ?? undefined,
  })

  // Propagate texture references from sources
  for (const geo of geometries) {
    if (geo.tiledAoTextures) {
      merged.tiledAoTextures = geo.tiledAoTextures
      merged.tiledAoScales = geo.tiledAoScales
      break
    }
  }
  if (result.tiledNormalData) {
    merged.tiledNormalData = result.tiledNormalData
    for (const geo of geometries) {
      if (geo.tiledNormalTextures) {
        merged.tiledNormalTextures = geo.tiledNormalTextures
        break
      }
    }
  }
  if (result.noiseColorData) {
    merged.noiseColorData = result.noiseColorData
  }

  return merged
}
