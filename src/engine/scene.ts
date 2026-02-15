import { Camera } from './camera.ts'
import { createRenderer, type Backend, type Renderer } from './gpu.ts'
import { Mesh } from './mesh.ts'
import { loadWasm } from './wasm.ts'

import type { Geometry } from './geometry.ts'
import type { WasmCore } from './wasm.ts'

const FLAG_VISIBLE = 0x02
const FLAG_UNLIT = 0x04

export interface SceneConfig {
  backend?: Backend
}

export class Scene {
  wasm!: WasmCore
  renderer!: Renderer
  camera!: Camera
  canvas: HTMLCanvasElement

  private meshes: (Mesh | null)[] = []
  private activeCount = 0
  private geometryRegistry = new Map<number, Geometry>()
  private nextGeometryId = 0
  private config: SceneConfig

  // Lighting
  private lightDir = new Float32Array([0.5, -1, 0.3])
  private lightColor = new Float32Array([1, 1, 1])
  private ambientColor = new Float32Array([0.15, 0.15, 0.15])

  // Stats
  visibleCount = 0
  drawCalls = 0

  // Scratch offsets for frustum planes (6 planes × 4 floats = 96 bytes)
  private planesOffset = 0

  private constructor(canvas: HTMLCanvasElement, config: SceneConfig) {
    this.canvas = canvas
    this.config = config
  }

  static async create(canvas: HTMLCanvasElement, config: SceneConfig = {}): Promise<Scene> {
    const scene = new Scene(canvas, config)
    scene.wasm = await loadWasm()
    scene.renderer = await createRenderer(canvas, config.backend)
    // Camera scratch starts after the main scratch offset
    scene.camera = new Camera(scene.wasm.scratchOffset)
    // Planes offset: after camera's 3 mat4s (192 bytes)
    scene.planesOffset = scene.wasm.scratchOffset + 192
    return scene
  }

  registerGeometry(geometry: Geometry): number {
    const id = this.nextGeometryId++
    this.geometryRegistry.set(id, geometry)
    this.renderer.registerGeometry(id, geometry.vertices, geometry.indices)
    return id
  }

  add(mesh: Mesh): Mesh {
    const entityId = this.activeCount++
    mesh._bind(this.wasm, entityId)
    this.meshes[entityId] = mesh

    // Compute bounding sphere from geometry
    const geo = this.geometryRegistry.get(mesh.geometryId)
    if (geo) {
      const { center, radius } = computeBoundingSphere(geo.vertices)
      mesh.bsphereRadius = radius
      mesh.bsphereCenterOffset.set(center)
      mesh.updateBsphere()
    }

    return mesh
  }

  remove(mesh: Mesh) {
    if (mesh.entityId < 0) return
    // Swap with last entity
    const lastIdx = this.activeCount - 1
    if (mesh.entityId !== lastIdx && lastIdx >= 0) {
      const lastMesh = this.meshes[lastIdx]
      if (lastMesh) {
        // Copy SoA data from last to removed slot
        copyEntityData(this.wasm, lastIdx, mesh.entityId)
        lastMesh._bind(this.wasm, mesh.entityId)
        this.meshes[mesh.entityId] = lastMesh
      }
    }
    this.meshes[lastIdx] = null
    mesh.entityId = -1
    this.activeCount--
  }

  setDirectionalLight(direction: [number, number, number], color: [number, number, number]) {
    this.lightDir.set(direction)
    this.lightColor.set(color)
  }

  setAmbientLight(color: [number, number, number]) {
    this.ambientColor.set(color)
  }

  render() {
    const { wasm, renderer, camera, activeCount } = this
    const aspect = this.canvas.width / this.canvas.height

    // 1. Compute world matrices for dirty entities
    wasm.exports.vc_compute_world_matrices(activeCount)

    // 2. Update camera
    camera.update(wasm, aspect)

    // 3. Update bounding spheres for all meshes
    for (let i = 0; i < activeCount; i++) {
      this.meshes[i]?.updateBsphere()
    }

    // 4. Frustum culling (if available in WASM)
    let visibleIndices: number[]
    let visibleCount: number

    if (wasm.exports.vc_extract_frustum_planes && wasm.exports.vc_frustum_cull) {
      // Extract frustum planes from VP matrix
      const vpOffset = wasm.scratchOffset + 128 // camera.vpOffset
      wasm.exports.vc_extract_frustum_planes(vpOffset, this.planesOffset)

      // Cull
      visibleCount = wasm.exports.vc_frustum_cull(
        activeCount,
        this.planesOffset,
        wasm.bspheresPtr,
        wasm.flagsPtr,
        wasm.visibleIndicesPtr,
      )

      // 5. Sort draw calls (if available)
      if (wasm.exports.vc_build_sort_keys && wasm.exports.vc_sort_draw_calls) {
        wasm.exports.vc_build_sort_keys(visibleCount, wasm.visibleIndicesPtr, wasm.geometryIdsPtr, wasm.sortKeysPtr)
        wasm.exports.vc_sort_draw_calls(visibleCount, wasm.sortKeysPtr, wasm.visibleIndicesPtr)
      }

      // Read visible indices
      visibleIndices = []
      const u32 = wasm.u32
      const base = wasm.visibleIndicesPtr / 4
      for (let i = 0; i < visibleCount; i++) {
        visibleIndices.push(u32[base + i]!)
      }
    } else {
      // No culling: draw all visible entities
      visibleIndices = []
      for (let i = 0; i < activeCount; i++) {
        const flags = wasm.u32[wasm.flagsPtr / 4 + i]!
        if (flags & FLAG_VISIBLE) {
          visibleIndices.push(i)
        }
      }
      visibleCount = visibleIndices.length
    }

    this.visibleCount = visibleCount

    // 6. Build draw entity list
    const drawEntities = []
    for (const idx of visibleIndices) {
      const mesh = this.meshes[idx]
      if (!mesh) continue
      drawEntities.push({
        worldMatrix: mesh.worldMatrix,
        color: mesh.color,
        geometryId: mesh.geometryId,
        unlit: (wasm.u32[wasm.flagsPtr / 4 + idx]! & FLAG_UNLIT) !== 0,
      })
    }

    // 7. Render
    renderer.updateCamera(camera.view, camera.projection)
    renderer.updateLighting(this.lightDir, this.lightColor, this.ambientColor)
    renderer.draw(drawEntities, drawEntities.length)
    this.drawCalls = drawEntities.length

    // 8. Reset per-frame arena
    wasm.exports.vc_frame_reset()
  }

  resize(width: number, height: number) {
    this.canvas.width = width
    this.canvas.height = height
    this.renderer.resize(width, height)
  }

  async switchBackend(canvas: HTMLCanvasElement, backend: Backend) {
    this.renderer.destroy()
    this.canvas = canvas
    this.renderer = await createRenderer(canvas, backend)

    // Re-register all geometries
    for (const [id, geo] of this.geometryRegistry) {
      this.renderer.registerGeometry(id, geo.vertices, geo.indices)
    }
  }

  destroy() {
    this.renderer.destroy()
  }
}

function copyEntityData(wasm: WasmCore, from: number, to: number) {
  const { f32, u32 } = wasm

  // positions (3 floats)
  const pFrom = wasm.positionsPtr / 4 + from * 3
  const pTo = wasm.positionsPtr / 4 + to * 3
  f32.copyWithin(pTo, pFrom, pFrom + 3)

  // euler rotations (3 floats)
  const rFrom = wasm.eulerRotationsPtr / 4 + from * 3
  const rTo = wasm.eulerRotationsPtr / 4 + to * 3
  f32.copyWithin(rTo, rFrom, rFrom + 3)

  // scales (3 floats)
  const sFrom = wasm.scalesPtr / 4 + from * 3
  const sTo = wasm.scalesPtr / 4 + to * 3
  f32.copyWithin(sTo, sFrom, sFrom + 3)

  // worldMatrices (16 floats)
  const wmFrom = wasm.worldMatricesPtr / 4 + from * 16
  const wmTo = wasm.worldMatricesPtr / 4 + to * 16
  f32.copyWithin(wmTo, wmFrom, wmFrom + 16)

  // colors (4 floats)
  const cFrom = wasm.colorsPtr / 4 + from * 4
  const cTo = wasm.colorsPtr / 4 + to * 4
  f32.copyWithin(cTo, cFrom, cFrom + 4)

  // flags (1 u32)
  u32[wasm.flagsPtr / 4 + to] = u32[wasm.flagsPtr / 4 + from]!

  // bspheres (4 floats)
  const bFrom = wasm.bspheresPtr / 4 + from * 4
  const bTo = wasm.bspheresPtr / 4 + to * 4
  f32.copyWithin(bTo, bFrom, bFrom + 4)

  // geometryIds (1 u32)
  u32[wasm.geometryIdsPtr / 4 + to] = u32[wasm.geometryIdsPtr / 4 + from]!
}

function computeBoundingSphere(vertices: Float32Array): {
  center: Float32Array
  radius: number
} {
  const stride = 6 // pos(3) + normal(3)
  const count = vertices.length / stride
  let cx = 0,
    cy = 0,
    cz = 0

  for (let i = 0; i < count; i++) {
    cx += vertices[i * stride]!
    cy += vertices[i * stride + 1]!
    cz += vertices[i * stride + 2]!
  }
  cx /= count
  cy /= count
  cz /= count

  let maxR2 = 0
  for (let i = 0; i < count; i++) {
    const dx = vertices[i * stride]! - cx
    const dy = vertices[i * stride + 1]! - cy
    const dz = vertices[i * stride + 2]! - cz
    maxR2 = Math.max(maxR2, dx * dx + dy * dy + dz * dz)
  }

  return {
    center: new Float32Array([cx, cy, cz]),
    radius: Math.sqrt(maxR2),
  }
}
