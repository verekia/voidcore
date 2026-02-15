import { Camera } from './camera.ts'
import { createRenderer, type Backend, type Renderer } from './gpu.ts'
import { m4Multiply } from './math.ts'
import { Mesh } from './mesh.ts'
import { loadWasm } from './wasm.ts'

import type { BVH, RaycastHit } from './bvh.ts'
import type { Geometry } from './geometry.ts'
import type { BloomConfig, DrawEntity } from './renderer.ts'
import type { WasmCore } from './wasm.ts'

const FLAG_VISIBLE = 0x02
const FLAG_UNLIT = 0x04

export interface SceneConfig {
  backend?: Backend
}

interface RegisteredGeometry {
  geometry: Geometry
  skinned: boolean
  joints?: Uint8Array
  weights?: Float32Array
  textured?: boolean
  uvs?: Float32Array
}

interface RegisteredTexture {
  data: Uint8Array
  width: number
  height: number
}

export class Scene {
  wasm!: WasmCore
  renderer!: Renderer
  camera!: Camera
  canvas: HTMLCanvasElement

  private meshes: (Mesh | null)[] = []
  private activeCount = 0
  private geometryRegistry = new Map<number, RegisteredGeometry>()
  private textureRegistry = new Map<number, RegisteredTexture>()
  private nextGeometryId = 0
  private nextTextureId = 0
  private config: SceneConfig

  // Lighting
  private lightDir = new Float32Array([0.5, -1, 0.3])
  private lightColor = new Float32Array([1, 1, 1])
  private ambientColor = new Float32Array([0.15, 0.15, 0.15])

  // Bloom
  private bloomConfig: BloomConfig | null = null

  // Stats
  visibleCount = 0
  drawCalls = 0

  // Scratch offsets for frustum planes (6 planes × 4 floats = 96 bytes)
  private planesOffset = 0

  // Temp mat4s for bone attachment world matrix computation
  private _tempWorldMat = new Float32Array(16)
  private _tempRSMat = new Float32Array(16)

  // BVH cache and scratch offsets for raycasting
  private _bvhCache = new Map<number, BVH>()
  private _rayResultOffset = 0 // scratch offset for raycast result (16 bytes)
  private _invMatOffset = 0 // scratch offset for inverse matrix (64 bytes)

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
    // Raycast scratch: after frustum planes (96 bytes)
    scene._rayResultOffset = scene.planesOffset + 96
    scene._invMatOffset = scene._rayResultOffset + 16
    return scene
  }

  registerGeometry(geometry: Geometry): number {
    const id = this.nextGeometryId++
    this.geometryRegistry.set(id, { geometry, skinned: false })
    this.renderer.registerGeometry(id, geometry.vertices, geometry.indices)
    return id
  }

  registerSkinnedGeometry(geometry: Geometry, joints: Uint8Array, weights: Float32Array): number {
    const id = this.nextGeometryId++
    this.geometryRegistry.set(id, { geometry, skinned: true, joints, weights })
    this.renderer.registerSkinnedGeometry(id, geometry.vertices, geometry.indices, joints, weights)
    return id
  }

  registerTexturedGeometry(geometry: Geometry, uvs: Float32Array): number {
    const id = this.nextGeometryId++
    this.geometryRegistry.set(id, { geometry, skinned: false, textured: true, uvs })
    this.renderer.registerTexturedGeometry(id, geometry.vertices, geometry.indices, uvs)
    return id
  }

  registerTexture(data: Uint8Array, width: number, height: number): number {
    const id = this.nextTextureId++
    this.textureRegistry.set(id, { data, width, height })
    this.renderer.registerTexture(id, data, width, height)
    return id
  }

  add(mesh: Mesh): Mesh {
    const entityId = this.activeCount++
    mesh._bind(this.wasm, entityId)
    this.meshes[entityId] = mesh

    // Compute bounding sphere from geometry
    const reg = this.geometryRegistry.get(mesh.geometryId)
    if (reg) {
      const { center, radius } = computeBoundingSphere(reg.geometry.vertices)
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

  setBloom(config: BloomConfig) {
    this.bloomConfig = config
    this.renderer.setBloom(config)
  }

  buildBVH(geometryId: number) {
    if (this._bvhCache.has(geometryId)) return

    const reg = this.geometryRegistry.get(geometryId)
    if (!reg) return

    const { vertices, indices } = reg.geometry
    const stride = 10 // floats per vertex

    // Write geometry data to frame arena region (temporary, used only during build)
    const frameBase = 16 * 1024 * 1024
    const vertOffset = frameBase
    const vertBytes = vertices.byteLength
    const idxOffset = (frameBase + vertBytes + 3) & ~3 // align to 4 bytes

    // Copy vertices into WASM memory
    new Float32Array(this.wasm.memory.buffer, vertOffset, vertices.length).set(vertices)

    // Copy indices into WASM memory
    const isU32 = indices instanceof Uint32Array ? 1 : 0
    if (isU32) {
      new Uint32Array(this.wasm.memory.buffer, idxOffset, indices.length).set(indices as Uint32Array)
    } else {
      new Uint16Array(this.wasm.memory.buffer, idxOffset, indices.length).set(indices as Uint16Array)
    }

    const bvhOffset = this.wasm.exports.vc_bvh_build(vertOffset, idxOffset, indices.length, stride, isU32)

    if (bvhOffset > 0) {
      // Validate BVH header
      const hdr = this.wasm.u32
      const hb = bvhOffset / 4
      const nodesOff = hdr[hb]!
      const nodeCount = hdr[hb + 1]!
      const triIdxOff = hdr[hb + 2]!
      const posOff = hdr[hb + 3]!
      const idxOff = hdr[hb + 4]!
      const triCount = hdr[hb + 6]!
      console.log(
        `BVH built: header@${bvhOffset} nodes@${nodesOff}(${nodeCount}) triIdx@${triIdxOff} pos@${posOff} idx@${idxOff} tris=${triCount}`,
      )

      this._bvhCache.set(geometryId, {
        offset: bvhOffset,
        triCount: Math.floor(indices.length / 3),
      })
    } else {
      console.warn(`BVH build failed for geometry ${geometryId} (returned 0)`)
    }
  }

  raycast(
    ox: number,
    oy: number,
    oz: number,
    dx: number,
    dy: number,
    dz: number,
    result: RaycastHit,
    meshFilter?: (mesh: Mesh) => boolean,
    maxT = Infinity,
  ): boolean {
    const { wasm } = this
    const { f32 } = wasm

    result.hit = false
    result.distance = Infinity
    result.mesh = null

    let closestT = maxT

    for (let i = 0; i < this.activeCount; i++) {
      const mesh = this.meshes[i]
      if (!mesh) continue
      if (!mesh.visible) continue
      if (mesh.skinInstance) continue
      if (meshFilter && !meshFilter(mesh)) continue

      // Get or lazy-build BVH
      let bvh = this._bvhCache.get(mesh.geometryId)
      if (!bvh) {
        this.buildBVH(mesh.geometryId)
        bvh = this._bvhCache.get(mesh.geometryId)
        if (!bvh) continue
      }

      // Invert the mesh's world matrix
      const wmByteOffset = wasm.worldMatricesPtr + mesh.entityId * 64
      wasm.exports.vc_m4_invert(this._invMatOffset, wmByteOffset)

      const ib = this._invMatOffset / 4
      // Transform ray origin to local space: localOrigin = M^-1 * worldOrigin (w=1)
      const lox = f32[ib]! * ox + f32[ib + 4]! * oy + f32[ib + 8]! * oz + f32[ib + 12]!
      const loy = f32[ib + 1]! * ox + f32[ib + 5]! * oy + f32[ib + 9]! * oz + f32[ib + 13]!
      const loz = f32[ib + 2]! * ox + f32[ib + 6]! * oy + f32[ib + 10]! * oz + f32[ib + 14]!

      // Transform ray direction to local space (w=0, no translation)
      const ldx = f32[ib]! * dx + f32[ib + 4]! * dy + f32[ib + 8]! * dz
      const ldy = f32[ib + 1]! * dx + f32[ib + 5]! * dy + f32[ib + 9]! * dz
      const ldz = f32[ib + 2]! * dx + f32[ib + 6]! * dy + f32[ib + 10]! * dz

      // Local direction length (for converting local t to world t)
      const dirLen = Math.sqrt(ldx * ldx + ldy * ldy + ldz * ldz)
      if (dirLen < 1e-8) continue
      const localMaxT = closestT * dirLen

      const t = wasm.exports.vc_bvh_raycast(bvh.offset, lox, loy, loz, ldx, ldy, ldz, localMaxT, this._rayResultOffset)

      if (t >= 0) {
        const worldT = t / dirLen
        if (worldT < closestT) {
          closestT = worldT

          // Read local normal from result (3 floats)
          const rb = this._rayResultOffset / 4
          const lnx = f32[rb]!
          const lny = f32[rb + 1]!
          const lnz = f32[rb + 2]!

          // Transform normal to world space via inverse-transpose: (M^-1)^T * n
          const wnx = f32[ib]! * lnx + f32[ib + 1]! * lny + f32[ib + 2]! * lnz
          const wny = f32[ib + 4]! * lnx + f32[ib + 5]! * lny + f32[ib + 6]! * lnz
          const wnz = f32[ib + 8]! * lnx + f32[ib + 9]! * lny + f32[ib + 10]! * lnz
          const nLen = Math.sqrt(wnx * wnx + wny * wny + wnz * wnz)

          result.hit = true
          result.distance = worldT
          result.pointX = ox + dx * worldT
          result.pointY = oy + dy * worldT
          result.pointZ = oz + dz * worldT
          result.normalX = nLen > 1e-8 ? wnx / nLen : 0
          result.normalY = nLen > 1e-8 ? wny / nLen : 0
          result.normalZ = nLen > 1e-8 ? wnz / nLen : 0
          result.faceIndex = 0
          result.mesh = mesh
        }
      }
    }

    return result.hit
  }

  render() {
    const { wasm, renderer, camera, activeCount } = this
    const aspect = this.canvas.width / this.canvas.height

    // 1. Mark bone-attached entities dirty so their base TRS is recomputed each frame
    for (let i = 0; i < activeCount; i++) {
      const mesh = this.meshes[i]
      if (mesh?.boneAttachment) mesh.setDirty()
    }

    // 2. Compute world matrices for dirty entities
    wasm.exports.vc_compute_world_matrices(activeCount)

    // 3. Update camera
    camera.update(wasm, aspect)

    // 4. Apply bone attachments (override world matrices)
    // final = T(position) * boneGlobal * RS(rotation, scale)
    // This ensures rotation/scale are applied relative to the bone, not the entity origin
    for (let i = 0; i < activeCount; i++) {
      const mesh = this.meshes[i]
      if (!mesh?.boneAttachment) continue
      const { skinInstance, boneNodeIndex } = mesh.boneAttachment
      const boneGlobal = skinInstance.globalMatrices

      // Extract RS part (zero out translation from entity world matrix)
      this._tempRSMat.set(mesh.worldMatrix)
      const tx = this._tempRSMat[12]!
      const ty = this._tempRSMat[13]!
      const tz = this._tempRSMat[14]!
      this._tempRSMat[12] = 0
      this._tempRSMat[13] = 0
      this._tempRSMat[14] = 0

      // temp = boneGlobal * RS
      m4Multiply(this._tempWorldMat, 0, boneGlobal, boneNodeIndex * 16, this._tempRSMat, 0)

      // final = T(pos) * temp (pre-multiplying by pure translation just adds to column 3)
      this._tempWorldMat[12] = this._tempWorldMat[12]! + tx
      this._tempWorldMat[13] = this._tempWorldMat[13]! + ty
      this._tempWorldMat[14] = this._tempWorldMat[14]! + tz

      mesh.worldMatrix.set(this._tempWorldMat)
    }

    // 5. Update bounding spheres for all meshes
    for (let i = 0; i < activeCount; i++) {
      this.meshes[i]?.updateBsphere()
    }

    // 6. Frustum culling (if available in WASM)
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

      // 7. Sort draw calls (if available)
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

    // 7. Build draw entity list
    const drawEntities: DrawEntity[] = []
    for (const idx of visibleIndices) {
      const mesh = this.meshes[idx]
      if (!mesh) continue
      const entity: DrawEntity = {
        worldMatrix: mesh.worldMatrix,
        color: mesh.color,
        geometryId: mesh.geometryId,
        unlit: (wasm.u32[wasm.flagsPtr / 4 + idx]! & FLAG_UNLIT) !== 0,
      }
      if (mesh.skinInstance) {
        entity.jointMatrices = mesh.skinInstance.jointMatrices
      }
      if (mesh.aoMap >= 0) {
        entity.textureId = mesh.aoMap
        entity.isTextured = true
        entity.aoIntensity = mesh.aoIntensity
      }
      drawEntities.push(entity)
    }

    // 8. Render
    renderer.updateCamera(camera.view, camera.projection)
    renderer.updateLighting(this.lightDir, this.lightColor, this.ambientColor)
    renderer.draw(drawEntities, drawEntities.length)
    this.drawCalls = drawEntities.length

    // 9. Reset per-frame arena
    wasm.exports.vc_frame_reset()
  }

  resize(width: number, height: number) {
    this.canvas.width = width
    this.canvas.height = height
    this.renderer.resize(width, height)
  }

  setRenderer(canvas: HTMLCanvasElement, renderer: Renderer) {
    this.canvas = canvas
    this.renderer = renderer

    // Re-register all geometries with the new renderer
    for (const [id, reg] of this.geometryRegistry) {
      if (reg.skinned && reg.joints && reg.weights) {
        this.renderer.registerSkinnedGeometry(id, reg.geometry.vertices, reg.geometry.indices, reg.joints, reg.weights)
      } else if (reg.textured && reg.uvs) {
        this.renderer.registerTexturedGeometry(id, reg.geometry.vertices, reg.geometry.indices, reg.uvs)
      } else {
        this.renderer.registerGeometry(id, reg.geometry.vertices, reg.geometry.indices)
      }
    }

    // Re-register all textures
    for (const [id, tex] of this.textureRegistry) {
      this.renderer.registerTexture(id, tex.data, tex.width, tex.height)
    }

    // Re-apply bloom config
    if (this.bloomConfig) {
      this.renderer.setBloom(this.bloomConfig)
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
  const stride = 10 // pos(3) + normal(3) + color(3) + bloom(1)
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
