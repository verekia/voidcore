import type { WasmCore } from './wasm.ts'

// Flags
const FLAG_DIRTY = 0x01
const FLAG_VISIBLE = 0x02
const FLAG_UNLIT = 0x04

export interface MeshOptions {
  geometryId: number
  position?: [number, number, number]
  rotation?: [number, number, number]
  scale?: [number, number, number]
  color?: [number, number, number, number]
  visible?: boolean
  unlit?: boolean
}

export class Mesh {
  entityId = -1

  // Sub-views into WASM SoA arrays (set when added to scene)
  position!: Float32Array
  rotation!: Float32Array
  scale!: Float32Array
  color!: Float32Array
  worldMatrix!: Float32Array

  private wasm!: WasmCore
  private flags!: Uint32Array
  geometryId: number
  private initOptions: MeshOptions

  // Bounding sphere (center offset from entity origin, radius)
  bsphereRadius = 0.5
  bsphereCenterOffset = new Float32Array(3)

  constructor(options: MeshOptions) {
    this.geometryId = options.geometryId
    this.initOptions = options
  }

  /** Called by Scene when entity is assigned */
  _bind(wasm: WasmCore, entityId: number) {
    this.entityId = entityId
    this.wasm = wasm

    const { f32, u32 } = wasm
    const posBase = wasm.positionsPtr / 4 + entityId * 3
    const rotBase = wasm.eulerRotationsPtr / 4 + entityId * 3
    const scBase = wasm.scalesPtr / 4 + entityId * 3
    const colBase = wasm.colorsPtr / 4 + entityId * 4
    const wmBase = wasm.worldMatricesPtr / 4 + entityId * 16

    this.position = f32.subarray(posBase, posBase + 3)
    this.rotation = f32.subarray(rotBase, rotBase + 3)
    this.scale = f32.subarray(scBase, scBase + 3)
    this.color = f32.subarray(colBase, colBase + 4)
    this.worldMatrix = f32.subarray(wmBase, wmBase + 16)
    this.flags = u32.subarray(wasm.flagsPtr / 4 + entityId, wasm.flagsPtr / 4 + entityId + 1)

    // Apply initial options
    const opts = this.initOptions
    if (opts.position) {
      this.position[0] = opts.position[0]
      this.position[1] = opts.position[1]
      this.position[2] = opts.position[2]
    }
    if (opts.rotation) {
      this.rotation[0] = opts.rotation[0]
      this.rotation[1] = opts.rotation[1]
      this.rotation[2] = opts.rotation[2]
    }
    if (opts.scale) {
      this.scale[0] = opts.scale[0]
      this.scale[1] = opts.scale[1]
      this.scale[2] = opts.scale[2]
    }
    if (opts.color) {
      this.color[0] = opts.color[0]
      this.color[1] = opts.color[1]
      this.color[2] = opts.color[2]
      this.color[3] = opts.color[3]
    }

    // Set flags
    let fl = FLAG_DIRTY | FLAG_VISIBLE
    if (opts.visible === false) fl &= ~FLAG_VISIBLE
    if (opts.unlit) fl |= FLAG_UNLIT
    this.flags[0] = fl

    // Set geometry ID in SoA
    u32[wasm.geometryIdsPtr / 4 + entityId] = this.geometryId
  }

  setDirty() {
    if (this.flags) {
      this.flags[0]! |= FLAG_DIRTY
    }
  }

  get visible(): boolean {
    return (this.flags[0]! & FLAG_VISIBLE) !== 0
  }

  set visible(v: boolean) {
    if (v) this.flags[0]! |= FLAG_VISIBLE
    else this.flags[0]! &= ~FLAG_VISIBLE
  }

  get unlit(): boolean {
    return (this.flags[0]! & FLAG_UNLIT) !== 0
  }

  set unlit(v: boolean) {
    if (v) this.flags[0]! |= FLAG_UNLIT
    else this.flags[0]! &= ~FLAG_UNLIT
  }

  /** Update bounding sphere in WASM memory */
  updateBsphere() {
    const f32 = this.wasm.f32
    const base = this.wasm.bspheresPtr / 4 + this.entityId * 4
    f32[base] = this.position[0]! + this.bsphereCenterOffset[0]!
    f32[base + 1] = this.position[1]! + this.bsphereCenterOffset[1]!
    f32[base + 2] = this.position[2]! + this.bsphereCenterOffset[2]!
    const maxScale = Math.max(this.scale[0]!, this.scale[1]!, this.scale[2]!)
    f32[base + 3] = this.bsphereRadius * maxScale
  }
}
