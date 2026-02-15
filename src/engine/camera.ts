import type { WasmCore } from './wasm.ts'

export class Camera {
  eye = new Float32Array([0, -10, 5])
  target = new Float32Array([0, 0, 0])
  up = new Float32Array([0, 0, 1])
  fov = Math.PI / 4 // 45 degrees
  near = 0.1
  far = 1000

  // Output matrices
  view = new Float32Array(16)
  projection = new Float32Array(16)
  vp = new Float32Array(16)

  // Byte offsets in WASM scratch area
  private viewOffset: number
  private projOffset: number
  private vpOffset: number

  constructor(scratchOffset: number) {
    // Allocate space in WASM scratch area for 3 mat4s (192 bytes)
    this.viewOffset = scratchOffset
    this.projOffset = scratchOffset + 64
    this.vpOffset = scratchOffset + 128
  }

  update(wasm: WasmCore, aspect: number) {
    const { exports, f32 } = wasm

    exports.vc_look_at(
      this.viewOffset,
      this.eye[0]!,
      this.eye[1]!,
      this.eye[2]!,
      this.target[0]!,
      this.target[1]!,
      this.target[2]!,
      this.up[0]!,
      this.up[1]!,
      this.up[2]!,
    )

    exports.vc_perspective(this.projOffset, this.fov, aspect, this.near, this.far)
    exports.vc_m4_multiply(this.vpOffset, this.projOffset, this.viewOffset)

    // Copy results to JS-side arrays
    const vi = this.viewOffset / 4
    const pi = this.projOffset / 4
    const vpi = this.vpOffset / 4
    this.view.set(f32.subarray(vi, vi + 16))
    this.projection.set(f32.subarray(pi, pi + 16))
    this.vp.set(f32.subarray(vpi, vpi + 16))
  }
}
