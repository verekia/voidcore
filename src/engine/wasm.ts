export interface WasmExports {
  memory: WebAssembly.Memory
  vc_init: (pages: number) => number
  vc_perspective: (offset: number, fov: number, aspect: number, near: number, far: number) => void
  vc_look_at: (
    offset: number,
    ex: number,
    ey: number,
    ez: number,
    tx: number,
    ty: number,
    tz: number,
    ux: number,
    uy: number,
    uz: number,
  ) => void
  vc_m4_multiply: (out: number, a: number, b: number) => void
  vc_frame_reset: () => void
  vc_compute_world_matrices: (count: number) => number
  vc_get_positions_ptr: () => number
  vc_get_euler_rotations_ptr: () => number
  vc_get_scales_ptr: () => number
  vc_get_world_matrices_ptr: () => number
  vc_get_colors_ptr: () => number
  vc_get_flags_ptr: () => number
  vc_get_bspheres_ptr: () => number
  vc_get_geometry_ids_ptr: () => number
  vc_get_sort_keys_ptr: () => number
  vc_get_visible_indices_ptr: () => number
  vc_extract_frustum_planes: (vpOffset: number, planesOffset: number) => void
  vc_frustum_cull: (
    count: number,
    planesOffset: number,
    bspheresOffset: number,
    flagsOffset: number,
    outOffset: number,
  ) => number
  vc_build_sort_keys: (count: number, visibleOffset: number, geoIdsOffset: number, keysOutOffset: number) => void
  vc_sort_draw_calls: (count: number, keysOffset: number, indicesOffset: number) => void
  vc_m4_invert: (outOffset: number, mOffset: number) => void
  vc_bvh_build: (
    verticesOffset: number,
    indicesOffset: number,
    indexCount: number,
    stride: number,
    isU32Indices: number,
  ) => number
  vc_bvh_raycast: (
    bvhOffset: number,
    ox: number,
    oy: number,
    oz: number,
    dx: number,
    dy: number,
    dz: number,
    maxT: number,
    resultOffset: number,
  ) => number
  vc_bvh_alloc_reset: () => void
}

export interface WasmCore {
  exports: WasmExports
  memory: WebAssembly.Memory
  f32: Float32Array
  u32: Uint32Array
  u8: Uint8Array
  scratchOffset: number
  // SoA byte offsets
  positionsPtr: number
  eulerRotationsPtr: number
  scalesPtr: number
  worldMatricesPtr: number
  colorsPtr: number
  flagsPtr: number
  bspheresPtr: number
  geometryIdsPtr: number
  sortKeysPtr: number
  visibleIndicesPtr: number
}

export async function loadWasm(): Promise<WasmCore> {
  const response = await fetch('/voidcore.wasm')
  const memory = new WebAssembly.Memory({ initial: 512 }) // 512 pages = 32 MB
  const { instance } = await WebAssembly.instantiateStreaming(response, {
    env: { memory },
  })

  const exports = instance.exports as unknown as WasmExports
  const buffer = memory.buffer

  const f32 = new Float32Array(buffer)
  const u32 = new Uint32Array(buffer)
  const u8 = new Uint8Array(buffer)

  const scratchOffset = exports.vc_init(512)

  return {
    exports,
    memory,
    f32,
    u32,
    u8,
    scratchOffset,
    positionsPtr: exports.vc_get_positions_ptr(),
    eulerRotationsPtr: exports.vc_get_euler_rotations_ptr(),
    scalesPtr: exports.vc_get_scales_ptr(),
    worldMatricesPtr: exports.vc_get_world_matrices_ptr(),
    colorsPtr: exports.vc_get_colors_ptr(),
    flagsPtr: exports.vc_get_flags_ptr(),
    bspheresPtr: exports.vc_get_bspheres_ptr(),
    geometryIdsPtr: exports.vc_get_geometry_ids_ptr(),
    sortKeysPtr: exports.vc_get_sort_keys_ptr(),
    visibleIndicesPtr: exports.vc_get_visible_indices_ptr(),
  }
}
