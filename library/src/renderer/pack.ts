// Vertex Packing – Converts vertex attribute data to compact GPU-friendly formats.
//
// When uploading geometry to the GPU, not every attribute needs full 32-bit float precision.
// These packing functions convert from the engine's internal Float32Array representation to
// smaller GPU formats, reducing vertex buffer size and memory bandwidth:
//
//   packNormalsSnorm8()    – Float32 normals (12 bytes/vertex) → Snorm8x4 (4 bytes/vertex).
//                            Unit-length normals only need [-1,1] range; 8-bit snorm gives
//                            1/127 precision (~0.008), imperceptible for lighting.
//
//   packUVsFloat16()       – Float32 UVs (8 bytes/vertex) → Float16x2 (4 bytes/vertex).
//                            Texture coordinates rarely need more than ~3 decimal digits of
//                            precision. Float16 covers [0,1] UVs with ~0.001 accuracy.
//
//   packWeightsUnorm8()    – Float32 bone weights (16 bytes/vertex) → Unorm8x4 (4 bytes/vertex).
//                            Blend weights are in [0,1] and sum to 1.0; 8-bit unorm (1/255
//                            precision) is sufficient for smooth skeletal deformation. The four
//                            quantized values are normalized to sum to exactly 255 — without this,
//                            rounding errors cause the skin matrix to be slightly non-unit-sum,
//                            and the resulting vertex offset grows with distance from origin.
//
//   packColorsUnorm8()     – Float32 RGBA colors (16 bytes/vertex) → Unorm8x4 (4 bytes/vertex).
//                            Vertex colors are in [0,1] range; 8-bit unorm is sufficient for
//                            palette-baked colors.
//
//   packEmissiveFloat16()  – Float32 RGBA emissive (16 bytes/vertex) → Float16x4 (8 bytes/vertex).
//                            Emissive values can exceed 1.0 (HDR), so half-float precision is
//                            needed to preserve intensity information.

// ─── Float32-to-Float16 conversion ──────────────────────────────────

// Shared buffer for bit manipulation (reused across all conversions)
const _f32 = new Float32Array(1)
const _u32 = new Uint32Array(_f32.buffer)

/** Convert a single float32 value to float16 (stored as uint16 bit pattern). */
const floatToHalf = (val: number): number => {
  _f32[0] = val
  const b = _u32[0]!
  // Extract sign, exponent, mantissa from float32
  const sign = (b >>> 16) & 0x8000
  const exp = ((b >>> 23) & 0xff) - 112 // Rebias exponent: f32 bias 127 → f16 bias 15
  const man = (b & 0x7fffff) >>> 13 // Truncate mantissa from 23 to 10 bits (rounds toward zero)

  if (exp <= 0) return sign // Underflow → ±0 (handles f32 zero and denormals)
  if (exp >= 31) return sign | 0x7c00 // Overflow → ±Infinity (handles f32 inf/NaN)
  return sign | (exp << 10) | man
}

// ─── Packing functions ──────────────────────────────────────────────

/**
 * Pack float32 normals (vec3 per vertex) into snorm8x4 (4 bytes per vertex).
 * Each component is clamped to [-1,1] and mapped to [-127,127] as Int8.
 * The 4th component (W) defaults to 0 but can be set via wValue (e.g. 127 for outline flag).
 */
export const packNormalsSnorm8 = (normals: Float32Array, vertexCount: number, wValue = 0): Int8Array => {
  const packed = new Int8Array(vertexCount * 4)
  for (let i = 0, j = 0; i < vertexCount * 3; i += 3, j += 4) {
    packed[j] = Math.round(Math.max(-1, Math.min(1, normals[i]!)) * 127)
    packed[j + 1] = Math.round(Math.max(-1, Math.min(1, normals[i + 1]!)) * 127)
    packed[j + 2] = Math.round(Math.max(-1, Math.min(1, normals[i + 2]!)) * 127)
    packed[j + 3] = wValue
  }
  return packed
}

/**
 * Pack float32 UVs into float16 (stored as Uint16Array).
 * Each float32 value is converted to IEEE 754 half-precision (16-bit float).
 */
export const packUVsFloat16 = (uvs: Float32Array): Uint16Array => {
  const packed = new Uint16Array(uvs.length)
  for (let i = 0; i < uvs.length; i++) {
    packed[i] = floatToHalf(uvs[i]!)
  }
  return packed
}

/**
 * Pack float32 bone weights into unorm8 (1 byte per component).
 * Each group of 4 weights is normalized so the quantized values sum to exactly 255,
 * preventing distance-dependent vertex drift in skinned meshes.
 */
export const packWeightsUnorm8 = (weights: Float32Array): Uint8Array => {
  const packed = new Uint8Array(weights.length)
  for (let i = 0; i < weights.length; i += 4) {
    const w0 = Math.max(0, weights[i]!)
    const w1 = Math.max(0, weights[i + 1]!)
    const w2 = Math.max(0, weights[i + 2]!)
    const w3 = Math.max(0, weights[i + 3]!)
    const sum = w0 + w1 + w2 + w3
    if (sum === 0) continue // all zeros → leave as 0

    // Scale so the group sums to 255, then round
    const scale = 255 / sum
    let a = Math.round(w0 * scale)
    let b = Math.round(w1 * scale)
    let c = Math.round(w2 * scale)
    let d = Math.round(w3 * scale)

    // Fix rounding residual on the largest component
    const diff = 255 - (a + b + c + d)
    if (a >= b && a >= c && a >= d) a += diff
    else if (b >= c && b >= d) b += diff
    else if (c >= d) c += diff
    else d += diff

    packed[i] = a
    packed[i + 1] = b
    packed[i + 2] = c
    packed[i + 3] = d
  }
  return packed
}

/**
 * Pack float32 RGBA vertex colors into unorm8 (1 byte per component).
 * Each component is clamped to [0,1] and mapped to [0,255] as Uint8.
 */
export const packColorsUnorm8 = (colors: Float32Array, vertexCount: number): Uint8Array => {
  const packed = new Uint8Array(vertexCount * 4)
  for (let i = 0; i < vertexCount * 4; i++) {
    packed[i] = Math.round(Math.max(0, Math.min(1, colors[i]!)) * 255)
  }
  return packed
}

/**
 * Pack float32 RGBA emissive colors into float16 (stored as Uint16Array).
 * Uses half-float to preserve HDR values above 1.0.
 */
export const packEmissiveFloat16 = (emissiveColors: Float32Array, vertexCount: number): Uint16Array => {
  const packed = new Uint16Array(vertexCount * 4)
  for (let i = 0; i < vertexCount * 4; i++) {
    packed[i] = floatToHalf(emissiveColors[i]!)
  }
  return packed
}
