import { expect, test, describe } from 'bun:test'

import { packNormalsSnorm8, packUVsFloat16, packWeightsUnorm8 } from './pack.ts'

describe('packNormalsSnorm8', () => {
  test('packs unit axes correctly', () => {
    // 2 vertices: +X normal and +Z normal
    const normals = new Float32Array([1, 0, 0, 0, 0, 1])
    const packed = packNormalsSnorm8(normals, 2)
    expect(packed).toBeInstanceOf(Int8Array)
    expect(packed.length).toBe(8) // 2 vertices × 4 components
    // +X: (127, 0, 0, 0)
    expect(packed[0]).toBe(127)
    expect(packed[1]).toBe(0)
    expect(packed[2]).toBe(0)
    expect(packed[3]).toBe(0) // W padding
    // +Z: (0, 0, 127, 0)
    expect(packed[4]).toBe(0)
    expect(packed[5]).toBe(0)
    expect(packed[6]).toBe(127)
    expect(packed[7]).toBe(0)
  })

  test('packs negative values', () => {
    const normals = new Float32Array([-1, -1, -1])
    const packed = packNormalsSnorm8(normals, 1)
    expect(packed[0]).toBe(-127)
    expect(packed[1]).toBe(-127)
    expect(packed[2]).toBe(-127)
  })

  test('clamps out-of-range values', () => {
    const normals = new Float32Array([2, -3, 0.5])
    const packed = packNormalsSnorm8(normals, 1)
    expect(packed[0]).toBe(127) // clamped from 2
    expect(packed[1]).toBe(-127) // clamped from -3
    expect(packed[2]).toBe(64) // round(0.5 * 127) = 64
  })

  test('packs zero normals', () => {
    const normals = new Float32Array([0, 0, 0])
    const packed = packNormalsSnorm8(normals, 1)
    expect(packed[0]).toBe(0)
    expect(packed[1]).toBe(0)
    expect(packed[2]).toBe(0)
    expect(packed[3]).toBe(0)
  })
})

describe('packUVsFloat16', () => {
  test('packs zero', () => {
    const uvs = new Float32Array([0, 0])
    const packed = packUVsFloat16(uvs)
    expect(packed).toBeInstanceOf(Uint16Array)
    expect(packed.length).toBe(2)
    expect(packed[0]).toBe(0)
    expect(packed[1]).toBe(0)
  })

  test('packs 1.0', () => {
    const uvs = new Float32Array([1.0])
    const packed = packUVsFloat16(uvs)
    // Float16 1.0 = 0 01111 0000000000 = 0x3C00
    expect(packed[0]).toBe(0x3c00)
  })

  test('packs 0.5', () => {
    const uvs = new Float32Array([0.5])
    const packed = packUVsFloat16(uvs)
    // Float16 0.5 = 0 01110 0000000000 = 0x3800
    expect(packed[0]).toBe(0x3800)
  })

  test('packs negative values', () => {
    const uvs = new Float32Array([-1.0])
    const packed = packUVsFloat16(uvs)
    // Float16 -1.0 = 1 01111 0000000000 = 0xBC00
    expect(packed[0]).toBe(0xbc00)
  })

  test('handles overflow as infinity', () => {
    const uvs = new Float32Array([100000])
    const packed = packUVsFloat16(uvs)
    // Should be +Infinity = 0x7C00
    expect(packed[0]).toBe(0x7c00)
  })

  test('handles very small values as zero', () => {
    const uvs = new Float32Array([1e-10])
    const packed = packUVsFloat16(uvs)
    // Too small for float16 → underflows to 0
    expect(packed[0]).toBe(0)
  })

  test('round-trips typical UV values with acceptable precision', () => {
    // Reconstruct float16 to check precision
    const testValues = [0.0, 0.25, 0.5, 0.75, 1.0]
    const uvs = new Float32Array(testValues)
    const packed = packUVsFloat16(uvs)

    for (let i = 0; i < testValues.length; i++) {
      // Decode float16 back to float32 for comparison
      const h = packed[i]!
      const sign = (h >>> 15) & 1
      const exp = (h >>> 10) & 0x1f
      const man = h & 0x3ff
      let val: number
      if (exp === 0) val = (sign ? -1 : 1) * (man / 1024) * Math.pow(2, -14)
      else if (exp === 31) val = man === 0 ? (sign ? -Infinity : Infinity) : NaN
      else val = (sign ? -1 : 1) * (1 + man / 1024) * Math.pow(2, exp - 15)
      expect(Math.abs(val - testValues[i]!)).toBeLessThan(0.001)
    }
  })
})

describe('packWeightsUnorm8', () => {
  test('packs zero and one', () => {
    const weights = new Float32Array([0, 1, 0, 0])
    const packed = packWeightsUnorm8(weights)
    expect(packed).toBeInstanceOf(Uint8Array)
    expect(packed.length).toBe(4)
    expect(packed[0]).toBe(0)
    expect(packed[1]).toBe(255)
    expect(packed[2]).toBe(0)
    expect(packed[3]).toBe(0)
  })

  test('packs 0.5', () => {
    const weights = new Float32Array([0.5])
    const packed = packWeightsUnorm8(weights)
    // round(0.5 * 255) = 128
    expect(packed[0]).toBe(128)
  })

  test('clamps negative values to 0', () => {
    const weights = new Float32Array([-0.5])
    const packed = packWeightsUnorm8(weights)
    expect(packed[0]).toBe(0)
  })

  test('clamps values above 1 to 255', () => {
    const weights = new Float32Array([1.5])
    const packed = packWeightsUnorm8(weights)
    expect(packed[0]).toBe(255)
  })

  test('packs typical blend weights', () => {
    // Common case: one dominant bone
    const weights = new Float32Array([0.8, 0.15, 0.05, 0])
    const packed = packWeightsUnorm8(weights)
    expect(packed[0]).toBe(204) // round(0.8 * 255)
    expect(packed[1]).toBe(38) // round(0.15 * 255)
    expect(packed[2]).toBe(13) // round(0.05 * 255)
    expect(packed[3]).toBe(0)
  })
})
