import { expect, test, describe } from 'bun:test'

import {
  mat4Create,
  mat4Identity,
  mat4Perspective,
  mat4Ortho,
  mat4OrthoZO,
  aabbCreate,
  aabbTransform,
  vec3TransformMat4,
} from './index'

import type { AABB, Mat4, Vec3 } from './index'

// ─── mat4Identity ──────────────────────────────────────────────────

describe('mat4Identity', () => {
  test('sets all 16 elements correctly', () => {
    const out = mat4Create()
    // Dirty the matrix first to ensure every element is written
    out.fill(99)
    mat4Identity(out)
    // prettier-ignore
    expect(Array.from(out)).toEqual([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ])
  })
})

// ─── mat4Perspective ───────────────────────────────────────────────

describe('mat4Perspective', () => {
  test('sets all 16 elements (zero-to-one)', () => {
    const out = mat4Create()
    out.fill(99)
    mat4Perspective(out, Math.PI / 4, 1.5, 0.1, 100, 'zero-to-one')
    // No element should remain 99
    for (let i = 0; i < 16; i++) {
      expect(out[i]).not.toBe(99)
    }
    // Standard perspective invariants
    expect(out[15]).toBe(0)
    expect(out[11]).toBe(-1)
    expect(out[1]).toBe(0)
    expect(out[3]).toBe(0)
    expect(out[4]).toBe(0)
    expect(out[7]).toBe(0)
    expect(out[12]).toBe(0)
    expect(out[13]).toBe(0)
  })

  test('sets all 16 elements (neg-one-to-one)', () => {
    const out = mat4Create()
    out.fill(99)
    mat4Perspective(out, Math.PI / 3, 2, 1, 500, 'neg-one-to-one')
    for (let i = 0; i < 16; i++) {
      expect(out[i]).not.toBe(99)
    }
    expect(out[15]).toBe(0)
    expect(out[11]).toBe(-1)
  })
})

// ─── mat4Ortho ─────────────────────────────────────────────────────

describe('mat4Ortho', () => {
  test('sets all 16 elements', () => {
    const out = mat4Create()
    out.fill(99)
    mat4Ortho(out, -10, 10, -5, 5, 0.1, 100)
    for (let i = 0; i < 16; i++) {
      expect(out[i]).not.toBe(99)
    }
    expect(out[15]).toBe(1)
    // Off-diagonal should be zero
    expect(out[1]).toBe(0)
    expect(out[2]).toBe(0)
    expect(out[3]).toBe(0)
    expect(out[4]).toBe(0)
    expect(out[6]).toBe(0)
    expect(out[7]).toBe(0)
    expect(out[8]).toBe(0)
    expect(out[9]).toBe(0)
    expect(out[11]).toBe(0)
  })
})

// ─── mat4OrthoZO ──────────────────────────────────────────────────

describe('mat4OrthoZO', () => {
  test('sets all 16 elements', () => {
    const out = mat4Create()
    out.fill(99)
    mat4OrthoZO(out, -10, 10, -5, 5, 0.1, 100)
    for (let i = 0; i < 16; i++) {
      expect(out[i]).not.toBe(99)
    }
    expect(out[15]).toBe(1)
    expect(out[1]).toBe(0)
    expect(out[2]).toBe(0)
    expect(out[3]).toBe(0)
    expect(out[4]).toBe(0)
    expect(out[6]).toBe(0)
    expect(out[7]).toBe(0)
    expect(out[8]).toBe(0)
    expect(out[9]).toBe(0)
    expect(out[11]).toBe(0)
  })
})

// ─── aabbTransform ─────────────────────────────────────────────────

/** Brute-force AABB transform by transforming all 8 corners */
const aabbTransformBruteForce = (out: AABB, a: AABB, m: Mat4): AABB => {
  const minX = a[0]!,
    minY = a[1]!,
    minZ = a[2]!
  const maxX = a[3]!,
    maxY = a[4]!,
    maxZ = a[5]!

  const corners: [number, number, number][] = [
    [minX, minY, minZ],
    [maxX, minY, minZ],
    [minX, maxY, minZ],
    [maxX, maxY, minZ],
    [minX, minY, maxZ],
    [maxX, minY, maxZ],
    [minX, maxY, maxZ],
    [maxX, maxY, maxZ],
  ]

  out[0] = Infinity
  out[1] = Infinity
  out[2] = Infinity
  out[3] = -Infinity
  out[4] = -Infinity
  out[5] = -Infinity

  const tv = new Float32Array(3) as Vec3
  for (const [cx, cy, cz] of corners) {
    tv[0] = cx
    tv[1] = cy
    tv[2] = cz
    vec3TransformMat4(tv, tv, m)
    if (tv[0]! < out[0]!) out[0] = tv[0]!
    if (tv[1]! < out[1]!) out[1] = tv[1]!
    if (tv[2]! < out[2]!) out[2] = tv[2]!
    if (tv[0]! > out[3]!) out[3] = tv[0]!
    if (tv[1]! > out[4]!) out[4] = tv[1]!
    if (tv[2]! > out[5]!) out[5] = tv[2]!
  }
  return out
}

describe('aabbTransform', () => {
  test('matches brute-force with identity matrix', () => {
    const aabb: AABB = new Float32Array([-1, -2, -3, 4, 5, 6]) as AABB
    const m = mat4Create()
    mat4Identity(m)

    const arvo = aabbCreate()
    const brute = aabbCreate()
    aabbTransform(arvo, aabb, m)
    aabbTransformBruteForce(brute, aabb, m)

    for (let i = 0; i < 6; i++) {
      expect(arvo[i]).toBeCloseTo(brute[i]!, 5)
    }
  })

  test('matches brute-force with translation', () => {
    const aabb: AABB = new Float32Array([-1, -1, -1, 1, 1, 1]) as AABB
    const m = mat4Create()
    mat4Identity(m)
    m[12] = 10
    m[13] = 20
    m[14] = 30

    const arvo = aabbCreate()
    const brute = aabbCreate()
    aabbTransform(arvo, aabb, m)
    aabbTransformBruteForce(brute, aabb, m)

    for (let i = 0; i < 6; i++) {
      expect(arvo[i]).toBeCloseTo(brute[i]!, 5)
    }
  })

  test('matches brute-force with uniform scale', () => {
    const aabb: AABB = new Float32Array([0, 0, 0, 2, 3, 4]) as AABB
    const m = mat4Create()
    mat4Identity(m)
    m[0] = 3
    m[5] = 3
    m[10] = 3

    const arvo = aabbCreate()
    const brute = aabbCreate()
    aabbTransform(arvo, aabb, m)
    aabbTransformBruteForce(brute, aabb, m)

    for (let i = 0; i < 6; i++) {
      expect(arvo[i]).toBeCloseTo(brute[i]!, 5)
    }
  })

  test('matches brute-force with rotation (90° around Z)', () => {
    const aabb: AABB = new Float32Array([1, 0, 0, 3, 2, 1]) as AABB
    // 90° rotation around Z: [cos90, sin90, 0; -sin90, cos90, 0; 0, 0, 1]
    const m = mat4Create()
    mat4Identity(m)
    m[0] = 0
    m[1] = 1
    m[4] = -1
    m[5] = 0

    const arvo = aabbCreate()
    const brute = aabbCreate()
    aabbTransform(arvo, aabb, m)
    aabbTransformBruteForce(brute, aabb, m)

    for (let i = 0; i < 6; i++) {
      expect(arvo[i]).toBeCloseTo(brute[i]!, 5)
    }
  })

  test('matches brute-force with combined transform', () => {
    const aabb: AABB = new Float32Array([-2, -1, 0, 3, 4, 5]) as AABB
    // Arbitrary affine transform: rotation + scale + translation
    const m = mat4Create()
    // 45° rotation around Z with 2x scale + translation
    const c = Math.cos(Math.PI / 4) * 2
    const s = Math.sin(Math.PI / 4) * 2
    mat4Identity(m)
    m[0] = c
    m[1] = s
    m[4] = -s
    m[5] = c
    m[10] = 1.5
    m[12] = 5
    m[13] = -3
    m[14] = 7

    const arvo = aabbCreate()
    const brute = aabbCreate()
    aabbTransform(arvo, aabb, m)
    aabbTransformBruteForce(brute, aabb, m)

    for (let i = 0; i < 6; i++) {
      expect(arvo[i]).toBeCloseTo(brute[i]!, 4)
    }
  })

  test('handles negative scale (mirror)', () => {
    const aabb: AABB = new Float32Array([1, 2, 3, 4, 5, 6]) as AABB
    const m = mat4Create()
    mat4Identity(m)
    m[0] = -1 // Mirror X

    const arvo = aabbCreate()
    const brute = aabbCreate()
    aabbTransform(arvo, aabb, m)
    aabbTransformBruteForce(brute, aabb, m)

    for (let i = 0; i < 6; i++) {
      expect(arvo[i]).toBeCloseTo(brute[i]!, 5)
    }
  })
})
