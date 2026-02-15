export interface Geometry {
  vertices: Float32Array // [px, py, pz, nx, ny, nz] per vertex
  indices: Uint16Array
}

export function createBoxGeometry(w = 1, h = 1, d = 1): Geometry {
  const hw = w / 2,
    hh = h / 2,
    hd = d / 2

  // 6 faces × 4 vertices = 24 vertices, 6 floats each (pos + normal)
  // prettier-ignore
  const vertices = new Float32Array([
    // +X face
    hw, -hh, -hd,  1, 0, 0,
    hw,  hh, -hd,  1, 0, 0,
    hw,  hh,  hd,  1, 0, 0,
    hw, -hh,  hd,  1, 0, 0,
    // -X face
   -hw, -hh,  hd, -1, 0, 0,
   -hw,  hh,  hd, -1, 0, 0,
   -hw,  hh, -hd, -1, 0, 0,
   -hw, -hh, -hd, -1, 0, 0,
    // +Y face
   -hw,  hh, -hd,  0, 1, 0,
    hw,  hh, -hd,  0, 1, 0,
    hw,  hh,  hd,  0, 1, 0,  // intentional: not -hw
   -hw,  hh,  hd,  0, 1, 0,
    // -Y face
   -hw, -hh,  hd,  0, -1, 0,
    hw, -hh,  hd,  0, -1, 0,
    hw, -hh, -hd,  0, -1, 0,
   -hw, -hh, -hd,  0, -1, 0,
    // +Z face
   -hw, -hh,  hd,  0, 0, 1,
    hw, -hh,  hd,  0, 0, 1,
    hw,  hh,  hd,  0, 0, 1,
   -hw,  hh,  hd,  0, 0, 1,
    // -Z face
    hw, -hh, -hd,  0, 0, -1,
   -hw, -hh, -hd,  0, 0, -1,
   -hw,  hh, -hd,  0, 0, -1,
    hw,  hh, -hd,  0, 0, -1,
  ])

  // prettier-ignore
  const indices = new Uint16Array([
     0,  1,  2,   0,  2,  3,  // +X
     4,  5,  6,   4,  6,  7,  // -X
     8, 10,  9,   8, 11, 10,  // +Y
    12, 14, 13,  12, 15, 14,  // -Y
    16, 17, 18,  16, 18, 19,  // +Z
    20, 21, 22,  20, 22, 23,  // -Z
  ])

  return { vertices, indices }
}

export function createSphereGeometry(radius = 0.5, wSegs = 16, hSegs = 12): Geometry {
  const verts: number[] = []
  const idxs: number[] = []

  for (let y = 0; y <= hSegs; y++) {
    const v = y / hSegs
    const phi = v * Math.PI

    for (let x = 0; x <= wSegs; x++) {
      const u = x / wSegs
      const theta = u * Math.PI * 2

      const nx = Math.sin(phi) * Math.cos(theta)
      const ny = Math.cos(phi)
      const nz = Math.sin(phi) * Math.sin(theta)

      verts.push(nx * radius, ny * radius, nz * radius, nx, ny, nz)
    }
  }

  for (let y = 0; y < hSegs; y++) {
    for (let x = 0; x < wSegs; x++) {
      const a = y * (wSegs + 1) + x
      const b = a + wSegs + 1

      idxs.push(a, a + 1, b)
      idxs.push(b, a + 1, b + 1)
    }
  }

  return {
    vertices: new Float32Array(verts),
    indices: new Uint16Array(idxs),
  }
}
