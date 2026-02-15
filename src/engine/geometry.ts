export interface Geometry {
  vertices: Float32Array // [px, py, pz, nx, ny, nz, cr, cg, cb, bloom] per vertex
  indices: Uint16Array | Uint32Array
  uvs?: Float32Array // [u, v] per vertex (optional, for textured meshes)
}

export function createBoxGeometry(w = 1, h = 1, d = 1): Geometry {
  const hw = w / 2,
    hh = h / 2,
    hd = d / 2

  // 6 faces × 4 vertices = 24 vertices, 10 floats each (pos + normal + color + bloom)
  // prettier-ignore
  const vertices = new Float32Array([
    // +X face
    hw, -hh, -hd,  1, 0, 0,  1, 1, 1, 0,
    hw,  hh, -hd,  1, 0, 0,  1, 1, 1, 0,
    hw,  hh,  hd,  1, 0, 0,  1, 1, 1, 0,
    hw, -hh,  hd,  1, 0, 0,  1, 1, 1, 0,
    // -X face
   -hw, -hh,  hd, -1, 0, 0,  1, 1, 1, 0,
   -hw,  hh,  hd, -1, 0, 0,  1, 1, 1, 0,
   -hw,  hh, -hd, -1, 0, 0,  1, 1, 1, 0,
   -hw, -hh, -hd, -1, 0, 0,  1, 1, 1, 0,
    // +Y face
   -hw,  hh, -hd,  0, 1, 0,  1, 1, 1, 0,
    hw,  hh, -hd,  0, 1, 0,  1, 1, 1, 0,
    hw,  hh,  hd,  0, 1, 0,  1, 1, 1, 0,  // intentional: not -hw
   -hw,  hh,  hd,  0, 1, 0,  1, 1, 1, 0,
    // -Y face
   -hw, -hh,  hd,  0, -1, 0,  1, 1, 1, 0,
    hw, -hh,  hd,  0, -1, 0,  1, 1, 1, 0,
    hw, -hh, -hd,  0, -1, 0,  1, 1, 1, 0,
   -hw, -hh, -hd,  0, -1, 0,  1, 1, 1, 0,
    // +Z face
   -hw, -hh,  hd,  0, 0, 1,  1, 1, 1, 0,
    hw, -hh,  hd,  0, 0, 1,  1, 1, 1, 0,
    hw,  hh,  hd,  0, 0, 1,  1, 1, 1, 0,
   -hw,  hh,  hd,  0, 0, 1,  1, 1, 1, 0,
    // -Z face
    hw, -hh, -hd,  0, 0, -1,  1, 1, 1, 0,
   -hw, -hh, -hd,  0, 0, -1,  1, 1, 1, 0,
   -hw,  hh, -hd,  0, 0, -1,  1, 1, 1, 0,
    hw,  hh, -hd,  0, 0, -1,  1, 1, 1, 0,
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

// ── Merge primitives ────────────────────────────────────────────────

export function mergeGeometries(
  primitives: {
    vertices: Float32Array
    indices: Uint16Array | Uint32Array
    color: [number, number, number]
    bloom?: number
    uvs?: Float32Array
  }[],
): Geometry {
  let totalVerts = 0
  let totalIdxs = 0
  const hasUVs = primitives.some(p => p.uvs)
  for (const p of primitives) {
    totalVerts += p.vertices.length / 10
    totalIdxs += p.indices.length
  }
  const vertices = new Float32Array(totalVerts * 10)
  const indices = new Uint32Array(totalIdxs)
  const uvs = hasUVs ? new Float32Array(totalVerts * 2) : undefined
  let vertOff = 0
  let idxOff = 0
  for (const p of primitives) {
    const vCount = p.vertices.length / 10
    for (let v = 0; v < vCount; v++) {
      const src = v * 10
      const dst = (vertOff + v) * 10
      vertices[dst]! = p.vertices[src]!
      vertices[dst + 1]! = p.vertices[src + 1]!
      vertices[dst + 2]! = p.vertices[src + 2]!
      vertices[dst + 3]! = p.vertices[src + 3]!
      vertices[dst + 4]! = p.vertices[src + 4]!
      vertices[dst + 5]! = p.vertices[src + 5]!
      vertices[dst + 6]! = p.color[0]
      vertices[dst + 7]! = p.color[1]
      vertices[dst + 8]! = p.color[2]
      vertices[dst + 9]! = p.bloom ?? 0
      if (uvs) {
        const uvDst = (vertOff + v) * 2
        if (p.uvs) {
          uvs[uvDst] = p.uvs[v * 2]!
          uvs[uvDst + 1] = p.uvs[v * 2 + 1]!
        }
      }
    }
    for (let j = 0; j < p.indices.length; j++) {
      indices[idxOff + j] = p.indices[j]! + vertOff
    }
    vertOff += vCount
    idxOff += p.indices.length
  }
  return uvs ? { vertices, indices, uvs } : { vertices, indices }
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

      verts.push(nx * radius, ny * radius, nz * radius, nx, ny, nz, 1, 1, 1, 0)
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
