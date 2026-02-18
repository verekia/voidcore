import { Geometry } from './geometry.ts'

// All primitives: Z-up, centered at origin

export interface PlaneGeometryOptions {
  width?: number
  height?: number
  widthSegments?: number
  heightSegments?: number
}

export const createPlaneGeometry = (opts: PlaneGeometryOptions = {}): Geometry => {
  const w = opts.width ?? 1,
    h = opts.height ?? 1
  const ws = opts.widthSegments ?? 1,
    hs = opts.heightSegments ?? 1
  const vCount = (ws + 1) * (hs + 1)
  const positions = new Float32Array(vCount * 3)
  const normals = new Float32Array(vCount * 3)
  const uvs = new Float32Array(vCount * 2)
  const indices: number[] = []

  let vi = 0,
    ui = 0
  for (let iy = 0; iy <= hs; iy++) {
    for (let ix = 0; ix <= ws; ix++) {
      positions[vi] = (ix / ws - 0.5) * w // X
      positions[vi + 1] = (iy / hs - 0.5) * h // Y
      positions[vi + 2] = 0 // Z (lies on XY plane)
      normals[vi] = 0
      normals[vi + 1] = 0
      normals[vi + 2] = 1 // +Z normal
      uvs[ui] = ix / ws
      uvs[ui + 1] = iy / hs
      vi += 3
      ui += 2
    }
  }

  for (let iy = 0; iy < hs; iy++) {
    for (let ix = 0; ix < ws; ix++) {
      const a = iy * (ws + 1) + ix
      const b = a + 1
      const c = a + (ws + 1)
      const d = c + 1
      indices.push(a, b, d, a, d, c)
    }
  }

  return new Geometry({
    positions,
    normals,
    uvs,
    indices: vCount > 65535 ? new Uint32Array(indices) : new Uint16Array(indices),
  })
}

export interface BoxGeometryOptions {
  width?: number
  height?: number
  depth?: number
}

export const createBoxGeometry = (opts: BoxGeometryOptions = {}): Geometry => {
  const w = (opts.width ?? 1) / 2
  const h = (opts.height ?? 1) / 2
  const d = (opts.depth ?? 1) / 2

  const p: number[] = []
  const n: number[] = []
  const u: number[] = []
  const idx: number[] = []

  // +X
  p.push(w, -h, -d, w, h, -d, w, h, d, w, -h, d)
  n.push(1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0)
  u.push(0, 0, 1, 0, 1, 1, 0, 1)
  // -X
  p.push(-w, h, -d, -w, -h, -d, -w, -h, d, -w, h, d)
  n.push(-1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0)
  u.push(0, 0, 1, 0, 1, 1, 0, 1)
  // +Y
  p.push(w, h, -d, -w, h, -d, -w, h, d, w, h, d)
  n.push(0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0)
  u.push(0, 0, 1, 0, 1, 1, 0, 1)
  // -Y
  p.push(-w, -h, -d, w, -h, -d, w, -h, d, -w, -h, d)
  n.push(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0)
  u.push(0, 0, 1, 0, 1, 1, 0, 1)
  // +Z (top)
  p.push(-w, -h, d, w, -h, d, w, h, d, -w, h, d)
  n.push(0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1)
  u.push(0, 0, 1, 0, 1, 1, 0, 1)
  // -Z (bottom)
  p.push(-w, h, -d, w, h, -d, w, -h, -d, -w, -h, -d)
  n.push(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1)
  u.push(0, 0, 1, 0, 1, 1, 0, 1)

  for (let i = 0; i < 6; i++) {
    const o = i * 4
    idx.push(o, o + 1, o + 2, o, o + 2, o + 3)
  }

  return new Geometry({
    positions: new Float32Array(p),
    normals: new Float32Array(n),
    uvs: new Float32Array(u),
    indices: new Uint16Array(idx),
  })
}

export interface SphereGeometryOptions {
  radius?: number
  widthSegments?: number
  heightSegments?: number
}

export const createSphereGeometry = (opts: SphereGeometryOptions = {}): Geometry => {
  const r = opts.radius ?? 1
  const ws = opts.widthSegments ?? 32
  const hs = opts.heightSegments ?? 16

  const positions: number[] = []
  const normals: number[] = []
  const uvs: number[] = []
  const indices: number[] = []

  // Z-up sphere: poles along Z axis
  for (let iy = 0; iy <= hs; iy++) {
    const v = iy / hs
    const phi = v * Math.PI // 0 (top/+Z) to PI (bottom/-Z)
    for (let ix = 0; ix <= ws; ix++) {
      const u = ix / ws
      const theta = u * Math.PI * 2

      const sinPhi = Math.sin(phi)
      const cosPhi = Math.cos(phi)
      const x = sinPhi * Math.cos(theta)
      const y = sinPhi * Math.sin(theta)
      const z = cosPhi

      positions.push(x * r, y * r, z * r)
      normals.push(x, y, z)
      uvs.push(u, v)
    }
  }

  for (let iy = 0; iy < hs; iy++) {
    for (let ix = 0; ix < ws; ix++) {
      const a = iy * (ws + 1) + ix
      const b = a + 1
      const c = a + (ws + 1)
      const d = c + 1
      if (iy !== 0) indices.push(a, b, d)
      if (iy !== hs - 1) indices.push(a, d, c)
    }
  }

  const vCount = positions.length / 3
  return new Geometry({
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    uvs: new Float32Array(uvs),
    indices: vCount > 65535 ? new Uint32Array(indices) : new Uint16Array(indices),
  })
}

export interface ConeGeometryOptions {
  radius?: number
  height?: number
  radialSegments?: number
}

export const createConeGeometry = (opts: ConeGeometryOptions = {}): Geometry => {
  const radius = opts.radius ?? 1
  const height = opts.height ?? 1
  const rs = opts.radialSegments ?? 32

  const positions: number[] = []
  const normals: number[] = []
  const uvs: number[] = []
  const indices: number[] = []

  // Z-up: base at z=0, tip at z=height
  const slopeLen = Math.sqrt(radius * radius + height * height)
  const nzSlope = radius / slopeLen
  const nrSlope = height / slopeLen

  // Side vertices
  for (let i = 0; i <= rs; i++) {
    const theta = (i / rs) * Math.PI * 2
    const cos = Math.cos(theta),
      sin = Math.sin(theta)

    // Base vertex
    positions.push(cos * radius, sin * radius, 0)
    normals.push(cos * nrSlope, sin * nrSlope, nzSlope)
    uvs.push(i / rs, 0)

    // Tip vertex
    positions.push(0, 0, height)
    normals.push(cos * nrSlope, sin * nrSlope, nzSlope)
    uvs.push(i / rs, 1)
  }

  for (let i = 0; i < rs; i++) {
    const a = i * 2
    const b = a + 2
    indices.push(a, b, a + 1)
  }

  // Base cap
  const baseCenter = positions.length / 3
  positions.push(0, 0, 0)
  normals.push(0, 0, -1)
  uvs.push(0.5, 0.5)

  for (let i = 0; i <= rs; i++) {
    const theta = (i / rs) * Math.PI * 2
    positions.push(Math.cos(theta) * radius, Math.sin(theta) * radius, 0)
    normals.push(0, 0, -1)
    uvs.push(Math.cos(theta) * 0.5 + 0.5, Math.sin(theta) * 0.5 + 0.5)
  }

  for (let i = 0; i < rs; i++) {
    indices.push(baseCenter, baseCenter + 1 + ((i + 1) % (rs + 1)), baseCenter + 1 + i)
  }

  const vCount = positions.length / 3
  return new Geometry({
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    uvs: new Float32Array(uvs),
    indices: vCount > 65535 ? new Uint32Array(indices) : new Uint16Array(indices),
  })
}

export interface CylinderGeometryOptions {
  radiusTop?: number
  radiusBottom?: number
  height?: number
  radialSegments?: number
}

export const createCylinderGeometry = (opts: CylinderGeometryOptions = {}): Geometry => {
  const rt = opts.radiusTop ?? 1
  const rb = opts.radiusBottom ?? 1
  const h = opts.height ?? 1
  const rs = opts.radialSegments ?? 32
  const halfH = h / 2

  const positions: number[] = []
  const normals: number[] = []
  const uvs: number[] = []
  const indices: number[] = []

  // Side
  const slopeAngle = Math.atan2(rb - rt, h)
  const nzSlope = -Math.sin(slopeAngle)
  const nrSlope = Math.cos(slopeAngle)

  for (let iy = 0; iy <= 1; iy++) {
    const r = iy === 0 ? rb : rt
    const z = iy === 0 ? -halfH : halfH
    for (let ix = 0; ix <= rs; ix++) {
      const theta = (ix / rs) * Math.PI * 2
      const cos = Math.cos(theta),
        sin = Math.sin(theta)
      positions.push(cos * r, sin * r, z)
      normals.push(cos * nrSlope, sin * nrSlope, nzSlope)
      uvs.push(ix / rs, iy)
    }
  }

  for (let ix = 0; ix < rs; ix++) {
    const a = ix,
      b = ix + 1
    const c = ix + (rs + 1),
      d = c + 1
    indices.push(a, b, d, a, d, c)
  }

  // Top cap
  const topCenter = positions.length / 3
  positions.push(0, 0, halfH)
  normals.push(0, 0, 1)
  uvs.push(0.5, 0.5)
  for (let i = 0; i <= rs; i++) {
    const theta = (i / rs) * Math.PI * 2
    positions.push(Math.cos(theta) * rt, Math.sin(theta) * rt, halfH)
    normals.push(0, 0, 1)
    uvs.push(Math.cos(theta) * 0.5 + 0.5, Math.sin(theta) * 0.5 + 0.5)
  }
  for (let i = 0; i < rs; i++) {
    indices.push(topCenter, topCenter + 1 + i, topCenter + 1 + ((i + 1) % (rs + 1)))
  }

  // Bottom cap
  const botCenter = positions.length / 3
  positions.push(0, 0, -halfH)
  normals.push(0, 0, -1)
  uvs.push(0.5, 0.5)
  for (let i = 0; i <= rs; i++) {
    const theta = (i / rs) * Math.PI * 2
    positions.push(Math.cos(theta) * rb, Math.sin(theta) * rb, -halfH)
    normals.push(0, 0, -1)
    uvs.push(Math.cos(theta) * 0.5 + 0.5, Math.sin(theta) * 0.5 + 0.5)
  }
  for (let i = 0; i < rs; i++) {
    indices.push(botCenter, botCenter + 1 + ((i + 1) % (rs + 1)), botCenter + 1 + i)
  }

  const vCount = positions.length / 3
  return new Geometry({
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    uvs: new Float32Array(uvs),
    indices: vCount > 65535 ? new Uint32Array(indices) : new Uint16Array(indices),
  })
}

export interface CapsuleGeometryOptions {
  radius?: number
  height?: number
  radialSegments?: number
  heightSegments?: number
}

export const createCapsuleGeometry = (opts: CapsuleGeometryOptions = {}): Geometry => {
  const r = opts.radius ?? 0.5
  const totalH = opts.height ?? 1
  const rs = opts.radialSegments ?? 32
  const capSegs = 8
  const cylH = Math.max(0, totalH - 2 * r)
  const halfCylH = cylH / 2

  const positions: number[] = []
  const normals: number[] = []
  const uvs: number[] = []
  const indices: number[] = []

  // Top hemisphere (z-up)
  for (let iy = 0; iy <= capSegs; iy++) {
    const phi = (iy / capSegs) * (Math.PI / 2) // 0 to PI/2
    for (let ix = 0; ix <= rs; ix++) {
      const theta = (ix / rs) * Math.PI * 2
      const sinPhi = Math.sin(phi),
        cosPhi = Math.cos(phi)
      const x = cosPhi * Math.cos(theta)
      const y = cosPhi * Math.sin(theta)
      const z = sinPhi
      positions.push(x * r, y * r, z * r + halfCylH)
      normals.push(x, y, z)
      uvs.push(ix / rs, 1 - iy / (capSegs * 2))
    }
  }

  // Bottom hemisphere
  for (let iy = 0; iy <= capSegs; iy++) {
    const phi = (iy / capSegs) * (Math.PI / 2) // 0 to PI/2
    for (let ix = 0; ix <= rs; ix++) {
      const theta = (ix / rs) * Math.PI * 2
      const sinPhi = Math.sin(phi),
        cosPhi = Math.cos(phi)
      const x = cosPhi * Math.cos(theta)
      const y = cosPhi * Math.sin(theta)
      const z = -sinPhi
      positions.push(x * r, y * r, z * r - halfCylH)
      normals.push(x, y, z)
      uvs.push(ix / rs, 0.5 + iy / (capSegs * 2))
    }
  }

  const totalRows = (capSegs + 1) * 2
  for (let iy = 0; iy < totalRows - 1; iy++) {
    for (let ix = 0; ix < rs; ix++) {
      const a = iy * (rs + 1) + ix
      const b = a + 1
      const c = a + (rs + 1)
      const d = c + 1
      indices.push(a, b, d, a, d, c)
    }
  }

  const vCount = positions.length / 3
  return new Geometry({
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    uvs: new Float32Array(uvs),
    indices: vCount > 65535 ? new Uint32Array(indices) : new Uint16Array(indices),
  })
}

export interface CircleGeometryOptions {
  radius?: number
  segments?: number
}

export const createCircleGeometry = (opts: CircleGeometryOptions = {}): Geometry => {
  const r = opts.radius ?? 1
  const segs = opts.segments ?? 32

  const positions: number[] = [0, 0, 0] // Center
  const normals: number[] = [0, 0, 1]
  const uvs: number[] = [0.5, 0.5]
  const indices: number[] = []

  for (let i = 0; i <= segs; i++) {
    const theta = (i / segs) * Math.PI * 2
    const x = Math.cos(theta) * r
    const y = Math.sin(theta) * r
    positions.push(x, y, 0)
    normals.push(0, 0, 1)
    uvs.push(Math.cos(theta) * 0.5 + 0.5, Math.sin(theta) * 0.5 + 0.5)
  }

  for (let i = 0; i < segs; i++) {
    indices.push(0, i + 1, ((i + 1) % segs) + 1)
  }

  return new Geometry({
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    uvs: new Float32Array(uvs),
    indices: new Uint16Array(indices),
  })
}
