// Bake GI Probes – CPU-based probe baking from scene geometry.
//
// Fills a GIProbeGrid with indirect lighting data by sampling nearby scene geometry.
// For each probe position, iterates over all triangles within a search radius and
// accumulates their color contributions into L1 Spherical Harmonics coefficients.
//
// Each triangle contributes color based on:
//   - Its vertex colors (from bakePalette or direct assignment)
//   - Its face normal (only front-facing triangles contribute — light bounces off surfaces)
//   - The solid angle it subtends from the probe (area / distance², clamped)
//   - The direction from the probe, projected onto the L1 SH basis [1, x, y, z]
//
// After accumulating geometry contributions, an optional sky color is added for probes
// with upward visibility (the Z+ hemisphere), simulating ambient sky illumination.
//
// The optional `maxDistance` parameter culls probes that are too far from any mesh surface.
// Probes beyond this distance get zeroed out (neutral tint in the shader). This avoids
// wasting computation on probes floating in empty space (e.g. high in the air above terrain).
//
// The optional `cullInterior` flag detects probes that are inside solid terrain geometry.
// For each probe, it finds the nearest triangle and checks the signed distance from the probe
// to that triangle's plane: if negative (probe is on the back-face side), the probe is buried
// underground and gets zeroed out. This prevents incorrect lighting from probes embedded in
// terrain surfaces.
//
// The optional `surfaceOnly` flag keeps only probes on the outer surface of the world mesh.
// First, every probe is classified as inside or outside using the nearest-triangle signed
// distance. Then only outside probes that neighbor at least one inside cell are kept — giving
// a single layer of probes hugging the exterior of the mesh. Inside probes (embedded in
// terrain) and distant outside probes (floating in empty space) are all zeroed out. This
// gives full coverage with no wasted probes and no need for maxDistance.
//
// bakeGIProbes(grid, meshes, options?) – Fills the grid with baked indirect lighting data.

import type { Geometry } from '../geometry/geometry'
import type { GIProbeGrid } from './gi-probes'

export interface BakeGIMesh {
  /** Geometry with vertex colors (from bakePalette). Must have `colors` attribute. */
  geometry: Geometry
  /** World-space offset of this mesh (default [0,0,0]). */
  position?: [number, number, number]
}

export interface BakeGIOptions {
  /** Search radius around each probe (default: 2× the largest grid cell dimension). */
  radius?: number
  /** Maximum distance from any mesh surface for a probe to be active.
   *  Probes farther than this from all geometry are zeroed out (no tint).
   *  When unset, all probes are baked regardless of distance. */
  maxDistance?: number
  /** Sky color to add for upward-facing probes (default: [0.4, 0.6, 0.9]). */
  skyColor?: [number, number, number]
  /** Sky intensity multiplier (default: 0.15). */
  skyIntensity?: number
  /** Cull probes that are inside terrain geometry.
   *  For each probe, finds the nearest triangle and checks if the probe is on the
   *  back-face side of that triangle's plane. If so, the probe is underground and
   *  gets zeroed out. Default: false. */
  cullInterior?: boolean
  /** Keep only probes on the outer surface of the world mesh.
   *  Classifies each probe as inside/outside, then keeps only boundary probes
   *  (those with at least one neighbor of a different classification). All deep
   *  interior and far exterior probes are zeroed out. Overrides maxDistance and
   *  cullInterior. Default: false. */
  surfaceOnly?: boolean
}

/**
 * Bake GI probes from scene geometry.
 *
 * Samples nearby triangles' vertex colors and accumulates into L1 SH coefficients.
 * The grid's `needsUpdate` flag is set automatically.
 */
export const bakeGIProbes = (grid: GIProbeGrid, meshes: BakeGIMesh[], options?: BakeGIOptions): void => {
  const [rx, ry, rz] = grid.resolution
  const bMin = grid.boundsMin
  const bMax = grid.boundsMax
  const stepX = rx > 1 ? (bMax[0] - bMin[0]) / (rx - 1) : 0
  const stepY = ry > 1 ? (bMax[1] - bMin[1]) / (ry - 1) : 0
  const stepZ = rz > 1 ? (bMax[2] - bMin[2]) / (rz - 1) : 0

  const defaultRadius = 2 * Math.max(stepX, stepY, stepZ)
  const radius = options?.radius ?? defaultRadius
  const radiusSq = radius * radius
  const maxDistance = options?.maxDistance ?? 0
  const maxDistSq = maxDistance > 0 ? maxDistance * maxDistance : 0
  const skyColor = options?.skyColor ?? [0.4, 0.6, 0.9]
  const skyIntensity = options?.skyIntensity ?? 0.15
  const cullInterior = options?.cullInterior ?? false
  const surfaceOnly = options?.surfaceOnly ?? false

  // Pre-extract all triangle data from meshes (centroid, normal, color, area)
  // to avoid per-probe per-mesh iteration overhead
  const triData = _extractTriangles(meshes)

  // For each probe, accumulate contributions from nearby triangles
  for (let iz = 0; iz < rz; iz++) {
    for (let iy = 0; iy < ry; iy++) {
      for (let ix = 0; ix < rx; ix++) {
        const px = bMin[0] + ix * stepX
        const py = bMin[1] + iy * stepY
        const pz = bMin[2] + iz * stepZ

        // Skip probes too far from any mesh surface
        if (maxDistSq > 0 || cullInterior) {
          let nearestSq = Infinity
          let nearestIdx = -1
          for (let t = 0; t < triData.count; t++) {
            const off = t * 13
            const dx = triData.data[off]! - px
            const dy = triData.data[off + 1]! - py
            const dz = triData.data[off + 2]! - pz
            const dSq = dx * dx + dy * dy + dz * dz
            if (dSq < nearestSq) {
              nearestSq = dSq
              nearestIdx = t
              if (!cullInterior && nearestSq <= maxDistSq) break // Early exit: close enough
            }
          }
          if (maxDistSq > 0 && nearestSq > maxDistSq) {
            // Zero out this probe (neutral tint in shader)
            const probeOff = (iz * ry * rx + iy * rx + ix) * 12
            for (let k = 0; k < 12; k++) grid.data[probeOff + k] = 0
            continue
          }
          // Cull probes inside terrain: check if probe is on the back-face side
          // of the nearest triangle's plane (signed distance < 0 means underground)
          if (cullInterior && nearestIdx >= 0) {
            const off = nearestIdx * 13
            // Vector from triangle centroid to probe
            const toPx = px - triData.data[off]!
            const toPy = py - triData.data[off + 1]!
            const toPz = pz - triData.data[off + 2]!
            // Dot with face normal: negative means probe is on the back side
            const signedDist =
              toPx * triData.data[off + 3]! + toPy * triData.data[off + 4]! + toPz * triData.data[off + 5]!
            if (signedDist < 0) {
              const probeOff = (iz * ry * rx + iy * rx + ix) * 12
              for (let k = 0; k < 12; k++) grid.data[probeOff + k] = 0
              continue
            }
          }
        }

        // SH accumulators: [dc, x, y, z] per channel
        let rDC = 0,
          rX = 0,
          rY = 0,
          rZ = 0
        let gDC = 0,
          gX = 0,
          gY = 0,
          gZ = 0
        let bDC = 0,
          bX = 0,
          bY = 0,
          bZ = 0
        let totalWeight = 0

        for (let t = 0; t < triData.count; t++) {
          const off = t * 13
          const cx = triData.data[off]! // centroid x
          const cy = triData.data[off + 1]! // centroid y
          const cz = triData.data[off + 2]! // centroid z

          // Distance check
          const dx = cx - px
          const dy = cy - py
          const dz = cz - pz
          const distSq = dx * dx + dy * dy + dz * dz
          if (distSq > radiusSq) continue

          const dist = Math.sqrt(distSq)
          if (dist < 0.01) continue // Skip degenerate (probe inside triangle)

          // Direction from probe to triangle centroid (normalized)
          const invDist = 1 / dist
          const dirX = dx * invDist
          const dirY = dy * invDist
          const dirZ = dz * invDist

          // Face normal
          const nx = triData.data[off + 3]!
          const ny = triData.data[off + 4]!
          const nz = triData.data[off + 5]!

          // Only count front-facing triangles (normal faces toward probe)
          const facing = -(dirX * nx + dirY * ny + dirZ * nz) // dot(normal, -dir)
          if (facing <= 0) continue

          // Triangle area
          const area = triData.data[off + 6]!

          // Form factor: how much light this triangle sends toward the probe
          // area × cos(angle between normal and probe direction) / distance²
          const formFactor = (area * facing) / (distSq + 1) // +1 avoids singularity

          // Vertex color
          const cr = triData.data[off + 7]!
          const cg = triData.data[off + 8]!
          const cb = triData.data[off + 9]!

          // Accumulate SH L1: basis functions are [1, dir.x, dir.y, dir.z]
          const w = formFactor
          rDC += cr * w
          rX += cr * w * dirX
          rY += cr * w * dirY
          rZ += cr * w * dirZ
          gDC += cg * w
          gX += cg * w * dirX
          gY += cg * w * dirY
          gZ += cg * w * dirZ
          bDC += cb * w
          bX += cb * w * dirX
          bY += cb * w * dirY
          bZ += cb * w * dirZ
          totalWeight += w
        }

        // Normalize by total weight
        if (totalWeight > 0.001) {
          const inv = 1 / totalWeight
          rDC *= inv
          rX *= inv
          rY *= inv
          rZ *= inv
          gDC *= inv
          gX *= inv
          gY *= inv
          gZ *= inv
          bDC *= inv
          bX *= inv
          bY *= inv
          bZ *= inv
        }

        // Add sky contribution (top hemisphere, Z+)
        const skyR = skyColor[0] * skyIntensity
        const skyG = skyColor[1] * skyIntensity
        const skyB = skyColor[2] * skyIntensity
        rDC += skyR
        gDC += skyG
        bDC += skyB
        // Sky comes from +Z direction
        rZ += skyR * 0.5
        gZ += skyG * 0.5
        bZ += skyB * 0.5

        // Write raw SH coefficients
        const probeOff = (iz * ry * rx + iy * rx + ix) * 12
        grid.data[probeOff] = rDC
        grid.data[probeOff + 1] = rX
        grid.data[probeOff + 2] = rY
        grid.data[probeOff + 3] = rZ
        grid.data[probeOff + 4] = gDC
        grid.data[probeOff + 5] = gX
        grid.data[probeOff + 6] = gY
        grid.data[probeOff + 7] = gZ
        grid.data[probeOff + 8] = bDC
        grid.data[probeOff + 9] = bX
        grid.data[probeOff + 10] = bY
        grid.data[probeOff + 11] = bZ
      }
    }
  }

  // Surface-only pass: classify probes as inside/outside, keep only boundary probes
  if (surfaceOnly) {
    _cullToSurface(grid, triData)
  }

  grid.needsUpdate = true
}

// ─── Internal: surface-only culling ──────────────────────────────────

/**
 * Classify every probe as inside (1) or outside (0) the mesh using the nearest
 * triangle's signed distance. Then keep only outside probes that neighbor at least
 * one inside cell — a single layer hugging the mesh exterior. Inside probes are
 * always culled. Boundary probes must also be within proximity of actual geometry
 * (2× max cell step) to avoid false boundaries from noisy classification of distant probes.
 */
const _cullToSurface = (grid: GIProbeGrid, triData: TriangleData): void => {
  const [rx, ry, rz] = grid.resolution
  const bMin = grid.boundsMin
  const bMax = grid.boundsMax
  const stepX = rx > 1 ? (bMax[0] - bMin[0]) / (rx - 1) : 0
  const stepY = ry > 1 ? (bMax[1] - bMin[1]) / (ry - 1) : 0
  const stepZ = rz > 1 ? (bMax[2] - bMin[2]) / (rz - 1) : 0
  const probeCount = rx * ry * rz

  // Proximity threshold: probes farther than this from any triangle are not surface probes
  const maxStep = Math.max(stepX, stepY, stepZ)
  const proxThresholdSq = (2 * maxStep) * (2 * maxStep)

  // 1) Classify each probe: 1 = inside, 0 = outside. Also store nearest distance².
  const inside = new Uint8Array(probeCount)
  const nearestDistSq = new Float32Array(probeCount)
  for (let iz = 0; iz < rz; iz++) {
    for (let iy = 0; iy < ry; iy++) {
      for (let ix = 0; ix < rx; ix++) {
        const px = bMin[0] + ix * stepX
        const py = bMin[1] + iy * stepY
        const pz = bMin[2] + iz * stepZ
        const idx = iz * ry * rx + iy * rx + ix

        // Find nearest triangle
        let bestSq = Infinity
        let nearestIdx = -1
        for (let t = 0; t < triData.count; t++) {
          const off = t * 13
          const dx = triData.data[off]! - px
          const dy = triData.data[off + 1]! - py
          const dz = triData.data[off + 2]! - pz
          const dSq = dx * dx + dy * dy + dz * dz
          if (dSq < bestSq) {
            bestSq = dSq
            nearestIdx = t
          }
        }

        nearestDistSq[idx] = bestSq

        if (nearestIdx >= 0) {
          const off = nearestIdx * 13
          // Signed distance: dot(probe - centroid, face normal)
          const toPx = px - triData.data[off]!
          const toPy = py - triData.data[off + 1]!
          const toPz = pz - triData.data[off + 2]!
          const signedDist =
            toPx * triData.data[off + 3]! + toPy * triData.data[off + 4]! + toPz * triData.data[off + 5]!
          if (signedDist < 0) {
            inside[idx] = 1
          }
        }
      }
    }
  }

  // 2) Keep only outside boundary probes (outside probes neighboring at least one inside cell).
  //    This gives a single-layer shell on the exterior of the mesh, rather than a two-layer
  //    boundary (inside+outside) which wastes probes embedded under surfaces.
  for (let iz = 0; iz < rz; iz++) {
    for (let iy = 0; iy < ry; iy++) {
      for (let ix = 0; ix < rx; ix++) {
        const idx = iz * ry * rx + iy * rx + ix
        const val = inside[idx]!

        // Inside probes are always culled — we only keep exterior probes
        if (val === 1) {
          const probeOff = idx * 12
          for (let k = 0; k < 12; k++) grid.data[probeOff + k] = 0
          continue
        }

        // Outside probe: keep only if it neighbors at least one inside cell
        let hasInsideNeighbor = false
        if (ix > 0 && inside[idx - 1] === 1) hasInsideNeighbor = true
        else if (ix < rx - 1 && inside[idx + 1] === 1) hasInsideNeighbor = true
        else if (iy > 0 && inside[idx - rx] === 1) hasInsideNeighbor = true
        else if (iy < ry - 1 && inside[idx + rx] === 1) hasInsideNeighbor = true
        else if (iz > 0 && inside[idx - ry * rx] === 1) hasInsideNeighbor = true
        else if (iz < rz - 1 && inside[idx + ry * rx] === 1) hasInsideNeighbor = true

        // Outside probes must also be near actual geometry to avoid false boundaries
        // from noisy inside/outside classification of distant probes
        if (hasInsideNeighbor && nearestDistSq[idx]! > proxThresholdSq) {
          hasInsideNeighbor = false
        }

        if (!hasInsideNeighbor) {
          const probeOff = idx * 12
          for (let k = 0; k < 12; k++) grid.data[probeOff + k] = 0
        }
      }
    }
  }
}

// ─── Internal: pre-extract triangle data ─────────────────────────────

interface TriangleData {
  /** Packed per-triangle: [cx, cy, cz, nx, ny, nz, area, cr, cg, cb, 0, 0, 0] × count */
  data: Float32Array
  count: number
}

const _extractTriangles = (meshes: BakeGIMesh[]): TriangleData => {
  // Count total triangles
  let totalTris = 0
  for (const m of meshes) {
    totalTris += m.geometry.indexCount / 3
  }

  // 13 floats per triangle (padded for alignment)
  const data = new Float32Array(totalTris * 13)
  let triIdx = 0

  for (const m of meshes) {
    const geo = m.geometry
    const positions = geo.positions
    const colors = geo.colors
    const indices = geo.indices
    const ox = m.position?.[0] ?? 0
    const oy = m.position?.[1] ?? 0
    const oz = m.position?.[2] ?? 0

    if (!colors) continue // Skip meshes without vertex colors

    for (let i = 0; i < indices.length; i += 3) {
      const i0 = indices[i]!
      const i1 = indices[i + 1]!
      const i2 = indices[i + 2]!

      // Vertex positions in world space
      const x0 = positions[i0 * 3]! + ox
      const y0 = positions[i0 * 3 + 1]! + oy
      const z0 = positions[i0 * 3 + 2]! + oz
      const x1 = positions[i1 * 3]! + ox
      const y1 = positions[i1 * 3 + 1]! + oy
      const z1 = positions[i1 * 3 + 2]! + oz
      const x2 = positions[i2 * 3]! + ox
      const y2 = positions[i2 * 3 + 1]! + oy
      const z2 = positions[i2 * 3 + 2]! + oz

      // Centroid
      const cx = (x0 + x1 + x2) / 3
      const cy = (y0 + y1 + y2) / 3
      const cz = (z0 + z1 + z2) / 3

      // Face normal (cross product of edges)
      const e1x = x1 - x0,
        e1y = y1 - y0,
        e1z = z1 - z0
      const e2x = x2 - x0,
        e2y = y2 - y0,
        e2z = z2 - z0
      let fnx = e1y * e2z - e1z * e2y
      let fny = e1z * e2x - e1x * e2z
      let fnz = e1x * e2y - e1y * e2x
      const fnLen = Math.sqrt(fnx * fnx + fny * fny + fnz * fnz)

      // Triangle area = half the cross product magnitude
      const area = fnLen * 0.5
      if (area < 0.0001) continue // Skip degenerate triangles

      // Normalize face normal
      const invFnLen = 1 / fnLen
      fnx *= invFnLen
      fny *= invFnLen
      fnz *= invFnLen

      // Average vertex color (RGB only, skip alpha)
      const cr = (colors[i0 * 4]! + colors[i1 * 4]! + colors[i2 * 4]!) / 3
      const cg = (colors[i0 * 4 + 1]! + colors[i1 * 4 + 1]! + colors[i2 * 4 + 1]!) / 3
      const cb = (colors[i0 * 4 + 2]! + colors[i1 * 4 + 2]! + colors[i2 * 4 + 2]!) / 3

      const off = triIdx * 13
      data[off] = cx
      data[off + 1] = cy
      data[off + 2] = cz
      data[off + 3] = fnx
      data[off + 4] = fny
      data[off + 5] = fnz
      data[off + 6] = area
      data[off + 7] = cr
      data[off + 8] = cg
      data[off + 9] = cb
      triIdx++
    }
  }

  return { data, count: triIdx }
}
