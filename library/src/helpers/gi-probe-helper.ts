// GI Probe Helper – Debug visualization for GI probe grids.
//
// Renders each probe in a GIProbeGrid as a small colored sphere showing its full L1 SH
// irradiance — each vertex on the sphere is colored by evaluating the SH in that vertex's
// direction, so you can see both the average color and the directional variation at each
// probe location. This is much more informative than showing only the DC (constant) term.
//
// Colors are auto-normalized so the brightest value across all probes maps to full white,
// making the relative spatial and directional variation clearly visible.
//
// Probes with zero DC values (inactive/culled by maxDistance) are skipped entirely — no geometry
// is generated for them.
//
// Implementation: Generates a merged geometry with one low-poly sphere per active probe. Per-vertex
// SH-evaluated colors are stored in the normal attribute (abusing normals as a color channel), and
// a custom shader on a BasicMaterial reads the raw normals in the vertex stage — bypassing the
// normal-matrix transform — then outputs them as fragment color.
//
// new GIProbeHelper(grid, opts?) – Creates the helper. Add helper.mesh to the scene.
// helper.update(grid)            – Rebuilds colors when probe data changes.
// helper.dispose()               – Releases geometry resources.

import { Geometry } from '../geometry/geometry'
import { BasicMaterial } from '../materials/material'
import { Mesh } from '../scene/mesh'

import type { GIProbeGrid } from '../scene/gi-probes'

// Low-poly sphere parameters (per probe)
const WS = 6 // width segments (around equator)
const HS = 4 // height segments (pole to pole)
const VERTS_PER_PROBE = (WS + 1) * (HS + 1) // 35

export interface GIProbeHelperOptions {
  /** Radius of each probe sphere (default 0.4). */
  radius?: number
}

/** Check if a probe has any non-zero DC data (i.e. it's active). */
const _isProbeActive = (data: Float32Array, probeIndex: number): boolean => {
  const off = probeIndex * 12
  return Math.abs(data[off]!) > 0.0001 || Math.abs(data[off + 4]!) > 0.0001 || Math.abs(data[off + 8]!) > 0.0001
}

export class GIProbeHelper {
  mesh: Mesh
  private _geometry: Geometry
  private _radius: number
  /** Per-vertex unit sphere directions (xyz), same layout as normals. */
  private _directions: Float32Array

  constructor(grid: GIProbeGrid, options?: GIProbeHelperOptions) {
    this._radius = options?.radius ?? 0.4

    // Build merged sphere geometry (only for active probes)
    const { positions, normals, uvs, indices, directions } = this._buildMergedSpheres(grid)
    this._directions = directions
    this._geometry = new Geometry({
      positions,
      normals,
      indices,
      uvs,
    })

    // Write L1 SH-evaluated probe colors into normals
    this._writeColors(grid)

    const material = new BasicMaterial({
      color: [1, 1, 1],
      customShader: {
        // Bypass normal-matrix transform: use raw attribute as color data
        vertexWGSL: 'out.normal = a_normal.xyz;',
        vertexGLSL: 'v_normal = a_normal.xyz;',
        // Display normalized probe color (brightest probe = white)
        fragmentWGSL: 'finalColor = max(in.normal, vec3f(0.0));',
        fragmentGLSL: 'finalColor = max(v_normal, vec3(0.0));',
      },
    })

    this.mesh = new Mesh(this._geometry, material)
    this.mesh.frustumCulled = false
    this.mesh.castShadow = false
  }

  /** Rebuild probe colors when grid data changes. */
  update(grid: GIProbeGrid): void {
    this._writeColors(grid)
    this._geometry.needsUpdate = true
  }

  dispose(): void {
    this._geometry.dispose()
  }

  /**
   * Build merged sphere geometry for active probes only. Positions are computed from the
   * grid's bounds and resolution. Normals are placeholder (overwritten by _writeColors).
   * Also returns a directions array with per-vertex unit sphere directions for SH evaluation.
   */
  private _buildMergedSpheres(grid: GIProbeGrid): {
    positions: Float32Array
    normals: Float32Array
    uvs: Float32Array
    indices: Uint16Array | Uint32Array
    directions: Float32Array
  } {
    const [rx, ry, rz] = grid.resolution
    const probeCount = rx * ry * rz

    // Find active probes (non-zero DC)
    const activeIndices: number[] = []
    for (let i = 0; i < probeCount; i++) {
      if (_isProbeActive(grid.data, i)) activeIndices.push(i)
    }
    const activeCount = activeIndices.length

    // Compute indices per probe sphere
    const indicesPerProbe: number[] = []
    for (let iy = 0; iy < HS; iy++) {
      for (let ix = 0; ix < WS; ix++) {
        const a = iy * (WS + 1) + ix
        const b = a + 1
        const c = a + (WS + 1)
        const d = c + 1
        if (iy !== 0) indicesPerProbe.push(a, d, b)
        if (iy !== HS - 1) indicesPerProbe.push(a, c, d)
      }
    }
    const idxPerProbe = indicesPerProbe.length

    const totalVerts = activeCount * VERTS_PER_PROBE
    const totalIndices = activeCount * idxPerProbe
    const positions = new Float32Array(totalVerts * 3)
    const normals = new Float32Array(totalVerts * 3)
    const directions = new Float32Array(totalVerts * 3)
    const uvs = new Float32Array(totalVerts * 2)
    const useUint32 = totalVerts > 65535
    const indices = useUint32 ? new Uint32Array(totalIndices) : new Uint16Array(totalIndices)

    const r = this._radius
    const bMin = grid.boundsMin
    const bMax = grid.boundsMax
    const sx = rx > 1 ? (bMax[0] - bMin[0]) / (rx - 1) : 0
    const sy = ry > 1 ? (bMax[1] - bMin[1]) / (ry - 1) : 0
    const sz = rz > 1 ? (bMax[2] - bMin[2]) / (rz - 1) : 0

    for (let ai = 0; ai < activeCount; ai++) {
      const probeIdx = activeIndices[ai]!
      const ix = probeIdx % rx
      const iy = Math.floor(probeIdx / rx) % ry
      const iz = Math.floor(probeIdx / (rx * ry))

      // World position of this probe
      const cx = bMin[0] + ix * sx
      const cy = bMin[1] + iy * sy
      const cz = bMin[2] + iz * sz

      const vOff = ai * VERTS_PER_PROBE
      const iOff = ai * idxPerProbe

      // Generate sphere vertices (Z-up, same as SphereGeometry)
      let vi = 0
      for (let sy2 = 0; sy2 <= HS; sy2++) {
        const v = sy2 / HS
        const phi = v * Math.PI // 0 (top/+Z) to PI (bottom/-Z)
        const sinPhi = Math.sin(phi)
        const cosPhi = Math.cos(phi)

        for (let sx2 = 0; sx2 <= WS; sx2++) {
          const u = sx2 / WS
          const theta = u * Math.PI * 2
          const nx = sinPhi * Math.cos(theta)
          const ny = sinPhi * Math.sin(theta)
          const nz = cosPhi

          const p = (vOff + vi) * 3
          positions[p] = cx + nx * r
          positions[p + 1] = cy + ny * r
          positions[p + 2] = cz + nz * r

          // Store unit sphere direction for SH evaluation in _writeColors
          directions[p] = nx
          directions[p + 1] = ny
          directions[p + 2] = nz

          // Normals will be overwritten by _writeColors
          normals[p] = 0
          normals[p + 1] = 0
          normals[p + 2] = 1

          const uv = (vOff + vi) * 2
          uvs[uv] = u
          uvs[uv + 1] = v

          vi++
        }
      }

      // Write indices (offset by vertex base)
      for (let i = 0; i < idxPerProbe; i++) {
        indices[iOff + i] = vOff + indicesPerProbe[i]!
      }
    }

    return { positions, normals, uvs, indices, directions }
  }

  /**
   * Evaluate full L1 SH per vertex and write normalized colors into normals.
   * Each vertex gets: color(dir) = c0 + c1*dir.x + c2*dir.y + c3*dir.z per channel.
   * The brightest value across all probes/vertices maps to 1.0.
   */
  private _writeColors(grid: GIProbeGrid): void {
    const [rx, ry, rz] = grid.resolution
    const normals = this._geometry.normals
    const dirs = this._directions
    const probeCount = rx * ry * rz

    // First pass: find max SH-evaluated value across all active probes for normalization.
    // Sample a few representative directions per probe (±X, ±Y, ±Z) for efficiency.
    let maxVal = 0
    for (let i = 0; i < probeCount; i++) {
      if (!_isProbeActive(grid.data, i)) continue
      const off = i * 12
      const rDC = grid.data[off]!,     rX = grid.data[off + 1]!, rY = grid.data[off + 2]!, rZ = grid.data[off + 3]!
      const gDC = grid.data[off + 4]!, gX = grid.data[off + 5]!, gY = grid.data[off + 6]!, gZ = grid.data[off + 7]!
      const bDC = grid.data[off + 8]!, bX = grid.data[off + 9]!, bY = grid.data[off + 10]!, bZ = grid.data[off + 11]!
      // Max possible value: DC + |directional| (when direction aligns perfectly)
      maxVal = Math.max(maxVal,
        rDC + Math.abs(rX), rDC + Math.abs(rY), rDC + Math.abs(rZ),
        gDC + Math.abs(gX), gDC + Math.abs(gY), gDC + Math.abs(gZ),
        bDC + Math.abs(bX), bDC + Math.abs(bY), bDC + Math.abs(bZ),
      )
    }

    const invMax = maxVal > 0.001 ? 1.0 / maxVal : 1.0

    // Second pass: evaluate SH per vertex and write colors
    let activeIdx = 0
    for (let i = 0; i < probeCount; i++) {
      if (!_isProbeActive(grid.data, i)) continue

      const off = i * 12
      const rDC = grid.data[off]!,     rX = grid.data[off + 1]!, rY = grid.data[off + 2]!, rZ = grid.data[off + 3]!
      const gDC = grid.data[off + 4]!, gX = grid.data[off + 5]!, gY = grid.data[off + 6]!, gZ = grid.data[off + 7]!
      const bDC = grid.data[off + 8]!, bX = grid.data[off + 9]!, bY = grid.data[off + 10]!, bZ = grid.data[off + 11]!

      const vBase = activeIdx * VERTS_PER_PROBE
      for (let v = 0; v < VERTS_PER_PROBE; v++) {
        const p = (vBase + v) * 3
        const dx = dirs[p]!
        const dy = dirs[p + 1]!
        const dz = dirs[p + 2]!

        // Evaluate L1 SH: c0 + c1*dir.x + c2*dir.y + c3*dir.z
        normals[p] = Math.max(0, (rDC + rX * dx + rY * dy + rZ * dz) * invMax)
        normals[p + 1] = Math.max(0, (gDC + gX * dx + gY * dy + gZ * dz) * invMax)
        normals[p + 2] = Math.max(0, (bDC + bX * dx + bY * dy + bZ * dz) * invMax)
      }

      activeIdx++
    }
  }
}
