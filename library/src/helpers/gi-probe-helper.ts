// GI Probe Helper – Debug visualization for GI probe grids.
//
// Renders each probe in a GIProbeGrid as a small colored sphere. The sphere color represents
// the probe's constant (DC) irradiance — the omnidirectional ambient color at that location.
// This lets you see the spatial distribution of indirect lighting across the scene.
//
// Colors are auto-normalized so the brightest probe maps to full white, making the relative
// spatial variation clearly visible regardless of absolute GI intensity.
//
// Probes with zero DC values (inactive/culled by maxDistance) are skipped entirely — no geometry
// is generated for them.
//
// Implementation: Generates a merged geometry with one low-poly sphere per active probe. Probe
// colors are stored in the normal attribute (abusing normals as a per-vertex color channel), and
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

  constructor(grid: GIProbeGrid, options?: GIProbeHelperOptions) {
    this._radius = options?.radius ?? 0.4

    // Build merged sphere geometry (only for active probes)
    const { positions, normals, uvs, indices } = this._buildMergedSpheres(grid)
    this._geometry = new Geometry({
      positions,
      normals,
      indices,
      uvs,
    })

    // Write normalized probe colors into normals
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
   */
  private _buildMergedSpheres(grid: GIProbeGrid): {
    positions: Float32Array
    normals: Float32Array
    uvs: Float32Array
    indices: Uint16Array | Uint32Array
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

          // Normals will be overwritten by _writeColors; just zero for now
          normals[p] = 0
          normals[p + 1] = 0
          normals[p + 2] = 1 // Default to avoid zero-length normal issues

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

    return { positions, normals, uvs, indices }
  }

  /**
   * Write normalized probe DC colors into the normals array.
   * The brightest channel across all probes maps to 1.0.
   * Only writes to active (non-zero DC) probes matching _buildMergedSpheres.
   */
  private _writeColors(grid: GIProbeGrid): void {
    const [rx, ry, rz] = grid.resolution
    const normals = this._geometry.normals
    const probeCount = rx * ry * rz

    // Find max DC value across all active probes for normalization
    let maxDC = 0
    for (let i = 0; i < probeCount; i++) {
      const off = i * 12
      maxDC = Math.max(maxDC, Math.abs(grid.data[off]!), Math.abs(grid.data[off + 4]!), Math.abs(grid.data[off + 8]!))
    }

    const invMax = maxDC > 0.001 ? 1.0 / maxDC : 1.0

    // Write colors only for active probes (same order as _buildMergedSpheres)
    let activeIdx = 0
    for (let i = 0; i < probeCount; i++) {
      if (!_isProbeActive(grid.data, i)) continue

      const dataOff = i * 12
      const cr = grid.data[dataOff]! * invMax
      const cg = grid.data[dataOff + 4]! * invMax
      const cb = grid.data[dataOff + 8]! * invMax

      const vBase = activeIdx * VERTS_PER_PROBE
      for (let v = 0; v < VERTS_PER_PROBE; v++) {
        const p = (vBase + v) * 3
        normals[p] = cr
        normals[p + 1] = cg
        normals[p + 2] = cb
      }

      activeIdx++
    }
  }
}
