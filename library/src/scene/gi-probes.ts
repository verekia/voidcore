// GI Probes – Global Illumination light probes for indirect lighting.
//
// GIProbeGrid stores a 3D grid of light probes, each encoding incoming incident radiance
// using L1 Spherical Harmonics (4 coefficients × 3 color channels = 12 floats per probe).
// At render time, the fragment shader looks up the nearest probes via hardware trilinear
// interpolation on a 3D texture, evaluates the SH for the surface normal, and adds the
// result to the ambient lighting term. This gives meshes a colored ambient tint that
// varies spatially — similar to Unity's Light Probe system.
//
// SH L1 captures directional variation: which direction is brighter at each probe location.
// The four basis functions are constant (DC), X, Y, and Z:
//   irradiance(n) = c0 + c1*n.x + c2*n.y + c3*n.z
// where c0..c3 are per-channel coefficients stored as incident radiance.
//
// The `dcWeight` property (0–1) controls how the SH is evaluated at shade time:
//   dcWeight=0 (default): directional-only evaluation — drops the DC term to prevent
//     surfaces from being tinted by their own color (no self-reinforcement).
//   dcWeight=1: full incident radiance with cosine convolution at shade time (L1 cosine
//     lobe ratio = 2/3 for directional terms). More physically based but may reinforce
//     surface colors in monochromatic environments.
//
// Data is uploaded to the GPU as a single 3D RGBA16F texture (gridX × gridY × gridZ*3).
// The three color channels (R, G, B) are tiled along the Z axis:
//   z=[0..gridZ)       → Red   SH coefficients [c0, c1, c2, c3]
//   z=[gridZ..2*gridZ) → Green SH coefficients [c0, c1, c2, c3]
//   z=[2*gridZ..3*gridZ) → Blue SH coefficients [c0, c1, c2, c3]
//
// new GIProbeGrid(options) – Creates a probe grid with given bounds and resolution.
// grid.setProbe(ix, iy, iz, opts) – Sets a probe's irradiance (constant or directional).
// grid.dcWeight – Controls DC vs directional-only SH evaluation (0–1).
// grid.needsUpdate – Flag that tells the renderer to re-upload the texture.

// ─── Float32-to-Float16 conversion (matches pack.ts) ─────────────────

const _f32 = new Float32Array(1)
const _u32 = new Uint32Array(_f32.buffer)

const floatToHalf = (val: number): number => {
  _f32[0] = val
  const b = _u32[0]!
  const sign = (b >>> 16) & 0x8000
  const exp = ((b >>> 23) & 0xff) - 112
  const man = (b & 0x7fffff) >>> 13
  if (exp <= 0) return sign
  if (exp >= 31) return sign | 0x7c00
  return sign | (exp << 10) | man
}

// ─── Types ──────────────────────────────────────────────────────────

export interface GIProbeGridOptions {
  /** World-space minimum corner of the probe grid. */
  boundsMin: [number, number, number]
  /** World-space maximum corner of the probe grid. */
  boundsMax: [number, number, number]
  /** Number of probes along each axis [X, Y, Z]. */
  resolution: [number, number, number]
  /** Intensity multiplier for the GI contribution (default 1.0). */
  intensity?: number
  /** Weight of the DC (constant) term in SH evaluation (0–1, default 0).
   *  0 = directional-only (no self-reinforcement), 1 = full incident radiance
   *  with cosine convolution at shade time (physically based but may reinforce colors). */
  dcWeight?: number
}

export interface ProbeData {
  /** Constant ambient color at this probe. */
  color: [number, number, number]
  /** Optional dominant direction for directional irradiance. */
  direction?: [number, number, number]
  /** Optional color for the directional component (defaults to `color` if omitted). */
  directionalColor?: [number, number, number]
}

// ─── GI Probe Grid ──────────────────────────────────────────────────

export class GIProbeGrid {
  /** World-space minimum corner. */
  readonly boundsMin: [number, number, number]
  /** World-space maximum corner. */
  readonly boundsMax: [number, number, number]
  /** Number of probes along each axis [X, Y, Z]. */
  readonly resolution: [number, number, number]
  /** Intensity multiplier for the GI contribution. */
  intensity: number
  /** Weight of the DC (constant) term in SH evaluation (0 = directional only, 1 = full). */
  dcWeight: number

  /**
   * Raw SH data: 12 floats per probe (4 coefficients × 3 channels).
   * Layout per probe (12 floats):
   *   [R0, R1, R2, R3, G0, G1, G2, G3, B0, B1, B2, B3]
   * where channel irradiance(n) = c0 + c1*n.x + c2*n.y + c3*n.z
   */
  readonly data: Float32Array

  /** When true, the renderer re-uploads the texture data. */
  needsUpdate = true

  /** Total number of probes in the grid. */
  readonly probeCount: number

  constructor(opts: GIProbeGridOptions) {
    this.boundsMin = [...opts.boundsMin]
    this.boundsMax = [...opts.boundsMax]
    this.resolution = [...opts.resolution]
    this.intensity = opts.intensity ?? 1.0
    this.dcWeight = opts.dcWeight ?? 0
    this.probeCount = opts.resolution[0] * opts.resolution[1] * opts.resolution[2]
    this.data = new Float32Array(this.probeCount * 12)
  }

  /** Get the flat index for a probe at grid position (ix, iy, iz). */
  private _index(ix: number, iy: number, iz: number): number {
    return (iz * this.resolution[1] * this.resolution[0] + iy * this.resolution[0] + ix) * 12
  }

  /**
   * Set a probe's irradiance using a simple color and optional direction.
   *
   * With just `color`: sets uniform ambient (same from all directions).
   * With `color` + `direction` + optional `directionalColor`: adds directional variation.
   */
  setProbe(ix: number, iy: number, iz: number, probe: ProbeData): void {
    const off = this._index(ix, iy, iz)
    const cr = probe.color[0]
    const cg = probe.color[1]
    const cb = probe.color[2]

    // DC term (constant irradiance)
    this.data[off] = cr
    this.data[off + 4] = cg
    this.data[off + 8] = cb

    // Directional terms
    if (probe.direction) {
      const dx = probe.direction[0]
      const dy = probe.direction[1]
      const dz = probe.direction[2]
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz)
      if (len > 0.001) {
        const nx = dx / len
        const ny = dy / len
        const nz = dz / len
        const dr = probe.directionalColor ? probe.directionalColor[0] : cr
        const dg = probe.directionalColor ? probe.directionalColor[1] : cg
        const db = probe.directionalColor ? probe.directionalColor[2] : cb
        // X, Y, Z directional coefficients
        this.data[off + 1] = dr * nx
        this.data[off + 2] = dr * ny
        this.data[off + 3] = dr * nz
        this.data[off + 5] = dg * nx
        this.data[off + 6] = dg * ny
        this.data[off + 7] = dg * nz
        this.data[off + 9] = db * nx
        this.data[off + 10] = db * ny
        this.data[off + 11] = db * nz
      }
    } else {
      this.data[off + 1] = 0
      this.data[off + 2] = 0
      this.data[off + 3] = 0
      this.data[off + 5] = 0
      this.data[off + 6] = 0
      this.data[off + 7] = 0
      this.data[off + 9] = 0
      this.data[off + 10] = 0
      this.data[off + 11] = 0
    }

    this.needsUpdate = true
  }

  /**
   * Set raw SH L1 coefficients for a probe.
   * Each vec4 is [c0, c1, c2, c3] where irradiance(n) = c0 + c1*n.x + c2*n.y + c3*n.z
   */
  setProbeRaw(
    ix: number,
    iy: number,
    iz: number,
    shR: [number, number, number, number],
    shG: [number, number, number, number],
    shB: [number, number, number, number],
  ): void {
    const off = this._index(ix, iy, iz)
    this.data[off] = shR[0]
    this.data[off + 1] = shR[1]
    this.data[off + 2] = shR[2]
    this.data[off + 3] = shR[3]
    this.data[off + 4] = shG[0]
    this.data[off + 5] = shG[1]
    this.data[off + 6] = shG[2]
    this.data[off + 7] = shG[3]
    this.data[off + 8] = shB[0]
    this.data[off + 9] = shB[1]
    this.data[off + 10] = shB[2]
    this.data[off + 11] = shB[3]
    this.needsUpdate = true
  }

  /**
   * Pack probe data into a Uint16Array (RGBA16F) for 3D texture upload.
   *
   * The 3D texture layout is: width=gridX, height=gridY, depth=gridZ*3
   *   z=[0..gridZ)       → Red   channel [c0, c1, c2, c3]
   *   z=[gridZ..2*gridZ) → Green channel [c0, c1, c2, c3]
   *   z=[2*gridZ..3*gridZ) → Blue channel [c0, c1, c2, c3]
   */
  getTextureData(): Uint16Array {
    const [rx, ry, rz] = this.resolution
    const totalDepth = rz * 3
    const texels = rx * ry * totalDepth
    const packed = new Uint16Array(texels * 4)

    for (let iz = 0; iz < rz; iz++) {
      for (let iy = 0; iy < ry; iy++) {
        for (let ix = 0; ix < rx; ix++) {
          const probeOff = this._index(ix, iy, iz)

          // Red channel → z slice iz
          const rTexelIdx = (iz * ry * rx + iy * rx + ix) * 4
          packed[rTexelIdx] = floatToHalf(this.data[probeOff]!)
          packed[rTexelIdx + 1] = floatToHalf(this.data[probeOff + 1]!)
          packed[rTexelIdx + 2] = floatToHalf(this.data[probeOff + 2]!)
          packed[rTexelIdx + 3] = floatToHalf(this.data[probeOff + 3]!)

          // Green channel → z slice iz + gridZ
          const gTexelIdx = ((iz + rz) * ry * rx + iy * rx + ix) * 4
          packed[gTexelIdx] = floatToHalf(this.data[probeOff + 4]!)
          packed[gTexelIdx + 1] = floatToHalf(this.data[probeOff + 5]!)
          packed[gTexelIdx + 2] = floatToHalf(this.data[probeOff + 6]!)
          packed[gTexelIdx + 3] = floatToHalf(this.data[probeOff + 7]!)

          // Blue channel → z slice iz + 2*gridZ
          const bTexelIdx = ((iz + rz * 2) * ry * rx + iy * rx + ix) * 4
          packed[bTexelIdx] = floatToHalf(this.data[probeOff + 8]!)
          packed[bTexelIdx + 1] = floatToHalf(this.data[probeOff + 9]!)
          packed[bTexelIdx + 2] = floatToHalf(this.data[probeOff + 10]!)
          packed[bTexelIdx + 3] = floatToHalf(this.data[probeOff + 11]!)
        }
      }
    }

    return packed
  }
}
