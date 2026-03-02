// Grass – Stylized billboard triangle grass blades scattered on terrain surfaces.
//
// The grass system generates isosceles triangle blades on terrain faces marked with
// `grass: true` in the palette system. All blades are merged into a single Geometry
// for a single draw call. All 3 vertices per blade store the base center position;
// per-vertex metadata (height factor, blade seed, horizontal sign) is encoded in the
// normal attribute (snorm8), and blade dimensions (half-width, height) in UVs (float16).
// A custom vertex shader on a BasicMaterial computes billboard orientation from the
// camera direction at runtime, plus wind animation and distance culling (degenerate
// triangles). The fragment shader handles per-blade color variation and base AO
// darkening — no renderer changes needed, grass is a regular Mesh.
//
// generateGrass(geometry, palette, options) – Scatters triangle blades on grass-marked faces.
// createGrassMaterial(palette, options)     – Returns a BasicMaterial with grass custom shaders.

import { BasicMaterial } from '../materials/material'
import { Geometry } from './geometry'

import type { PaletteEntry } from '../materials/material'

// ─── Types ───────────────────────────────────────────────────────────────────

export interface GrassOptions {
  /** Blades per square unit. Default 10. */
  density?: number
  /** Minimum blade height. Default 0.15. */
  minHeight?: number
  /** Maximum blade height. Default 0.4. */
  maxHeight?: number
  /** Minimum blade width. Default 0.03. */
  minWidth?: number
  /** Maximum blade width. Default 0.08. */
  maxWidth?: number
  /** RNG seed for deterministic placement. Default 42. */
  seed?: number
  /** Distance culling radius (XY plane). Default 30. */
  radius?: number
  /** Wind animation strength. Default 0.1. */
  windStrength?: number
  /** Base blade color [r, g, b]. Default [0.933, 0.733, 0.467] (#eb7). */
  color?: [number, number, number]
  /** Variation blade color [r, g, b]. Default [0.867, 0.667, 0.4] (#da6). */
  color2?: [number, number, number]
}

// ─── Seeded PRNG (Mulberry32) ────────────────────────────────────────────────

const mulberry32 = (seed: number) => {
  let s = seed | 0
  return () => {
    s = (s + 0x6d2b79f5) | 0
    let t = Math.imul(s ^ (s >>> 15), 1 | s)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// ─── Geometry Generation ─────────────────────────────────────────────────────

/**
 * Scatters triangle grass blades on terrain faces marked with `grass: true`.
 * Returns a Geometry with all blades merged for a single draw call.
 */
export const generateGrass = (geometry: Geometry, palette: PaletteEntry[], options?: GrassOptions): Geometry => {
  const density = options?.density ?? 10
  const minHeight = options?.minHeight ?? 0.15
  const maxHeight = options?.maxHeight ?? 0.4
  const minWidth = options?.minWidth ?? 0.06
  const maxWidth = options?.maxWidth ?? 0.15
  const seed = options?.seed ?? 42

  const rng = mulberry32(seed)
  const matIdx = geometry.materialIndices

  if (!matIdx) {
    return new Geometry({ positions: new Float32Array(0), normals: new Float32Array(0), indices: new Uint16Array(0) })
  }

  // Build grass flag lookup
  const grassFlags = new Uint8Array(palette.length)
  for (let i = 0; i < palette.length; i++) {
    grassFlags[i] = palette[i]!.grass ? 1 : 0
  }

  const pos = geometry.positions
  const idx = geometry.indices
  const triCount = idx.length / 3

  // First pass: count blades per triangle
  let totalBlades = 0
  const bladeCountPerTri = new Uint32Array(triCount)

  for (let t = 0; t < triCount; t++) {
    const i0 = idx[t * 3]!
    const i1 = idx[t * 3 + 1]!
    const i2 = idx[t * 3 + 2]!

    if (!grassFlags[matIdx[i0]!] || !grassFlags[matIdx[i1]!] || !grassFlags[matIdx[i2]!]) continue

    // Triangle area via cross product
    const ax = pos[i0 * 3]!,
      ay = pos[i0 * 3 + 1]!,
      az = pos[i0 * 3 + 2]!
    const e1x = pos[i1 * 3]! - ax,
      e1y = pos[i1 * 3 + 1]! - ay,
      e1z = pos[i1 * 3 + 2]! - az
    const e2x = pos[i2 * 3]! - ax,
      e2y = pos[i2 * 3 + 1]! - ay,
      e2z = pos[i2 * 3 + 2]! - az
    const cx = e1y * e2z - e1z * e2y
    const cy = e1z * e2x - e1x * e2z
    const cz = e1x * e2y - e1y * e2x
    const area = 0.5 * Math.sqrt(cx * cx + cy * cy + cz * cz)

    const count = Math.floor(area * density)
    bladeCountPerTri[t] = count
    totalBlades += count
  }

  if (totalBlades === 0) {
    return new Geometry({ positions: new Float32Array(0), normals: new Float32Array(0), indices: new Uint16Array(0) })
  }

  // Allocate output buffers
  const totalVerts = totalBlades * 3
  const outPos = new Float32Array(totalVerts * 3)
  const outNrm = new Float32Array(totalVerts * 3)
  const outIdx = totalVerts > 65535 ? new Uint32Array(totalBlades * 3) : new Uint16Array(totalBlades * 3)
  const outUvs = new Float32Array(totalVerts * 2)

  let bladeI = 0

  for (let t = 0; t < triCount; t++) {
    const count = bladeCountPerTri[t]!
    if (count === 0) continue

    const i0 = idx[t * 3]!
    const i1 = idx[t * 3 + 1]!
    const i2 = idx[t * 3 + 2]!

    // Triangle vertex positions
    const ax = pos[i0 * 3]!,
      ay = pos[i0 * 3 + 1]!,
      az = pos[i0 * 3 + 2]!
    const bx = pos[i1 * 3]!,
      by = pos[i1 * 3 + 1]!,
      bz = pos[i1 * 3 + 2]!
    const cx = pos[i2 * 3]!,
      cy = pos[i2 * 3 + 1]!,
      cz = pos[i2 * 3 + 2]!

    for (let b = 0; b < count; b++) {
      // Random barycentric coordinates
      let u = rng()
      let v = rng()
      if (u + v > 1) {
        u = 1 - u
        v = 1 - v
      }
      const w = 1 - u - v

      // Interpolate surface position (blade base center)
      const px = w * ax + u * bx + v * cx
      const py = w * ay + u * by + v * cy
      const pz = w * az + u * bz + v * cz

      // Random blade parameters
      const height = minHeight + rng() * (maxHeight - minHeight)
      const width = minWidth + rng() * (maxWidth - minWidth)
      rng() // skip (was angle, no longer needed for billboard)
      const bladeSeed = rng()
      const hw = width * 0.5

      // All 3 vertices store the blade base center — the vertex shader
      // computes billboard offsets from the camera direction at runtime.
      const vo = bladeI * 9
      const no = bladeI * 9
      const io = bladeI * 3

      // v0: base left  — v1: base right  — v2: tip
      outPos[vo] = px
      outPos[vo + 1] = py
      outPos[vo + 2] = pz
      outPos[vo + 3] = px
      outPos[vo + 4] = py
      outPos[vo + 5] = pz
      outPos[vo + 6] = px
      outPos[vo + 7] = py
      outPos[vo + 8] = pz

      // Encode per-vertex data in normals: [heightFactor, bladeSeed, horizontalSign]
      // horizontalSign: -1 = left, +1 = right, 0 = tip
      outNrm[no] = 0
      outNrm[no + 1] = bladeSeed
      outNrm[no + 2] = -1
      outNrm[no + 3] = 0
      outNrm[no + 4] = bladeSeed
      outNrm[no + 5] = 1
      outNrm[no + 6] = 1 // tip heightFactor
      outNrm[no + 7] = bladeSeed
      outNrm[no + 8] = 0

      // Encode blade dimensions in UVs: [halfWidth, height]
      const uo = bladeI * 6
      outUvs[uo] = hw
      outUvs[uo + 1] = height
      outUvs[uo + 2] = hw
      outUvs[uo + 3] = height
      outUvs[uo + 4] = hw
      outUvs[uo + 5] = height

      // Indices
      const vi = bladeI * 3
      outIdx[io] = vi
      outIdx[io + 1] = vi + 1
      outIdx[io + 2] = vi + 2

      bladeI++
    }
  }

  return new Geometry({ positions: outPos, normals: outNrm, indices: outIdx, uvs: outUvs })
}

// ─── Shader Snippets ─────────────────────────────────────────────────────────

const GRASS_VERTEX_WGSL = /* wgsl */ `
  out.normal = a_normal.xyz;
  let dx = out.worldPos.x - frame.cameraPos.x;
  let dy = out.worldPos.y - frame.cameraPos.y;
  let distSq = dx * dx + dy * dy;
  let radiusSq = uniforms.radius * uniforms.radius;
  let densityThreshold = 1.0 - distSq / radiusSq;
  if (distSq > radiusSq || a_normal.y > densityThreshold) {
    out.worldPos = vec3<f32>(0.0, 0.0, -99999.0);
  } else {
    let hf = a_normal.x;
    let hSign = a_normal.z;
    let hw = a_uv.x;
    let h = a_uv.y;
    let dz = out.worldPos.z - frame.cameraPos.z;
    let dist3D = sqrt(distSq + dz * dz);
    let topDown = clamp(abs(dz) / max(dist3D, 0.001), 0.0, 1.0);
    let lean = topDown * 0.7;
    let camLen = sqrt(distSq);
    var tx = 0.0;
    var ty = 1.0;
    var leanX = 0.0;
    var leanY = 0.0;
    if (camLen > 0.001) {
      tx = -dy / camLen;
      ty = dx / camLen;
      leanX = dx / camLen;
      leanY = dy / camLen;
    }
    out.worldPos = vec3<f32>(
      out.worldPos.x + tx * hw * hSign + leanX * lean * h * hf,
      out.worldPos.y + ty * hw * hSign + leanY * lean * h * hf,
      out.worldPos.z + h * hf * (1.0 - lean * 0.5)
    );
    let wp = out.worldPos.x * 0.5 + out.worldPos.y * 0.3 + frame.elapsed * 2.0;
    let wx = sin(wp) * uniforms.windStrength * hf;
    let wy = cos(wp * 0.7) * uniforms.windStrength * hf * 0.5;
    out.worldPos = vec3<f32>(out.worldPos.x + wx, out.worldPos.y + wy, out.worldPos.z);
  }
`

const GRASS_FRAGMENT_WGSL = /* wgsl */ `
  let heightFactor = in.normal.x;
  let bladeSeed = in.normal.y;
  let baseGreen = vec3<f32>(uniforms.colorR, uniforms.colorG, uniforms.colorB);
  let varYellow = vec3<f32>(uniforms.varR, uniforms.varG, uniforms.varB);
  finalColor = mix(baseGreen, varYellow, bladeSeed);
  let ao = mix(0.7, 1.0, heightFactor);
  finalColor = finalColor * ao;
`

const GRASS_VERTEX_GLSL = /* glsl */ `
  v_normal = a_normal.xyz;
  float dx = v_worldPos.x - u_cameraPos.x;
  float dy = v_worldPos.y - u_cameraPos.y;
  float distSq = dx * dx + dy * dy;
  float radiusSq = uniforms.radius * uniforms.radius;
  float densityThreshold = 1.0 - distSq / radiusSq;
  if (distSq > radiusSq || a_normal.y > densityThreshold) {
    v_worldPos = vec3(0.0, 0.0, -99999.0);
  } else {
    float hf = a_normal.x;
    float hSign = a_normal.z;
    float hw = a_uv.x;
    float h = a_uv.y;
    float dz = v_worldPos.z - u_cameraPos.z;
    float dist3D = sqrt(distSq + dz * dz);
    float topDown = clamp(abs(dz) / max(dist3D, 0.001), 0.0, 1.0);
    float lean = topDown * 0.7;
    float camLen = sqrt(distSq);
    float tx = 0.0;
    float ty = 1.0;
    float leanX = 0.0;
    float leanY = 0.0;
    if (camLen > 0.001) {
      tx = -dy / camLen;
      ty = dx / camLen;
      leanX = dx / camLen;
      leanY = dy / camLen;
    }
    v_worldPos = vec3(
      v_worldPos.x + tx * hw * hSign + leanX * lean * h * hf,
      v_worldPos.y + ty * hw * hSign + leanY * lean * h * hf,
      v_worldPos.z + h * hf * (1.0 - lean * 0.5)
    );
    float wp = v_worldPos.x * 0.5 + v_worldPos.y * 0.3 + u_elapsed * 2.0;
    float wx = sin(wp) * uniforms.windStrength * hf;
    float wy = cos(wp * 0.7) * uniforms.windStrength * hf * 0.5;
    v_worldPos = vec3(v_worldPos.x + wx, v_worldPos.y + wy, v_worldPos.z);
  }
`

const GRASS_FRAGMENT_GLSL = /* glsl */ `
  float heightFactor = v_normal.x;
  float bladeSeed = v_normal.y;
  vec3 baseGreen = vec3(uniforms.colorR, uniforms.colorG, uniforms.colorB);
  vec3 varYellow = vec3(uniforms.varR, uniforms.varG, uniforms.varB);
  finalColor = mix(baseGreen, varYellow, bladeSeed);
  float ao = mix(0.7, 1.0, heightFactor);
  finalColor = finalColor * ao;
`

// ─── Material Creation ───────────────────────────────────────────────────────

/**
 * Creates a BasicMaterial with custom shaders for grass rendering.
 * Uses `options.color`/`color2` if provided, otherwise reads from the first
 * `grass: true` palette entry, otherwise uses built-in defaults.
 */
export const createGrassMaterial = (palette: PaletteEntry[], options?: GrassOptions): BasicMaterial => {
  let baseColor: [number, number, number] = [0.933, 0.733, 0.467]
  let varColor: [number, number, number] = [0.867, 0.667, 0.4]

  if (options?.color) {
    baseColor = options.color
    varColor = options.color2 ?? [Math.min(1, baseColor[0] + 0.3), Math.min(1, baseColor[1] + 0.15), baseColor[2]]
  } else {
    for (const entry of palette) {
      if (entry.grass) {
        baseColor = entry.color
        varColor = entry.color2 ?? [Math.min(1, baseColor[0] + 0.3), Math.min(1, baseColor[1] + 0.15), baseColor[2]]
        break
      }
    }
  }

  const radius = options?.radius ?? 30
  const windStrength = options?.windStrength ?? 0.1

  return new BasicMaterial({
    side: 'double',
    customShader: {
      uniforms: {
        radius,
        windStrength,
        colorR: baseColor[0],
        colorG: baseColor[1],
        colorB: baseColor[2],
        varR: varColor[0],
        varG: varColor[1],
        varB: varColor[2],
      },
      vertexWGSL: GRASS_VERTEX_WGSL,
      fragmentWGSL: GRASS_FRAGMENT_WGSL,
      vertexGLSL: GRASS_VERTEX_GLSL,
      fragmentGLSL: GRASS_FRAGMENT_GLSL,
    },
  })
}
