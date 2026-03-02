// Grass – Stylized billboard triangle grass blades scattered on terrain surfaces.
//
// The grass system takes a grid geometry whose vertices are precomputed patch centers.
// Each vertex becomes one patch, spawning `bladesPerPatch` blades in a disk of `patchRadius`
// around its center. This creates natural-looking clumps while keeping vertex counts
// manageable. All blades are merged into a single
// Geometry for a single draw call. All 3 vertices per blade store the base center position;
// per-vertex metadata (height factor, blade seed, horizontal sign) is encoded in the
// normal attribute (snorm8), and blade dimensions (half-width, height) in UVs (float16).
// A custom vertex shader on a BasicMaterial computes billboard orientation from the camera
// direction at runtime, plus wind animation and distance culling (degenerate triangles).
// The fragment shader handles per-blade color variation and base AO darkening — no
// renderer changes needed, grass is a regular Mesh.
//
// generateGrass(geometry, palette, options) – Scatters patch-based grass blades on grid faces.
// createGrassMaterial(palette, options)     – Returns a BasicMaterial with grass custom shaders.

import { BasicMaterial } from '../materials/material'
import { Geometry } from './geometry'

import type { PaletteEntry } from '../materials/material'

// ─── Types ───────────────────────────────────────────────────────────────────

export interface GrassOptions {
  // /** Patch centers per square unit. Default 3. */
  // patchDensity?: number
  /** Blades per patch. Default 12. */
  bladesPerPatch?: number
  /** Disk radius around each patch center. Default 0.5. */
  patchRadius?: number
  /** Minimum blade height. Default 0.15. */
  minHeight?: number
  /** Maximum blade height. Default 0.4. */
  maxHeight?: number
  /** Minimum blade width. Default 0.06. */
  minWidth?: number
  /** Maximum blade width. Default 0.15. */
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
 * Generates grass blades from a grid geometry whose vertices are precomputed patch centers.
 * Each vertex becomes one patch, spawning `bladesPerPatch` blades in a disk of `patchRadius`.
 * Returns a Geometry with all blades merged for a single draw call.
 */
export const generateGrass = (geometry: Geometry, palette: PaletteEntry[], options?: GrassOptions): Geometry => {
  const bladesPerPatch = options?.bladesPerPatch ?? 12
  const patchRadius = options?.patchRadius ?? 0.5
  const minHeight = options?.minHeight ?? 0.15
  const maxHeight = options?.maxHeight ?? 0.4
  const minWidth = options?.minWidth ?? 0.06
  const maxWidth = options?.maxWidth ?? 0.15
  const seed = options?.seed ?? 42

  const rng = mulberry32(seed)
  const pos = geometry.positions
  const totalPatches = geometry.vertexCount

  // TODO: Density-based scatter on triangle faces (for non-grid geometries)
  // const patchDensity = options?.patchDensity ?? 3
  // const matIdx = geometry.materialIndices
  // let grassFlags: Uint8Array | null = null
  // if (matIdx) {
  //   grassFlags = new Uint8Array(palette.length)
  //   for (let i = 0; i < palette.length; i++) {
  //     grassFlags[i] = palette[i]!.grass ? 1 : 0
  //   }
  // }
  // const idx = geometry.indices
  // const triCount = idx.length / 3
  // let totalPatches = 0
  // const patchCountPerTri = new Uint32Array(triCount)
  // let fractionalAccum = 0
  // for (let t = 0; t < triCount; t++) {
  //   const i0 = idx[t * 3]!, i1 = idx[t * 3 + 1]!, i2 = idx[t * 3 + 2]!
  //   if (grassFlags && matIdx) {
  //     if (!grassFlags[matIdx[i0]!] || !grassFlags[matIdx[i1]!] || !grassFlags[matIdx[i2]!]) continue
  //   }
  //   const ax = pos[i0 * 3]!, ay = pos[i0 * 3 + 1]!, az = pos[i0 * 3 + 2]!
  //   const e1x = pos[i1 * 3]! - ax, e1y = pos[i1 * 3 + 1]! - ay, e1z = pos[i1 * 3 + 2]! - az
  //   const e2x = pos[i2 * 3]! - ax, e2y = pos[i2 * 3 + 1]! - ay, e2z = pos[i2 * 3 + 2]! - az
  //   const cx = e1y * e2z - e1z * e2y, cy = e1z * e2x - e1x * e2z, cz = e1x * e2y - e1y * e2x
  //   const area = 0.5 * Math.sqrt(cx * cx + cy * cy + cz * cz)
  //   fractionalAccum += area * patchDensity
  //   const count = Math.floor(fractionalAccum)
  //   fractionalAccum -= count
  //   patchCountPerTri[t] = count
  //   totalPatches += count
  // }

  const totalBlades = totalPatches * bladesPerPatch

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
  const TWO_PI = Math.PI * 2

  // Each vertex in the grid is a patch center
  for (let v = 0; v < totalPatches; v++) {
    const px = pos[v * 3]!
    const py = pos[v * 3 + 1]!
    const pz = pos[v * 3 + 2]!

    // TODO: Barycentric patch center placement on triangle faces
    // let u = rng(), v = rng()
    // if (u + v > 1) { u = 1 - u; v = 1 - v }
    // const w = 1 - u - v
    // const px = w * ax + u * bx + v * cx
    // const py = w * ay + u * by + v * cy
    // const pz = w * az + u * bz + v * cz

    // TODO: Interpolated surface normal at patch center (for tangent-frame disk scatter)
    // let nx = w * n0x + u * n1x + v * n2x
    // let ny = w * n0y + u * n1y + v * n2y
    // let nz = w * n0z + u * n1z + v * n2z
    // const nLen = Math.sqrt(nx * nx + ny * ny + nz * nz)
    // if (nLen > 0.0001) { nx /= nLen; ny /= nLen; nz /= nLen }
    //
    // // Build tangent frame from surface normal
    // // Cross normal with Z-up to get tangent; if parallel, use X-right
    // let tanx: number, tany: number, tanz: number
    // if (Math.abs(nz) < 0.999) {
    //   tanx = ny; tany = -nx; tanz = 0 // cross(normal, Z-up)
    // } else {
    //   tanx = 0; tany = nz; tanz = -ny // cross(normal, X-right)
    // }
    // const tLen = Math.sqrt(tanx * tanx + tany * tany + tanz * tanz)
    // tanx /= tLen; tany /= tLen; tanz /= tLen
    //
    // // Bitangent = cross(normal, tangent)
    // const btx = ny * tanz - nz * tany
    // const bty = nz * tanx - nx * tanz
    // const btz = nx * tany - ny * tanx

    // Scatter blades in a disk around the patch center (XY plane)
    for (let b = 0; b < bladesPerPatch; b++) {
      const angle = rng() * TWO_PI
      const r = Math.sqrt(rng()) * patchRadius // sqrt for uniform disk distribution
      const offsetX = Math.cos(angle) * r
      const offsetY = Math.sin(angle) * r

      const bladeX = px + offsetX
      const bladeY = py + offsetY
      const bladeZ = pz

      // TODO: Surface-aligned disk scatter using tangent frame
      // const cosA = Math.cos(angle)
      // const sinA = Math.sin(angle)
      // const bladeX = px + cosA * r * tanx + sinA * r * btx
      // const bladeY = py + cosA * r * tany + sinA * r * bty
      // const bladeZ = pz + cosA * r * tanz + sinA * r * btz

      // Random blade parameters
      const height = minHeight + rng() * (maxHeight - minHeight)
      const width = minWidth + rng() * (maxWidth - minWidth)
      const bladeSeed = rng()
      const hw = width * 0.5

      // All 3 vertices store the blade base center — the vertex shader
      // computes billboard offsets from the camera direction at runtime.
      const vo = bladeI * 9
      const no = bladeI * 9
      const io = bladeI * 3

      // v0: base left  — v1: base right  — v2: tip
      outPos[vo] = bladeX
      outPos[vo + 1] = bladeY
      outPos[vo + 2] = bladeZ
      outPos[vo + 3] = bladeX
      outPos[vo + 4] = bladeY
      outPos[vo + 5] = bladeZ
      outPos[vo + 6] = bladeX
      outPos[vo + 7] = bladeY
      outPos[vo + 8] = bladeZ

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
    let lean = topDown * 0.95;
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
    float lean = topDown * 0.95;
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
