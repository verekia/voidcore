// Material – Defines how a mesh surface looks when rendered.
//
// A material controls the color and shading model of a mesh. Two shading types are supported:
//   - "basic"   – Unlit flat color (ignores lights, good for UI or debug).
//   - "lambert" – Diffuse shading that reacts to directional and ambient light.
//
// Materials can define a palette: an array of color entries used with bakePalette() to bake
// per-vertex colors into the geometry. The palette is data only — the renderer reads vertex
// colors directly from geometry attributes, not from material uniforms. Palette entries can
// include per-material tiled AO textures (tiledAo, tiledAoIntensity, tiledAoScale) that
// repeat across surfaces using world-space XY coordinates, combining multiplicatively with
// the world AO map.
//
// Texture maps add per-pixel detail from images:
//   - colorMap – Multiplies the base color by a texture sample (diffuse/albedo map).
//   - aoMap    – Multiplies ambient light by the red channel of a texture (ambient occlusion).
//   - aoIntensity – Controls how strongly the AO map affects lighting (0 = no effect, 1 = full).
// When a material has texture maps, the renderer uses a textured shader variant that samples
// the textures using the mesh's UV coordinates.
//
// Emissive brightness controls how much bright emissive colors wash toward white (the "neon
// glow" effect). At 0, the raw emissive color is used. At 1 (default), high-intensity emissive
// areas desaturate toward white, simulating the overexposed look of real glowing surfaces.
//
// Transparency is supported via sorted alpha blending. Set `transparent: true` and an
// `opacity` value (0–1) to make a material see-through. Transparent meshes are drawn
// back-to-front after all opaque meshes, with blending enabled and depth writes off.
//
// Face culling is controlled by the `side` property:
//   - "front"  – Only front faces are visible (back faces culled). This is the default.
//   - "back"   – Only back faces are visible (front faces culled).
//   - "double" – Both faces are visible (no culling).
//
// Custom shaders let users inject code snippets into the vertex and fragment stages of
// existing materials. The engine inserts the snippet at a well-defined hook point so users
// can modify worldPos/normal/uv (vertex) or finalColor/alpha (fragment) without writing a
// full shader from scratch. Both WGSL (WebGPU) and GLSL (WebGL2) snippets can be provided.
// Custom uniforms allow passing arbitrary float values from JS to the shader via a uniform
// buffer. Declare them as `uniforms: { time: 0 }` in the CustomShader and access them in
// shader code as `uniforms.time`. Update values each frame from JS by writing directly to
// the uniforms object (e.g. `material.customShader.uniforms.time = elapsed`).
//
// new BasicMaterial()   – Unlit material (ignores lights).
// new LambertMaterial() – Diffuse-lit material (reacts to lights).

import type { Texture } from './texture'

export interface CustomShader {
  /** Float uniforms accessible in shader code as `uniforms.xxx`. Update from JS each frame. */
  uniforms?: Record<string, number>
  /** WGSL code injected into the vertex shader. Can read/write: out.worldPos, out.normal, out.uv. */
  vertexWGSL?: string
  /** WGSL code injected into the fragment shader. Can read/write: finalColor (vec3f), alpha (f32). */
  fragmentWGSL?: string
  /** GLSL code injected into the vertex shader. Can read/write: v_worldPos, v_normal, v_uv. */
  vertexGLSL?: string
  /** GLSL code injected into the fragment shader. Can read/write: finalColor (vec3), alpha (float). */
  fragmentGLSL?: string
}

export interface PaletteEntry {
  color: [number, number, number]
  emissive?: [number, number, number]
  emissiveIntensity?: number
  tiledAo?: Texture
  tiledAoIntensity?: number
  tiledAoScale?: number
  tiledNormal?: Texture
  tiledNormalIntensity?: number
  tiledNormalScale?: number
}

export type MaterialType = 'basic' | 'lambert'
export type MaterialSide = 'front' | 'back' | 'double'

let _nextMaterialId = 0

export class Material {
  readonly _id: number
  type: MaterialType
  color: [number, number, number]
  vertexColors: boolean
  palette?: PaletteEntry[]
  opacity: number
  transparent: boolean
  side: MaterialSide

  // Lambert-specific
  receiveShadow: boolean

  // Emissive
  emissiveBrightness: number

  // Texture maps
  colorMap?: Texture
  aoMap?: Texture
  aoIntensity: number

  // Custom shaders
  customShader?: CustomShader

  // Computed
  _hasEmissive = false
  _hasTextures = false
  _hasCustomShader = false
  _hasCustomUniforms = false
  needsUpdate = true

  constructor(type: MaterialType, opts: MaterialOptions = {}) {
    this._id = _nextMaterialId++
    this.type = type
    this.color = opts.color ?? [1, 1, 1]
    this.vertexColors = opts.vertexColors ?? false
    this.receiveShadow = opts.receiveShadow ?? true
    this.opacity = opts.opacity ?? 1.0
    this.transparent = opts.transparent ?? false
    this.side = opts.side ?? (opts.transparent ? 'double' : 'front')
    this.emissiveBrightness = opts.emissiveBrightness ?? 1.0
    this.colorMap = opts.colorMap
    this.aoMap = opts.aoMap
    this.aoIntensity = opts.aoIntensity ?? 1.0
    this._hasTextures = !!(opts.colorMap || opts.aoMap)

    if (opts.customShader) {
      this.customShader = opts.customShader
      this._hasCustomShader = true
      this._hasCustomUniforms = !!(opts.customShader.uniforms && Object.keys(opts.customShader.uniforms).length > 0)
    }

    if (opts.palette) {
      this.palette = opts.palette
    }
  }
}

export interface MaterialOptions {
  color?: [number, number, number]
  vertexColors?: boolean
  receiveShadow?: boolean
  palette?: PaletteEntry[]
  emissiveBrightness?: number
  opacity?: number
  transparent?: boolean
  side?: MaterialSide
  colorMap?: Texture
  aoMap?: Texture
  aoIntensity?: number
  customShader?: CustomShader
}

export class BasicMaterial extends Material {
  constructor(opts: MaterialOptions = {}) {
    super('basic', opts)
  }
}

export class LambertMaterial extends Material {
  constructor(opts: MaterialOptions = {}) {
    super('lambert', opts)
  }
}
