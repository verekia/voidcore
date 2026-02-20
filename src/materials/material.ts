// Material – Defines how a mesh surface looks when rendered.
//
// A material controls the color and shading model of a mesh. Two shading types are supported:
//   - "basic"   – Unlit flat color (ignores lights, good for UI or debug).
//   - "lambert" – Diffuse shading that reacts to directional and ambient light.
//
// Materials can also use a palette: an array of up to 32 color entries. Each vertex in the
// geometry can reference a palette index, allowing a single mesh to display multiple colors
// without needing textures. Palette entries can also specify emissive colors (self-glowing).
//
// Texture maps add per-pixel detail from images:
//   - colorMap – Multiplies the base color by a texture sample (diffuse/albedo map).
//   - aoMap    – Multiplies ambient light by the red channel of a texture (ambient occlusion).
// When a material has texture maps, the renderer uses a textured shader variant that samples
// the textures using the mesh's UV coordinates.
//
// Transparency is supported via sorted alpha blending. Set `transparent: true` and an
// `opacity` value (0–1) to make a material see-through. Transparent meshes are drawn
// back-to-front after all opaque meshes, with blending enabled and depth writes off.
//
// new BasicMaterial()   – Unlit material (ignores lights).
// new LambertMaterial() – Diffuse-lit material (reacts to lights).

import type { Texture } from './texture'

export interface PaletteEntry {
  color: [number, number, number]
  emissive?: [number, number, number]
  emissiveIntensity?: number
}

export type MaterialType = 'basic' | 'lambert'

let _nextMaterialId = 0

export class Material {
  readonly _id: number
  type: MaterialType
  color: [number, number, number]
  vertexColors: boolean
  palette?: PaletteEntry[]
  opacity: number
  transparent: boolean

  // Lambert-specific
  receiveShadow: boolean

  // Texture maps
  colorMap?: Texture
  aoMap?: Texture

  // Computed
  _hasEmissive = false
  _hasTextures = false
  needsUpdate = true

  constructor(type: MaterialType, opts: MaterialOptions = {}) {
    this._id = _nextMaterialId++
    this.type = type
    this.color = opts.color ?? [1, 1, 1]
    this.vertexColors = opts.vertexColors ?? false
    this.receiveShadow = opts.receiveShadow ?? true
    this.opacity = opts.opacity ?? 1.0
    this.transparent = opts.transparent ?? false
    this.colorMap = opts.colorMap
    this.aoMap = opts.aoMap
    this._hasTextures = !!(opts.colorMap || opts.aoMap)

    if (opts.palette) {
      this.palette = opts.palette
      for (const entry of opts.palette) {
        if (!this._hasEmissive && entry.emissive && entry.emissiveIntensity && entry.emissiveIntensity > 0) {
          this._hasEmissive = true
        }
      }
    }
  }
}

export interface MaterialOptions {
  color?: [number, number, number]
  vertexColors?: boolean
  receiveShadow?: boolean
  palette?: PaletteEntry[]
  opacity?: number
  transparent?: boolean
  colorMap?: Texture
  aoMap?: Texture
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
