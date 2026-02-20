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
// new BasicMaterial()   – Unlit material (ignores lights).
// new LambertMaterial() – Diffuse-lit material (reacts to lights).

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

  // Lambert-specific
  receiveShadow: boolean
  aoTexture?: unknown // For future texture support

  // Computed
  _hasEmissive = false
  needsUpdate = true

  constructor(type: MaterialType, opts: MaterialOptions = {}) {
    this._id = _nextMaterialId++
    this.type = type
    this.color = opts.color ?? [1, 1, 1]
    this.vertexColors = opts.vertexColors ?? false
    this.receiveShadow = opts.receiveShadow ?? true

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
