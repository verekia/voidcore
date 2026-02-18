export interface PaletteEntry {
  color: [number, number, number]
  opacity?: number
  emissive?: [number, number, number]
  emissiveIntensity?: number
}

export type MaterialType = 'basic' | 'lambert'

let _nextMaterialId = 0

export class Material {
  readonly _id: number
  type: MaterialType
  color: [number, number, number]
  opacity: number
  transparent: boolean
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
    this.opacity = opts.opacity ?? 1.0
    this.transparent = opts.transparent ?? false
    this.vertexColors = opts.vertexColors ?? false
    this.receiveShadow = opts.receiveShadow ?? true

    if (opts.palette) {
      this.palette = opts.palette
      for (const entry of opts.palette) {
        if (!this._hasEmissive && entry.emissive && entry.emissiveIntensity && entry.emissiveIntensity > 0) {
          this._hasEmissive = true
        }
        if (entry.opacity !== undefined && entry.opacity < 1.0) {
          this.transparent = true
        }
      }
    }

    if (this.opacity < 1.0) this.transparent = true
  }
}

export interface MaterialOptions {
  color?: [number, number, number]
  opacity?: number
  transparent?: boolean
  vertexColors?: boolean
  receiveShadow?: boolean
  palette?: PaletteEntry[]
}

export const createBasicMaterial = (opts: MaterialOptions = {}): Material => new Material('basic', opts)

export const createLambertMaterial = (opts: MaterialOptions = {}): Material => new Material('lambert', opts)
