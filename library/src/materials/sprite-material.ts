// SpriteMaterial – Material designed for Sprite objects (camera-facing billboard planes).
//
// SpriteMaterial is an unlit material (like BasicMaterial) with defaults tuned for sprites:
//   - transparent: true  – Sprites are transparent by default (opt out with transparent: false).
//   - side: 'double'     – Both faces are visible (sprites can be viewed from any angle).
//
// Sprite-specific properties:
//   - rotation          – 2D rotation angle (radians) around the billboard's view axis. Default 0.
//   - sizeAttenuation   – When true (default), sprites shrink with distance like normal geometry.
//                          When false, sprites maintain constant screen size regardless of distance.
//
// new SpriteMaterial() – Creates a sprite material with default settings.

import { Material } from './material'

import type { MaterialOptions } from './material'

export interface SpriteMaterialOptions extends MaterialOptions {
  rotation?: number
  sizeAttenuation?: boolean
}

export class SpriteMaterial extends Material {
  rotation: number
  sizeAttenuation: boolean

  constructor(opts: SpriteMaterialOptions = {}) {
    super('basic', {
      transparent: true,
      side: 'double',
      ...opts,
    })
    this.rotation = opts.rotation ?? 0
    this.sizeAttenuation = opts.sizeAttenuation ?? true
  }
}
