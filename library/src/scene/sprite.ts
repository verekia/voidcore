// Sprite – A billboard plane that always faces the camera.
//
// Sprites are flat rectangular planes that automatically orient to face the camera every frame.
// They use a shared 1x1 PlaneGeometry (centered at origin on the XY plane) and are rendered
// through the existing mesh pipeline with a CPU-computed billboard world matrix.
//
// Key differences from Mesh:
//   - castShadow defaults to false (sprites don't cast shadows by default).
//   - Uses SpriteMaterial by default (transparent, double-sided, unlit).
//   - The renderer replaces the sprite's world matrix with a billboard matrix that aligns
//     the plane to always face the camera, preserving position and scale.
//   - Scale X/Y controls the sprite's width/height in world units.
//
// new Sprite()          – Creates a sprite with a default SpriteMaterial.
// new Sprite(material)  – Creates a sprite with a custom SpriteMaterial.

import { PlaneGeometry } from '../geometry/primitives'
import { SpriteMaterial } from '../materials/sprite-material'
import { Mesh } from './mesh'

import type { Geometry } from '../geometry/geometry'

// Shared sprite geometry — all sprites use the same 1x1 plane to avoid per-sprite allocation.
let _spriteGeometry: Geometry | null = null
const getSpriteGeometry = (): Geometry => {
  if (!_spriteGeometry) {
    _spriteGeometry = new PlaneGeometry()
  }
  return _spriteGeometry
}

export class Sprite extends Mesh {
  constructor(material?: SpriteMaterial) {
    super(getSpriteGeometry(), material ?? new SpriteMaterial())
    this.type = 'sprite'
    this.castShadow = false
  }
}
