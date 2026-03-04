// Mesh – A visible 3D object composed of a Geometry (shape) and a Material (appearance).
//
// Meshes are the main renderable objects in the scene. The renderer draws each visible mesh
// by binding its geometry buffers and material uniforms, then issuing a draw call. A mesh
// can optionally have a Skeleton for skeletal animation (skinned meshes).
//
// Distance culling: set `maxDistance` to a positive number to automatically hide the mesh
// when the camera is farther than that distance from the mesh's world position. Uses squared
// distance comparison (no sqrt) for zero-allocation performance. A value of 0 (default)
// disables distance culling. Distance-culled meshes are excluded from both camera rendering
// and shadow casting.
//
// Outlines use the inverted hull technique merged into a single draw call: the geometry is
// doubled (original + outline vertices with smooth normals), and the
// fragment shader uses front_facing to discard front-facing outline triangles. This halves
// draw call count for outlined meshes. Set `outline` to a number (thickness) or an object
// with `thickness`, `color`, and `maxDistance` to enable. Outlines work with both static and
// skinned meshes. The `maxDistance` on the outline object (default 1000) sets outline thickness
// to 0 in the UBO when the camera is farther than that distance (squared distance comparison,
// zero allocation), and the shader discards all outline fragments when thickness is 0.
//
// Per-mesh tint overlays an unlit color on Lambert meshes for damage flash effects.
// `tintColor` sets the overlay RGB color and `tintIntensity` (0–1) controls the blend:
// at 0 the mesh renders normally, at 1 it is fully replaced by the unlit tint color.
// Animate tintIntensity from 1→0 over ~200ms for a smooth damage flash. Outlines are
// unaffected by the tint since they early-return before the tint mix in the fragment shader.
//
// new Mesh(geometry, material) – Creates a mesh from a geometry and material.
// Both parameters are optional to support deferred attachment (e.g. React reconciler).

import { Node } from './node'

import type { Skeleton } from '../animation/skeleton'
import type { Geometry } from '../geometry/geometry'
import type { Material } from '../materials/material'

export interface MeshOutline {
  thickness: number
  color?: [number, number, number]
  maxDistance?: number
}

// Default outline color (shared reference, avoids allocation per getter call)
const _defaultOutlineColor: [number, number, number] = [0, 0, 0]

export class Mesh extends Node {
  geometry!: Geometry
  material!: Material
  skeleton?: Skeleton
  outline?: MeshOutline | number
  maxDistance = 0
  tintColor: [number, number, number] = [1, 1, 1]
  tintIntensity = 0
  _batchIndex = 0
  _batchFrame = -1
  _isSkinned = false

  constructor(geometry?: Geometry, material?: Material) {
    super()
    this.type = 'mesh'
    if (geometry) this.geometry = geometry
    if (material) this.material = material
  }

  /** Resolved outline thickness (0 if no outline). */
  get _outlineThickness(): number {
    if (this.outline == null) return 0
    return typeof this.outline === 'number' ? this.outline : this.outline.thickness
  }

  /** Resolved outline color (default black). */
  get _outlineColor(): [number, number, number] {
    if (this.outline == null || typeof this.outline === 'number') return _defaultOutlineColor
    return this.outline.color ?? _defaultOutlineColor
  }

  /** Resolved outline max distance (0 = no distance culling, default 1000). */
  get _outlineMaxDistance(): number {
    if (this.outline == null || typeof this.outline === 'number') return 1000
    return this.outline.maxDistance ?? 1000
  }
}
