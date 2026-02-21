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
// Outlines use the inverted hull technique: a copy of the mesh is rendered with vertices
// inflated along their normals and front-face culling, so only the back faces peek out
// behind the original mesh, creating a silhouette effect. Set `outline` to a number
// (thickness) or an object with `thickness`, `color`, and `maxDistance` to enable. Outlines
// work with both static and skinned meshes. The optional `maxDistance` on the outline object
// skips the outline draw call when the camera is farther than that distance (squared distance
// comparison, zero allocation).
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

export class Mesh extends Node {
  geometry!: Geometry
  material!: Material
  skeleton?: Skeleton
  outline?: MeshOutline | number
  maxDistance = 0
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
    if (this.outline == null || typeof this.outline === 'number') return [0, 0, 0]
    return this.outline.color ?? [0, 0, 0]
  }

  /** Resolved outline max distance (0 = no distance culling). */
  get _outlineMaxDistance(): number {
    if (this.outline == null || typeof this.outline === 'number') return 0
    return this.outline.maxDistance ?? 0
  }
}
