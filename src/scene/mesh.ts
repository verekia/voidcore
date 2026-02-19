// Mesh – A visible 3D object composed of a Geometry (shape) and a Material (appearance).
//
// Meshes are the main renderable objects in the scene. The renderer draws each visible mesh
// by binding its geometry buffers and material uniforms, then issuing a draw call. A mesh
// can optionally have a Skeleton for skeletal animation (skinned meshes).
//
// new Mesh(geometry, material) – Creates a mesh from a geometry and material.
// Both parameters are optional to support deferred attachment (e.g. React reconciler).

import { Node } from './node.ts'

import type { Skeleton } from '../animation/skeleton.ts'
import type { Geometry } from '../geometry/geometry.ts'
import type { Material } from '../materials/material.ts'

export class Mesh extends Node {
  geometry!: Geometry
  material!: Material
  skeleton?: Skeleton
  _batchIndex = 0
  _batchFrame = -1
  _isSkinned = false

  constructor(geometry?: Geometry, material?: Material) {
    super()
    this.type = 'mesh'
    if (geometry) this.geometry = geometry
    if (material) this.material = material
  }
}
