import { Node } from './node.ts'

import type { Skeleton } from '../animation/skeleton.ts'
import type { Geometry } from '../geometry/geometry.ts'
import type { Material } from '../materials/material.ts'

export class Mesh extends Node {
  geometry: Geometry
  material: Material
  skeleton?: Skeleton

  constructor(geometry: Geometry, material: Material) {
    super()
    this.type = 'mesh'
    this.geometry = geometry
    this.material = material
  }
}

export const createMesh = (geometry: Geometry, material: Material): Mesh => new Mesh(geometry, material)
