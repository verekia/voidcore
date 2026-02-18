import { mat4Create, mat4Perspective } from '../math/index.ts'
import { Node } from './node.ts'

import type { Mat4 } from '../math/index.ts'

export class PerspectiveCamera extends Node {
  fov: number
  near: number
  far: number
  aspect = 1

  _projectionMatrix: Mat4 = mat4Create()
  _viewMatrix: Mat4 = mat4Create()
  _viewProjectionMatrix: Mat4 = mat4Create()
  _projectionDirty = true

  constructor(fov = 60, near = 0.1, far = 1000) {
    super()
    this.type = 'camera'
    this.fov = fov
    this.near = near
    this.far = far
    this.frustumCulled = false
    this.castShadow = false
    this.receiveShadow = false
  }

  updateProjection(depth: 'zero-to-one' | 'neg-one-to-one') {
    mat4Perspective(this._projectionMatrix, (this.fov * Math.PI) / 180, this.aspect, this.near, this.far, depth)
    this._projectionDirty = false
  }
}

export interface CameraOptions {
  fov?: number
  near?: number
  far?: number
}

export const createPerspectiveCamera = (opts: CameraOptions = {}): PerspectiveCamera =>
  new PerspectiveCamera(opts.fov ?? 60, opts.near ?? 0.1, opts.far ?? 1000)
