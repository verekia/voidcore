// Camera – Defines how the 3D scene is projected onto the 2D screen.
//
// A PerspectiveCamera uses perspective projection: objects farther away appear smaller
// (like real human vision). It is defined by:
//   fov    – Field of view angle in degrees (how "wide" the lens is)
//   near   – Near clipping plane (objects closer than this are invisible)
//   far    – Far clipping plane (objects farther than this are invisible)
//   aspect – Width/height ratio of the viewport (set automatically by the renderer)
//
// The camera also inherits position/rotation from Node, so it can be placed in the scene.
// The renderer uses two matrices from the camera:
//   Projection matrix – Converts 3D eye-space coords to clip space (applies perspective)
//   View matrix       – Converts world-space coords to eye-space (set by orbit controls)
//
// new PerspectiveCamera()   – Constructor with default values (60° FOV, 0.1–1000 range).

import { mat4Create, mat4Perspective } from '../math/index.ts'
import { Node } from './node.ts'

import type { Mat4 } from '../math/index.ts'

export interface CameraOptions {
  fov?: number
  near?: number
  far?: number
}

export class PerspectiveCamera extends Node {
  fov: number
  near: number
  far: number
  aspect = 1

  _projectionMatrix: Mat4 = mat4Create()
  _viewMatrix: Mat4 = mat4Create()
  _viewProjectionMatrix: Mat4 = mat4Create()
  _projectionDirty = true

  constructor(opts: CameraOptions = {}) {
    super()
    this.type = 'camera'
    this.fov = opts.fov ?? 60
    this.near = opts.near ?? 0.1
    this.far = opts.far ?? 1000
    this.frustumCulled = false
    this.castShadow = false
    this.receiveShadow = false
  }

  updateProjection(depth: 'zero-to-one' | 'neg-one-to-one') {
    mat4Perspective(this._projectionMatrix, (this.fov * Math.PI) / 180, this.aspect, this.near, this.far, depth)
    this._projectionDirty = false
  }
}
