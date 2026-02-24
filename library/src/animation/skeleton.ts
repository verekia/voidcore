// Skeleton – Manages a set of bones for skeletal (skinned) animation.
//
// A skeleton is a hierarchy of bones (scene nodes) plus "inverse bind matrices" that
// define how the mesh was originally posed. Each frame, the skeleton computes a
// "bone matrix" for every bone: boneMatrix = bone.worldMatrix * inverseBindMatrix.
// These matrices are written directly into the shared boneMatrices buffer using
// mat4MultiplyInto, avoiding a temporary matrix and a per-bone .set() copy.
// The resulting matrices are uploaded to the GPU so the vertex shader can deform
// the mesh vertices according to the current pose.
//
// Skeleton.update()   – Recomputes all bone matrices (called once per frame when dirty).
// Skeleton.getBone()  – Finds a bone node by name.

import { mat4MultiplyInto } from '../math/index'

import type { Mat4 } from '../math/index'
import type { Node } from '../scene/node'

export class Skeleton {
  bones: Node[]
  boneInverseBindMatrices: Mat4[]
  boneMatrices: Float32Array
  _dirty = true

  constructor(bones: Node[], inverseBindMatrices: Mat4[]) {
    this.bones = bones
    this.boneInverseBindMatrices = inverseBindMatrices
    // 32 bones max, 16 floats per mat4
    this.boneMatrices = new Float32Array(32 * 16)
  }

  getBone(name: string): Node | undefined {
    for (let i = 0; i < this.bones.length; i++) {
      if (this.bones[i]!.name === name) return this.bones[i]
    }
    return undefined
  }

  update() {
    if (!this._dirty) return
    for (let i = 0; i < this.bones.length; i++) {
      const bone = this.bones[i]!
      const ibm = this.boneInverseBindMatrices[i]!
      // boneMatrix = bone.worldMatrix * inverseBindMatrix (written directly, no temp copy)
      mat4MultiplyInto(this.boneMatrices, i * 16, bone._worldMatrix, ibm)
    }
    this._dirty = false
  }
}
