import { mat4Compose, mat4Copy, mat4Create, mat4Multiply, quatCreate, vec3Create } from '../math/index.ts'

import type { Mat4, Quat, Vec3 } from '../math/index.ts'

export type NodeType = 'group' | 'mesh' | 'camera' | 'directionalLight'

// Scratch for lookAt — avoids per-call allocation
const _lookAtFwd = new Float32Array(3)

export class Node {
  name = ''
  type: NodeType = 'group'

  position: Vec3 = vec3Create()
  rotation: Quat = quatCreate()
  scale: Vec3 = new Float32Array([1, 1, 1])

  parent: Node | null = null
  children: Node[] = []
  _scene: { _registerNames(node: Node): void; _unregisterNames(node: Node): void } | null = null

  visible = true
  frustumCulled = true
  castShadow = true
  receiveShadow = true

  _localMatrix: Mat4 = mat4Create()
  _worldMatrix: Mat4 = mat4Create()
  _dirtyLocal = true
  _dirtyWorld = true

  add(...nodes: Node[]) {
    for (const child of nodes) {
      if (child.parent) child.parent.remove(child)
      child.parent = this
      this.children.push(child)
      child._dirtyWorld = true
      if (this._scene) {
        propagateScene(child, this._scene)
        this._scene._registerNames(child)
      }
    }
  }

  remove(child: Node) {
    const idx = this.children.indexOf(child)
    if (idx !== -1) {
      if (this._scene) {
        this._scene._unregisterNames(child)
      }
      this.children.splice(idx, 1)
      child.parent = null
      propagateScene(child, null)
    }
  }

  traverse(callback: (node: Node) => void) {
    callback(this)
    for (let i = 0; i < this.children.length; i++) {
      this.children[i]!.traverse(callback)
    }
  }

  lookAt(target: Vec3 | [number, number, number]) {
    _lookAtFwd[0] = target[0]! - this.position[0]!
    _lookAtFwd[1] = target[1]! - this.position[1]!
    _lookAtFwd[2] = target[2]! - this.position[2]!
    const len = Math.sqrt(_lookAtFwd[0]! ** 2 + _lookAtFwd[1]! ** 2 + _lookAtFwd[2]! ** 2)
    if (len < 1e-6) return
    const invLen = 1 / len
    _lookAtFwd[0]! *= invLen
    _lookAtFwd[1]! *= invLen
    _lookAtFwd[2]! *= invLen

    // Z-up: compute rotation from default forward (0,1,0) to target direction
    const yaw = Math.atan2(_lookAtFwd[0]!, _lookAtFwd[1]!)
    const pitch = Math.asin(Math.max(-1, Math.min(1, _lookAtFwd[2]!)))

    // Convert to quaternion via ZXY order for Z-up
    const cy = Math.cos(yaw * 0.5),
      sy = Math.sin(yaw * 0.5)
    const cp = Math.cos(pitch * 0.5),
      sp = Math.sin(pitch * 0.5)

    this.rotation[0] = -sp * sy
    this.rotation[1] = sp * cy
    this.rotation[2] = sy * cp
    this.rotation[3] = cy * cp
    this._dirtyLocal = true
  }
}

const propagateScene = (node: Node, scene: Node['_scene']): void => {
  node._scene = scene
  for (let i = 0; i < node.children.length; i++) {
    propagateScene(node.children[i]!, scene)
  }
}

export const updateWorldMatrices = (node: Node, parentDirty = false): void => {
  if (node._dirtyLocal) {
    mat4Compose(node._localMatrix, node.position, node.rotation, node.scale)
    node._dirtyLocal = false
    parentDirty = true
  }
  if (parentDirty || node._dirtyWorld) {
    if (node.parent) {
      mat4Multiply(node._worldMatrix, node.parent._worldMatrix, node._localMatrix)
    } else {
      mat4Copy(node._worldMatrix, node._localMatrix)
    }
    node._dirtyWorld = false
    parentDirty = true
  }
  for (let i = 0; i < node.children.length; i++) {
    updateWorldMatrices(node.children[i]!, parentDirty)
  }
}
