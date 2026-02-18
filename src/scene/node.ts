import { mat4Compose, mat4Copy, mat4Create, mat4Multiply, quatCreate, vec3Create } from '../math/index.ts'

import type { Mat4, Quat, Vec3 } from '../math/index.ts'

export type NodeType = 'group' | 'mesh' | 'camera' | 'directionalLight'

export class Node {
  name = ''
  type: NodeType = 'group'

  position: Vec3 = vec3Create()
  rotation: Quat = quatCreate()
  scale: Vec3 = new Float32Array([1, 1, 1])

  parent: Node | null = null
  children: Node[] = []

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
    }
  }

  remove(child: Node) {
    const idx = this.children.indexOf(child)
    if (idx !== -1) {
      this.children.splice(idx, 1)
      child.parent = null
    }
  }

  traverse(callback: (node: Node) => void) {
    callback(this)
    for (let i = 0; i < this.children.length; i++) {
      this.children[i]!.traverse(callback)
    }
  }

  lookAt(target: Vec3 | [number, number, number]) {
    const t = target instanceof Float32Array ? target : new Float32Array(target)
    // Simplified lookAt - sets rotation to face target
    const forward = vec3Create()
    forward[0] = t[0]! - this.position[0]!
    forward[1] = t[1]! - this.position[1]!
    forward[2] = t[2]! - this.position[2]!
    const len = Math.sqrt(forward[0]! ** 2 + forward[1]! ** 2 + forward[2]! ** 2)
    if (len < 1e-6) return
    forward[0]! /= len
    forward[1]! /= len
    forward[2]! /= len

    // Z-up: compute rotation from default forward (0,1,0) to target direction
    // Using a simplified approach with atan2
    const yaw = Math.atan2(forward[0]!, forward[1]!)
    const pitch = Math.asin(Math.max(-1, Math.min(1, forward[2]!)))

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

export const propagateDirty = (node: Node): void => {
  if (node._dirtyLocal) node._dirtyWorld = true
  if (node._dirtyWorld) {
    for (let i = 0; i < node.children.length; i++) {
      const child = node.children[i]!
      if (!child._dirtyWorld) {
        child._dirtyWorld = true
        propagateDirty(child)
      }
    }
  }
}

export const updateWorldMatrices = (node: Node): void => {
  if (node._dirtyLocal) {
    mat4Compose(node._localMatrix, node.position, node.rotation, node.scale)
    node._dirtyLocal = false
  }
  if (node._dirtyWorld) {
    if (node.parent) {
      mat4Multiply(node._worldMatrix, node.parent._worldMatrix, node._localMatrix)
    } else {
      mat4Copy(node._worldMatrix, node._localMatrix)
    }
    node._dirtyWorld = false
  }
  for (let i = 0; i < node.children.length; i++) {
    updateWorldMatrices(node.children[i]!)
  }
}
