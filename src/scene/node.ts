// Node – Base class for all objects in the scene graph (the tree of 3D objects).
//
// Every object in the scene (meshes, cameras, lights, groups) is a Node. Nodes form a
// parent-child tree: moving a parent automatically moves all its children. Each node
// stores a local transform (position, rotation, scale) and a computed world matrix.
//
// The scene graph update traverses the tree top-down: if a node's local transform changed
// (_dirtyLocal), its local matrix is recomputed from position/rotation/scale via mat4Compose.
// If either the local matrix or a parent changed, the world matrix is recomputed as
// worldMatrix = parent.worldMatrix × localMatrix.
//
// node.setPosition(x, y, z) – Sets position and marks the transform dirty.
// node.setPositionX/Y/Z(v) – Sets a single position component and marks dirty.
// node.setRotation(x, y, z, w) – Sets quaternion rotation and marks the transform dirty.
// node.setScale(x, y, z) or node.setScale(s) – Sets scale (uniform or per-axis) and marks dirty.
// node.setScaleX/Y/Z(v) – Sets a single scale component and marks dirty.
// node.markTransformDirty() – Marks the local transform as needing recalculation (use after
//   directly writing to the position/rotation/scale Float32Arrays via math utilities).
// node.add()     – Adds child nodes (reparenting if already attached elsewhere).
// node.remove()  – Detaches a child node.
// node.traverse() – Walks the subtree, calling a callback on each node.
// node.lookAt()  – Orients the node to face a target point (Z-up convention).
// updateWorldMatrices() – Recursively recomputes world matrices for dirty nodes.

import { mat4Compose, mat4Copy, mat4Create, mat4Multiply, quatCreate, vec3Create } from '../math/index'

import type { Mat4, Quat, Vec3 } from '../math/index'

export type NodeType = 'group' | 'mesh' | 'camera' | 'directionalLight' | 'ambientLight'

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
    const fx = _lookAtFwd[0]!,
      fy = _lookAtFwd[1]!,
      fz = _lookAtFwd[2]!
    const len = Math.sqrt(fx * fx + fy * fy + fz * fz)
    if (len < 1e-6) return
    const invLen = 1 / len
    _lookAtFwd[0] = _lookAtFwd[0]! * invLen
    _lookAtFwd[1] = _lookAtFwd[1]! * invLen
    _lookAtFwd[2] = _lookAtFwd[2]! * invLen

    // Z-up: compute rotation from default forward (0,1,0) to target direction
    const yaw = Math.atan2(_lookAtFwd[0]!, _lookAtFwd[1]!)
    const pitch = Math.asin(Math.max(-1, Math.min(1, _lookAtFwd[2]!)))

    // Convert to quaternion via ZXY order for Z-up
    const cy = Math.cos(yaw * 0.5),
      sy = Math.sin(yaw * 0.5)
    const cp = Math.cos(pitch * 0.5),
      sp = Math.sin(pitch * 0.5)

    this.setRotation(-sp * sy, sp * cy, sy * cp, cy * cp)
  }

  setPosition(x: number, y: number, z: number) {
    this.position[0] = x
    this.position[1] = y
    this.position[2] = z
    this._dirtyLocal = true
  }

  setPositionX(x: number) {
    this.position[0] = x
    this._dirtyLocal = true
  }

  setPositionY(y: number) {
    this.position[1] = y
    this._dirtyLocal = true
  }

  setPositionZ(z: number) {
    this.position[2] = z
    this._dirtyLocal = true
  }

  setRotation(x: number, y: number, z: number, w: number) {
    this.rotation[0] = x
    this.rotation[1] = y
    this.rotation[2] = z
    this.rotation[3] = w
    this._dirtyLocal = true
  }

  setScale(x: number, y?: number, z?: number) {
    if (y === undefined) {
      this.scale[0] = x
      this.scale[1] = x
      this.scale[2] = x
    } else {
      this.scale[0] = x
      this.scale[1] = y
      this.scale[2] = z!
    }
    this._dirtyLocal = true
  }

  setScaleX(x: number) {
    this.scale[0] = x
    this._dirtyLocal = true
  }

  setScaleY(y: number) {
    this.scale[1] = y
    this._dirtyLocal = true
  }

  setScaleZ(z: number) {
    this.scale[2] = z
    this._dirtyLocal = true
  }

  markTransformDirty() {
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
