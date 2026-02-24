// Node – Base class for all objects in the scene graph (the tree of 3D objects).
//
// Every object in the scene (meshes, sprites, cameras, lights, groups) is a Node. Nodes form a
// parent-child tree: moving a parent automatically moves all its children. Each node
// stores a local transform (position, rotation, scale) and a computed world matrix.
//
// The scene graph update traverses the tree top-down: if a node's local transform changed
// (_dirtyLocal), its local matrix is recomputed from position/rotation/scale via mat4Compose.
// If either the local matrix or a parent changed, the world matrix is recomputed as
// worldMatrix = parent.worldMatrix × localMatrix.
//
// _subtreeDirty is propagated upward to ancestors whenever a node's local transform is
// marked dirty. Scene.updateGraph() uses it to skip entire subtrees that have nothing dirty,
// avoiding visits to static nodes when only animated characters have changed.
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
// updateWorldMatrices() – Iteratively recomputes world matrices for dirty nodes using
//   pre-allocated stacks to avoid recursion overhead and per-call allocations.

import { mat4Compose, mat4Copy, mat4Create, mat4Multiply, quatCreate, vec3Create } from '../math/index'

import type { Mat4, Quat, Vec3 } from '../math/index'

export type NodeType = 'group' | 'mesh' | 'sprite' | 'camera' | 'directionalLight' | 'ambientLight'

// Scratch for lookAt — avoids per-call allocation
const _lookAtFwd = vec3Create()

// Mark a node and all its ancestors as having a dirty subtree.
// Stops early once an ancestor already has _subtreeDirty set (it was already propagated).
// Only propagates on the FIRST dirty call per traversal cycle to keep the setter cost minimal.
const _markSubtreeDirty = (node: Node): void => {
  node._subtreeDirty = true
  let n: Node | null = node.parent
  while (n !== null && !n._subtreeDirty) {
    n._subtreeDirty = true
    n = n.parent
  }
}

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
  castShadow = false
  receiveShadow = false

  _localMatrix: Mat4 = mat4Create()
  _worldMatrix: Mat4 = mat4Create()
  _dirtyLocal = true
  _dirtyWorld = true
  // True when this node or any descendant has a dirty local transform.
  // Propagated upward on every setPosition/setRotation/setScale call.
  // Cleared by updateWorldMatrices as nodes are visited, enabling subtree skipping.
  _subtreeDirty = true

  add(...nodes: Node[]) {
    for (const child of nodes) {
      if (child.parent) child.parent.remove(child)
      child.parent = this
      this.children.push(child)
      child._dirtyWorld = true
      // Propagate up so the traversal knows to visit this subtree
      _markSubtreeDirty(child)
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
    if (!this._dirtyLocal) {
      this._dirtyLocal = true
      _markSubtreeDirty(this)
    }
  }

  setPositionX(x: number) {
    this.position[0] = x
    if (!this._dirtyLocal) {
      this._dirtyLocal = true
      _markSubtreeDirty(this)
    }
  }

  setPositionY(y: number) {
    this.position[1] = y
    if (!this._dirtyLocal) {
      this._dirtyLocal = true
      _markSubtreeDirty(this)
    }
  }

  setPositionZ(z: number) {
    this.position[2] = z
    if (!this._dirtyLocal) {
      this._dirtyLocal = true
      _markSubtreeDirty(this)
    }
  }

  setRotation(x: number, y: number, z: number, w: number) {
    this.rotation[0] = x
    this.rotation[1] = y
    this.rotation[2] = z
    this.rotation[3] = w
    if (!this._dirtyLocal) {
      this._dirtyLocal = true
      _markSubtreeDirty(this)
    }
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
    if (!this._dirtyLocal) {
      this._dirtyLocal = true
      _markSubtreeDirty(this)
    }
  }

  setScaleX(x: number) {
    this.scale[0] = x
    if (!this._dirtyLocal) {
      this._dirtyLocal = true
      _markSubtreeDirty(this)
    }
  }

  setScaleY(y: number) {
    this.scale[1] = y
    if (!this._dirtyLocal) {
      this._dirtyLocal = true
      _markSubtreeDirty(this)
    }
  }

  setScaleZ(z: number) {
    this.scale[2] = z
    if (!this._dirtyLocal) {
      this._dirtyLocal = true
      _markSubtreeDirty(this)
    }
  }

  markTransformDirty() {
    if (!this._dirtyLocal) {
      this._dirtyLocal = true
      _markSubtreeDirty(this)
    }
  }
}

const propagateScene = (node: Node, scene: Node['_scene']): void => {
  node._scene = scene
  for (let i = 0; i < node.children.length; i++) {
    propagateScene(node.children[i]!, scene)
  }
}

/**
 * Iteratively recompute world matrices for all dirty nodes in the subtree rooted at `root`.
 * Uses pre-allocated node and parentDirty stacks to avoid recursion overhead.
 *
 * Subtree skipping: if a node's _subtreeDirty flag is false and the parent's world matrix
 * did not change (parentDirty=false), the entire subtree is skipped without pushing any
 * children. This avoids visiting static nodes every frame when only animated characters
 * have dirty transforms.
 *
 * _subtreeDirty is cleared on each visited node and re-propagated bottom-up by the setters
 * whenever a transform changes, so skipping is always safe.
 */
export const updateWorldMatrices = (root: Node, nodeStack: Node[], dirtyStack: boolean[]): void => {
  let top = 0
  nodeStack[top] = root
  dirtyStack[top] = false
  top++

  while (top > 0) {
    top--
    const node = nodeStack[top]!
    const parentDirty = dirtyStack[top]!

    // If the parent's world matrix is unchanged and nothing in this subtree is dirty, skip it.
    if (!parentDirty && !node._subtreeDirty) continue

    // Clear the flag — this subtree is being processed now.
    node._subtreeDirty = false

    let worldDirty = parentDirty || node._dirtyWorld

    if (node._dirtyLocal) {
      mat4Compose(node._localMatrix, node.position, node.rotation, node.scale)
      node._dirtyLocal = false
      worldDirty = true
    }

    if (worldDirty) {
      if (node.parent) {
        mat4Multiply(node._worldMatrix, node.parent._worldMatrix, node._localMatrix)
      } else {
        mat4Copy(node._worldMatrix, node._localMatrix)
      }
      node._dirtyWorld = false
    }

    // Push children in reverse order so the first child is popped first (DFS pre-order).
    const children = node.children
    for (let i = children.length - 1; i >= 0; i--) {
      nodeStack[top] = children[i]!
      dirtyStack[top] = worldDirty
      top++
    }
  }
}
