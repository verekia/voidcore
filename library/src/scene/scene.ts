// Scene – The root container for all 3D objects, lights, and environment settings.
//
// The Scene is the top-level Node of the scene graph. It holds a name registry for quick
// node lookup by name. Ambient light is provided by adding an <ambientLight> node to the
// scene graph (just like directional lights).
//
// Each frame, the renderer calls scene.updateGraph() which drives an iterative depth-first
// traversal to recompute world matrices for any nodes whose transforms have changed.
// Pre-allocated nodeStack and dirtyStack arrays are kept on the Scene to avoid per-frame
// allocations. Subtrees whose _subtreeDirty flag is false and whose parent world matrix
// did not change are skipped entirely, so static objects cost nothing once settled.
//
// new Scene()          – Creates an empty scene.
// scene.getByName()    – Finds any node in the scene by its name string (O(1) lookup).
// scene.updateGraph()  – Recomputes all dirty world matrices in the scene graph.

import { Node, updateWorldMatrices } from './node'

export class Scene extends Node {
  private _nameMap = new Map<string, Node>()
  // Pre-allocated stacks for the iterative updateWorldMatrices traversal.
  // Sized for typical scenes; JavaScript arrays grow automatically if exceeded.
  private _updateNodeStack: Node[] = []
  private _updateDirtyStack: boolean[] = []

  constructor() {
    super()
    this.frustumCulled = false
    this.castShadow = false
    this.receiveShadow = false
    // Set _scene on the root so Node.add can register names
    this._scene = this
  }

  getByName(name: string): Node | undefined {
    return this._nameMap.get(name)
  }

  updateGraph() {
    updateWorldMatrices(this, this._updateNodeStack, this._updateDirtyStack)
  }

  _registerNames(node: Node) {
    if (node.name) this._nameMap.set(node.name, node)
    for (const child of node.children) this._registerNames(child)
  }

  _unregisterNames(node: Node) {
    if (node.name) this._nameMap.delete(node.name)
    for (const child of node.children) this._unregisterNames(child)
  }
}
