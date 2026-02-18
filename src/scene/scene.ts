import { Node, propagateDirty, updateWorldMatrices } from './node.ts'

export interface AmbientLight {
  color: [number, number, number]
  intensity: number
}

export class Scene extends Node {
  ambientLight: AmbientLight = { color: [0.4, 0.45, 0.5], intensity: 0.3 }
  private _nameMap = new Map<string, Node>()

  constructor() {
    super()
    this.frustumCulled = false
    this.castShadow = false
    this.receiveShadow = false
  }

  override add(...nodes: Node[]) {
    super.add(...nodes)
    for (const node of nodes) {
      this._registerNames(node)
    }
  }

  override remove(child: Node) {
    this._unregisterNames(child)
    super.remove(child)
  }

  getByName(name: string): Node | undefined {
    return this._nameMap.get(name)
  }

  updateGraph() {
    propagateDirty(this)
    updateWorldMatrices(this)
  }

  private _registerNames(node: Node) {
    if (node.name) this._nameMap.set(node.name, node)
    for (const child of node.children) this._registerNames(child)
  }

  private _unregisterNames(node: Node) {
    if (node.name) this._nameMap.delete(node.name)
    for (const child of node.children) this._unregisterNames(child)
  }
}

export const createScene = (): Scene => new Scene()
