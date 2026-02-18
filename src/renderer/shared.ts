import { aabbTransform, frustumContainsAABB, vec3Normalize } from '../math/index.ts'
import { Mesh } from '../scene/mesh.ts'

import type { AABB } from '../math/index.ts'
import type { DirectionalLight } from '../scene/light.ts'
import type { Node } from '../scene/node.ts'

// Pre-allocated result object to avoid per-frame allocations
export interface TraversalResult {
  culledCount: number
  dirLight: DirectionalLight | null
}

const _traversalResult: TraversalResult = { culledCount: 0, dirLight: null }

/**
 * Collect visible meshes and find the first directional light in a single traversal.
 * Uses a pre-allocated stack and result object to avoid allocations.
 */
export const collectVisibleMeshes = (
  root: Node,
  frustumPlanes: Float32Array,
  worldAABB: AABB,
  meshes: Mesh[],
  stack: Node[],
): TraversalResult => {
  meshes.length = 0
  let culledCount = 0
  let dirLight: DirectionalLight | null = null
  stack.length = 0
  stack.push(root)
  while (stack.length > 0) {
    const node = stack.pop()!
    if (!node.visible) continue
    if (node.type === 'mesh') {
      const mesh = node as Mesh
      if (mesh.frustumCulled) {
        aabbTransform(worldAABB, mesh.geometry.aabb, mesh._worldMatrix)
        if (!frustumContainsAABB(frustumPlanes, worldAABB)) {
          culledCount++
        } else {
          meshes.push(mesh)
        }
      } else {
        meshes.push(mesh)
      }
    }
    if (!dirLight && node.type === 'directionalLight') {
      dirLight = node as DirectionalLight
    }
    const children = node.children
    for (let i = children.length - 1; i >= 0; i--) {
      stack.push(children[i]!)
    }
  }
  _traversalResult.culledCount = culledCount
  _traversalResult.dirLight = dirLight
  return _traversalResult
}

/** Compute normalized light direction from a directional light's world position. */
export const computeLightDir = (
  lightDir: Float32Array,
  tempVec3: Float32Array,
  dirLight: DirectionalLight | null,
): void => {
  lightDir[0] = 0
  lightDir[1] = 0
  lightDir[2] = 0
  if (dirLight) {
    const lp = dirLight._worldMatrix
    tempVec3[0] = lp[12]!
    tempVec3[1] = lp[13]!
    tempVec3[2] = lp[14]!
    vec3Normalize(lightDir, tempVec3)
  }
}
