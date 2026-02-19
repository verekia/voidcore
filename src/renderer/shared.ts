// Shared Renderer Utilities – Logic used by both WebGPU and WebGL2 renderers.
//
// findDirectionalLight() – Quick scene graph traversal that returns the first directional
//   light found. Stops early once a light is located. Used to determine light direction
//   before computing cascade shadow maps.
//
// collectMeshes() – Walks the scene graph in a single pass to:
//   1. Collect camera-visible Mesh nodes (frustum culled against the camera frustum)
//   2. Collect shadow-only casters (meshes outside camera frustum but inside the broadest
//      shadow cascade frustum). This merges what used to be two separate traversals into one.
//   3. Skip invisible nodes and meshes outside both frustums
//   All arrays use index-based writes instead of push/pop to minimize GC pressure.
//
// computeLightDir() – Extracts the light direction from a directional light's world matrix.
//   The light's world position is treated as a direction vector (like the sun – infinitely
//   far away, only direction matters), then normalized.

import { aabbTransform, frustumContainsAABB, vec3Normalize } from '../math/index.ts'
import { Mesh } from '../scene/mesh.ts'

import type { AABB } from '../math/index.ts'
import type { DirectionalLight } from '../scene/light.ts'
import type { Node } from '../scene/node.ts'

/** Default max DPR: 1.25 on mobile (coarse pointer), 1.5 on desktop. */
export const defaultMaxDpr = (): number =>
  typeof matchMedia !== 'undefined' && matchMedia('(pointer: coarse)').matches ? 1.25 : 1.5

/**
 * Find the first directional light in the scene graph.
 * Uses a quick traversal that stops as soon as a light is found.
 */
export const findDirectionalLight = (root: Node, stack: Node[]): DirectionalLight | null => {
  let stackTop = 0
  stack[stackTop++] = root
  while (stackTop > 0) {
    const node = stack[--stackTop]!
    if (!node.visible) continue
    if (node.type === 'directionalLight') return node as DirectionalLight
    const children = node.children
    for (let i = children.length - 1; i >= 0; i--) {
      stack[stackTop++] = children[i]!
    }
  }
  return null
}

/**
 * Collect camera-visible meshes and shadow-only casters in a single traversal.
 * Uses index-based array writes to avoid push/pop GC overhead.
 * Returns the number of fully culled meshes (outside both camera and shadow frustums).
 */
export const collectMeshes = (
  root: Node,
  cameraFrustum: Float32Array,
  shadowFrustum: Float32Array | null,
  worldAABB: AABB,
  meshes: Mesh[],
  shadowMeshes: Mesh[],
  stack: Node[],
): number => {
  let meshCount = 0
  let shadowCount = 0
  let culledCount = 0
  let stackTop = 0
  stack[stackTop++] = root
  while (stackTop > 0) {
    const node = stack[--stackTop]!
    if (!node.visible) continue
    if (node.type === 'mesh') {
      const mesh = node as Mesh
      if (mesh.frustumCulled) {
        aabbTransform(worldAABB, mesh.geometry.aabb, mesh._worldMatrix)
        if (frustumContainsAABB(cameraFrustum, worldAABB)) {
          meshes[meshCount++] = mesh
        } else if (shadowFrustum && mesh.castShadow && frustumContainsAABB(shadowFrustum, worldAABB)) {
          shadowMeshes[shadowCount++] = mesh
        } else {
          culledCount++
        }
      } else {
        meshes[meshCount++] = mesh
      }
    }
    const children = node.children
    for (let i = children.length - 1; i >= 0; i--) {
      stack[stackTop++] = children[i]!
    }
  }
  meshes.length = meshCount
  shadowMeshes.length = shadowCount
  return culledCount
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
