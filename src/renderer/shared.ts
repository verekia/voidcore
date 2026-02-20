// Shared Renderer Utilities – Logic used by both WebGPU and WebGL2 renderers.
//
// findDirectionalLight() – Quick scene graph traversal that returns the first directional
//   light found. Stops early once a light is located. Used to determine light direction
//   before computing the shadow map.
//
// findAmbientLight() – Quick scene graph traversal that returns the first ambient light
//   found. Used to determine ambient color and intensity for the frame uniforms.
//
// collectMeshes() – Walks the scene graph in a single pass to:
//   1. Collect camera-visible Mesh nodes (frustum culled against the camera frustum)
//   2. Collect shadow-only casters (meshes outside camera frustum but inside the shadow
//      frustum). This merges what used to be two separate traversals into one.
//   3. Skip invisible nodes and meshes outside both frustums
//   All arrays use index-based writes instead of push/pop to minimize GC pressure.
//
// computeLightDir() – Extracts the light direction from a directional light's world matrix.
//   The light's world position is treated as a direction vector (like the sun – infinitely
//   far away, only direction matters), then normalized.
//
// findTransparentStart() – Scans sorted keys to find the first transparent mesh.
//   Transparent meshes have bit 31 set in the sort key, placing them after all opaques.
//   Returns the index in the sorted draw order where transparent meshes begin.
//
// computeShadowMatrix() – Builds the light-space view-projection matrix for the shadow map.
//   Both renderers share the same logic; the only difference is the orthographic
//   projection function (mat4Ortho for WebGL2 [-1,1] depth, mat4OrthoZO for WebGPU [0,1]).
//   The shadow volume is a fixed ortho box centered at the world origin, oriented along
//   the light direction. Size and depth are configured on the DirectionalLight itself
//   (shadowMapSize, shadowNear, shadowFar). The camera has no involvement.

import {
  aabbTransform,
  frustumContainsAABB,
  mat4LookAt,
  mat4Multiply,
  vec3Normalize,
  vec3Set,
  vec3TransformMat4,
  VEC3_UP,
  VEC3_RIGHT,
} from '../math/index'
import { Mesh } from '../scene/mesh'

import type { AABB, Mat4, Vec3 } from '../math/index'
import type { AmbientLight, DirectionalLight } from '../scene/light'
import type { Node } from '../scene/node'
import type { SortState } from './sort'

/**
 * Find the index of the first transparent mesh in the sorted draw list.
 * Transparent meshes have bit 30 set in their sort key, so they sort after all opaques.
 * Returns meshCount if no transparent meshes exist.
 */
export const findTransparentStart = (state: SortState, meshCount: number): number => {
  const { keys, indices } = state
  for (let i = 0; i < meshCount; i++) {
    if (keys[indices[i]!]! >>> 30) return i
  }
  return meshCount
}

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
 * Find the first ambient light in the scene graph.
 * Uses a quick traversal that stops as soon as a light is found.
 */
export const findAmbientLight = (root: Node, stack: Node[]): AmbientLight | null => {
  let stackTop = 0
  stack[stackTop++] = root
  while (stackTop > 0) {
    const node = stack[--stackTop]!
    if (!node.visible) continue
    if (node.type === 'ambientLight') return node as AmbientLight
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

/** Scratch buffers for shadow computation — avoids per-call allocation. */
const _csCenter: Vec3 = new Float32Array(3) as unknown as Vec3
const _csEye: Vec3 = new Float32Array(3) as unknown as Vec3
const _csSnap: Vec3 = new Float32Array(3) as unknown as Vec3

/** Build the light-space view-projection matrix for the shadow map. */
export const computeShadowMatrix = (
  shadowVP: Mat4,
  lightDir: Vec3,
  shadowMapSize: number,
  shadowNear: number,
  shadowFar: number,
  shadowResolution: number,
  lightView: Mat4,
  lightProj: Mat4,
  orthoFn: (out: Mat4, left: number, right: number, bottom: number, top: number, near: number, far: number) => void,
): void => {
  // Center = world origin (the directional light always looks at origin)
  vec3Set(_csCenter, 0, 0, 0)

  // Eye = placed behind the scene along the light direction at shadowFar distance
  vec3Set(_csEye, lightDir[0]! * shadowFar, lightDir[1]! * shadowFar, lightDir[2]! * shadowFar)

  // Use VEC3_RIGHT as up when light is nearly vertical (Z-up system)
  const upVec = Math.abs(lightDir[2]!) > 0.99 ? VEC3_RIGHT : VEC3_UP
  mat4LookAt(lightView, _csEye, _csCenter, upVec)

  // Texel snapping: transform center to light space, snap XY to texel grid
  vec3TransformMat4(_csSnap, _csCenter, lightView)
  const texelSize = shadowMapSize / shadowResolution
  const snappedX = Math.floor(_csSnap[0]! / texelSize) * texelSize
  const snappedY = Math.floor(_csSnap[1]! / texelSize) * texelSize

  const halfSize = shadowMapSize / 2

  // Orthographic projection (depth range depends on backend)
  orthoFn(
    lightProj,
    snappedX - halfSize,
    snappedX + halfSize,
    snappedY - halfSize,
    snappedY + halfSize,
    shadowNear,
    shadowFar * 2,
  )

  // Final shadow VP = proj * view
  mat4Multiply(shadowVP, lightProj, lightView)
}
