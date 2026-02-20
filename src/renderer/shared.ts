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
//
// findTransparentStart() – Scans sorted keys to find the first transparent mesh.
//   Transparent meshes have bit 31 set in the sort key, placing them after all opaques.
//   Returns the index in the sorted draw order where transparent meshes begin.
//
// computeCascadeSplits() – Computes logarithmic/linear blend split distances for CSM.
// computeCascadeMatrix() – Builds the light-space view-projection matrix for one cascade.
//   Both renderers share the same cascade logic; the only difference is the orthographic
//   projection function (mat4Ortho for WebGL2 [-1,1] depth, mat4OrthoZO for WebGPU [0,1]).

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
} from '../math/index.ts'
import { Mesh } from '../scene/mesh.ts'

import type { AABB, Mat4, Vec3 } from '../math/index.ts'
import type { PerspectiveCamera } from '../scene/camera.ts'
import type { DirectionalLight } from '../scene/light.ts'
import type { Node } from '../scene/node.ts'
import type { SortState } from './sort.ts'

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

export const NUM_CASCADES = 3

/** Compute logarithmic/linear blend split distances for cascaded shadow maps. */
export const computeCascadeSplits = (cascadeSplits: Float32Array, camera: PerspectiveCamera, lambda: number): void => {
  const near = camera.near
  const far = camera.far
  for (let i = 0; i < NUM_CASCADES; i++) {
    const p = (i + 1) / NUM_CASCADES
    const log = near * Math.pow(far / near, p)
    const linear = near + (far - near) * p
    cascadeSplits[i] = lambda * log + (1 - lambda) * linear
  }
}

/** Scratch buffers for cascade computation — avoids per-call allocation. */
const _csCenter: Vec3 = new Float32Array(3) as unknown as Vec3
const _csCorner: Vec3 = new Float32Array(3) as unknown as Vec3
const _csCorners = new Float32Array(24)

/** Build the light-space view-projection matrix for one cascade. */
export const computeCascadeMatrix = (
  cascadeVP: Mat4,
  camera: PerspectiveCamera,
  lightDir: Vec3,
  nearDist: number,
  farDist: number,
  shadowBackExtend: number,
  shadowResolution: number,
  lightView: Mat4,
  lightProj: Mat4,
  orthoFn: (out: Mat4, left: number, right: number, bottom: number, top: number, near: number, far: number) => void,
): void => {
  // Extract camera basis from _viewMatrix (correct even when orbit controls
  // bypass the node's rotation quaternion). View matrix rows = camera axes.
  const V = camera._viewMatrix
  const rx = V[0]!,
    ry = V[4]!,
    rz = V[8]! // right (row 0)
  const ux = V[1]!,
    uy = V[5]!,
    uz = V[9]! // up (row 1)
  const fx = -V[2]!,
    fy = -V[6]!,
    fz = -V[10]! // forward = -row2 (camera looks along -Z in view space)
  const px = camera.position[0]!,
    py = camera.position[1]!,
    pz = camera.position[2]!

  const fovY = camera.fov * (Math.PI / 180)
  const aspect = camera.aspect

  const nearH = Math.tan(fovY / 2) * nearDist
  const nearW = nearH * aspect
  const farH = Math.tan(fovY / 2) * farDist
  const farW = farH * aspect

  // 8 frustum corners written into pre-allocated Float32Array (zero allocation)
  // Order: 4 near + 4 far, each bottom-left, bottom-right, top-right, top-left
  const corners = _csCorners
  // Near
  corners[0] = px + fx * nearDist - rx * nearW - ux * nearH
  corners[1] = py + fy * nearDist - ry * nearW - uy * nearH
  corners[2] = pz + fz * nearDist - rz * nearW - uz * nearH
  corners[3] = px + fx * nearDist + rx * nearW - ux * nearH
  corners[4] = py + fy * nearDist + ry * nearW - uy * nearH
  corners[5] = pz + fz * nearDist + rz * nearW - uz * nearH
  corners[6] = px + fx * nearDist + rx * nearW + ux * nearH
  corners[7] = py + fy * nearDist + ry * nearW + uy * nearH
  corners[8] = pz + fz * nearDist + rz * nearW + uz * nearH
  corners[9] = px + fx * nearDist - rx * nearW + ux * nearH
  corners[10] = py + fy * nearDist - ry * nearW + uy * nearH
  corners[11] = pz + fz * nearDist - rz * nearW + uz * nearH
  // Far
  corners[12] = px + fx * farDist - rx * farW - ux * farH
  corners[13] = py + fy * farDist - ry * farW - uy * farH
  corners[14] = pz + fz * farDist - rz * farW - uz * farH
  corners[15] = px + fx * farDist + rx * farW - ux * farH
  corners[16] = py + fy * farDist + ry * farW - uy * farH
  corners[17] = pz + fz * farDist + rz * farW - uz * farH
  corners[18] = px + fx * farDist + rx * farW + ux * farH
  corners[19] = py + fy * farDist + ry * farW + uy * farH
  corners[20] = pz + fz * farDist + rz * farW + uz * farH
  corners[21] = px + fx * farDist - rx * farW + ux * farH
  corners[22] = py + fy * farDist - ry * farW + uy * farH
  corners[23] = pz + fz * farDist - rz * farW + uz * farH

  // Frustum center
  let cx = 0,
    cy = 0,
    cz = 0
  for (let i = 0; i < 24; i += 3) {
    cx += corners[i]!
    cy += corners[i + 1]!
    cz += corners[i + 2]!
  }
  cx /= 8
  cy /= 8
  cz /= 8

  // Light view matrix
  vec3Set(_csCenter, cx, cy, cz)
  const offset = shadowBackExtend + farDist
  vec3Set(_csCorner, cx + lightDir[0]! * offset, cy + lightDir[1]! * offset, cz + lightDir[2]! * offset)

  // Use VEC3_RIGHT as up when light is nearly vertical (Z-up system)
  const upVec = Math.abs(lightDir[2]!) > 0.99 ? VEC3_RIGHT : VEC3_UP
  mat4LookAt(lightView, _csCorner, _csCenter, upVec)

  // Transform corners to light space, compute tight AABB
  let minX = Infinity,
    minY = Infinity,
    minZ = Infinity
  let maxX = -Infinity,
    maxY = -Infinity,
    maxZ = -Infinity

  for (let i = 0; i < 24; i += 3) {
    vec3Set(_csCorner, corners[i]!, corners[i + 1]!, corners[i + 2]!)
    vec3TransformMat4(_csCorner, _csCorner, lightView)
    const lx = _csCorner[0]!
    const ly = _csCorner[1]!
    const lz = _csCorner[2]!
    if (lx < minX) minX = lx
    if (lx > maxX) maxX = lx
    if (ly < minY) minY = ly
    if (ly > maxY) maxY = ly
    if (lz < minZ) minZ = lz
    if (lz > maxZ) maxZ = lz
  }

  // Back-extend minZ to catch casters behind the frustum
  minZ -= shadowBackExtend

  // Texel snapping to prevent shadow edge shimmering
  const texelSizeX = (maxX - minX) / shadowResolution
  const texelSizeY = (maxY - minY) / shadowResolution
  minX = Math.floor(minX / texelSizeX) * texelSizeX
  maxX = Math.ceil(maxX / texelSizeX) * texelSizeX
  minY = Math.floor(minY / texelSizeY) * texelSizeY
  maxY = Math.ceil(maxY / texelSizeY) * texelSizeY

  // Orthographic projection (depth range depends on backend)
  orthoFn(lightProj, minX, maxX, minY, maxY, -maxZ, -minZ)

  // Final cascade VP = proj * view
  mat4Multiply(cascadeVP, lightProj, lightView)
}
