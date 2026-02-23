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
//   1. Distance-cull meshes beyond their maxDistance from the camera (squared distance,
//      no sqrt). Distance-culled meshes are excluded from both rendering and shadow casting.
//   2. Collect camera-visible Mesh and Sprite nodes (frustum culled against the camera frustum)
//   3. Collect shadow-only casters (meshes outside camera frustum but inside the shadow
//      frustum). This merges what used to be two separate traversals into one.
//   4. Skip invisible nodes and meshes outside both frustums
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
//
// computeBillboardMatrix() – Computes a camera-facing world matrix for sprites. Extracts
//   position and scale from the sprite's scene-graph world matrix, then builds a new matrix
//   using the camera's right/up vectors as the sprite's local X/Y axes. Supports optional
//   2D rotation (around the view axis) and constant-size mode (sizeAttenuation=false).

import {
  aabbTransform,
  frustumContainsAABB,
  mat4Create,
  mat4LookAt,
  mat4Multiply,
  vec3Create,
  vec3Normalize,
  vec3Set,
  vec3TransformMat4,
  VEC3_UP,
  VEC3_RIGHT,
} from '../math/index'
import { Mesh } from '../scene/mesh'

import type { AABB, Mat4, Vec3 } from '../math/index'
import type { PerspectiveCamera } from '../scene/camera'
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
 * Meshes with maxDistance > 0 are distance-culled using squared distance (no sqrt).
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
  cameraPosition: Vec3,
): number => {
  let meshCount = 0
  let shadowCount = 0
  let culledCount = 0
  let stackTop = 0
  stack[stackTop++] = root
  while (stackTop > 0) {
    const node = stack[--stackTop]!
    if (!node.visible) continue
    if (node.type === 'mesh' || node.type === 'sprite') {
      const mesh = node as Mesh

      // Distance culling: skip mesh entirely if beyond maxDistance
      if (mesh.maxDistance > 0) {
        const dx = mesh._worldMatrix[12]! - cameraPosition[0]!
        const dy = mesh._worldMatrix[13]! - cameraPosition[1]!
        const dz = mesh._worldMatrix[14]! - cameraPosition[2]!
        if (dx * dx + dy * dy + dz * dz > mesh.maxDistance * mesh.maxDistance) {
          culledCount++
          // Still process children below — distance culling is per-mesh
          const children = node.children
          for (let i = children.length - 1; i >= 0; i--) {
            stack[stackTop++] = children[i]!
          }
          continue
        }
      }

      if (!mesh.geometry || !mesh.material) {
        culledCount++
      } else if (mesh.frustumCulled) {
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
export const computeLightDir = (lightDir: Vec3, tempVec3: Vec3, dirLight: DirectionalLight | null): void => {
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
const _csCenter: Vec3 = vec3Create()
const _csEye: Vec3 = vec3Create()
const _csSnap: Vec3 = vec3Create()

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

/**
 * Scratch buffer for billboard matrix computation — avoids per-call allocation.
 * Both renderers use this via computeBillboardMatrix() in the batch fill phase.
 */
const _bbMatrix: Mat4 = mat4Create()

/**
 * Compute a billboard world matrix that makes a sprite always face the camera.
 *
 * Extracts the sprite's world position and scale from its scene-graph world matrix,
 * then constructs a new matrix whose orientation matches the camera's view axes
 * (right, up, backward). The result is written into the output buffer.
 *
 * Supports optional 2D rotation (around the view axis) and size attenuation toggle
 * (when false, the sprite maintains constant screen size regardless of distance).
 */
export const computeBillboardMatrix = (
  out: Float32Array,
  off: number,
  worldMatrix: Mat4,
  camera: PerspectiveCamera,
  rotation: number,
  sizeAttenuation: boolean,
): void => {
  const vm = camera._viewMatrix

  // Extract position from world matrix
  const tx = worldMatrix[12]!
  const ty = worldMatrix[13]!
  const tz = worldMatrix[14]!

  // Extract scale from world matrix columns
  let sx = Math.sqrt(
    worldMatrix[0]! * worldMatrix[0]! + worldMatrix[1]! * worldMatrix[1]! + worldMatrix[2]! * worldMatrix[2]!,
  )
  let sy = Math.sqrt(
    worldMatrix[4]! * worldMatrix[4]! + worldMatrix[5]! * worldMatrix[5]! + worldMatrix[6]! * worldMatrix[6]!,
  )

  // sizeAttenuation=false: scale by distance so sprite stays constant screen size
  if (!sizeAttenuation) {
    const dx = tx - camera.position[0]!
    const dy = ty - camera.position[1]!
    const dz = tz - camera.position[2]!
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)
    sx *= dist
    sy *= dist
  }

  // Camera basis vectors from view matrix rows (transpose of the rotation part)
  //   Right:    vm[0], vm[4], vm[8]
  //   Up:       vm[1], vm[5], vm[9]
  //   Backward: vm[2], vm[6], vm[10]
  let rx = vm[0]!,
    ry = vm[4]!,
    rz = vm[8]!
  let ux = vm[1]!,
    uy = vm[5]!,
    uz = vm[9]!

  // Apply 2D rotation around the view axis (rotate right/up vectors)
  if (rotation !== 0) {
    const cos = Math.cos(rotation)
    const sin = Math.sin(rotation)
    const rx2 = cos * rx + sin * ux
    const ry2 = cos * ry + sin * uy
    const rz2 = cos * rz + sin * uz
    ux = -sin * rx + cos * ux
    uy = -sin * ry + cos * uy
    uz = -sin * rz + cos * uz
    rx = rx2
    ry = ry2
    rz = rz2
  }

  // Build billboard matrix (column-major):
  //   Column 0: camera right * scaleX
  //   Column 1: camera up * scaleY
  //   Column 2: camera backward (unit length, Z axis irrelevant for flat plane)
  //   Column 3: world position
  out[off] = rx * sx
  out[off + 1] = ry * sx
  out[off + 2] = rz * sx
  out[off + 3] = 0
  out[off + 4] = ux * sy
  out[off + 5] = uy * sy
  out[off + 6] = uz * sy
  out[off + 7] = 0
  out[off + 8] = vm[2]!
  out[off + 9] = vm[6]!
  out[off + 10] = vm[10]!
  out[off + 11] = 0
  out[off + 12] = tx
  out[off + 13] = ty
  out[off + 14] = tz
  out[off + 15] = 1
}
