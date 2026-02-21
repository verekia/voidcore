// DirectionalLight Shadow Volume Helper – Visualizes the orthographic projection box
// used by the shadow map for a directional light.
//
// Debugging shadow maps requires seeing what the light "sees." This helper computes the
// same shadow view-projection matrix used by the renderer, inverts it to go from NDC back
// to world space, and draws a semi-transparent box showing the shadow frustum volume.
//
// The box has 8 corners corresponding to the 8 NDC cube corners (±1, ±1, ±1) transformed
// through the inverse shadow VP matrix. Each frame the positions are updated and re-uploaded
// to the GPU. Uses BasicMaterial with transparency enabled for see-through rendering.
//
// The shadow volume is a fixed ortho box centered at the world origin, oriented along the
// light direction. Size and depth come from the light's shadowMapSize, shadowNear, shadowFar.
//
// new DirectionalLightHelper(opts?) – Creates the helper mesh. Add helper.mesh to the scene.
// helper.update(light)              – Call each frame to recompute the shadow volume box.
// helper.dispose()                  – Releases geometry resources.

import { Geometry } from '../geometry/geometry'
import { BasicMaterial } from '../materials/material'
import { mat4Create, mat4Invert, mat4Ortho, vec3Create, vec3Set, vec3TransformMat4 } from '../math/index'
import { computeLightDir, computeShadowMatrix } from '../renderer/shared'
import { Mesh } from '../scene/mesh'

import type { Vec3 } from '../math/index'
import type { DirectionalLight } from '../scene/light'

// 8 NDC cube corners (±1, ±1, ±1)
const NDC_CORNERS: [number, number, number][] = [
  [-1, -1, -1], // 0: near bottom-left
  [1, -1, -1], //  1: near bottom-right
  [1, 1, -1], //   2: near top-right
  [-1, 1, -1], //  3: near top-left
  [-1, -1, 1], //  4: far bottom-left
  [1, -1, 1], //   5: far bottom-right
  [1, 1, 1], //    6: far top-right
  [-1, 1, 1], //   7: far top-left
]

// 24 vertices (4 per face) for proper per-face normals
const VERTEX_COUNT = 24

// Face definitions: [corner0, corner1, corner2, corner3] for each face
const FACES: [number, number, number, number][] = [
  [0, 1, 2, 3], // Near  (z = -1)
  [4, 7, 6, 5], // Far   (z = +1)  — reversed winding for outward normal
  [0, 4, 5, 1], // Bottom (y = -1)
  [3, 2, 6, 7], // Top    (y = +1)
  [0, 3, 7, 4], // Left   (x = -1)
  [1, 5, 6, 2], // Right  (x = +1)
]

// Face normals in NDC space (axis-aligned)
const FACE_NORMALS: [number, number, number][] = [
  [0, 0, -1], // Near
  [0, 0, 1], //  Far
  [0, -1, 0], // Bottom
  [0, 1, 0], //  Top
  [-1, 0, 0], // Left
  [1, 0, 0], //  Right
]

// 6 faces × 2 triangles × 3 = 36 indices (with 24-vertex layout)
const FACE_INDICES = new Uint16Array(36)
for (let f = 0; f < 6; f++) {
  const o = f * 4
  const i = f * 6
  FACE_INDICES[i] = o
  FACE_INDICES[i + 1] = o + 1
  FACE_INDICES[i + 2] = o + 2
  FACE_INDICES[i + 3] = o
  FACE_INDICES[i + 4] = o + 2
  FACE_INDICES[i + 5] = o + 3
}

export class DirectionalLightHelper {
  mesh: Mesh

  private _geometry: Geometry

  // Pre-allocated scratch buffers (zero allocation in update)
  private _shadowVP = mat4Create()
  private _invShadowVP = mat4Create()
  private _lightView = mat4Create()
  private _lightProj = mat4Create()
  private _lightDir: Vec3 = vec3Create()
  private _tempVec3: Vec3 = vec3Create()
  private _corners: Vec3[] = Array.from({ length: 8 }, () => vec3Create())

  constructor(
    options: {
      color?: [number, number, number]
      opacity?: number
    } = {},
  ) {
    const color = options.color ?? [1, 1, 0]
    const opacity = options.opacity ?? 0.15

    // Create geometry with pre-allocated buffers (24 vertices, 36 indices)
    const positions = new Float32Array(VERTEX_COUNT * 3)
    const normals = new Float32Array(VERTEX_COUNT * 3)

    this._geometry = new Geometry({ positions, normals, indices: FACE_INDICES })

    const material = new BasicMaterial({ color, opacity, transparent: true })
    this.mesh = new Mesh(this._geometry, material)
    this.mesh.frustumCulled = false
    this.mesh.castShadow = false
  }

  update(light: DirectionalLight): void {
    // 1. Compute shadow VP matrix using the same function as the renderer (GL convention)
    computeLightDir(this._lightDir, this._tempVec3, light)
    computeShadowMatrix(
      this._shadowVP,
      this._lightDir,
      light.shadowMapSize,
      light.shadowNear,
      light.shadowFar,
      2048,
      this._lightView,
      this._lightProj,
      mat4Ortho,
    )

    // 2. Invert shadow VP to map NDC -> world space
    if (!mat4Invert(this._invShadowVP, this._shadowVP)) return

    // 3. Transform 8 NDC corners to world space
    for (let i = 0; i < 8; i++) {
      const ndc = NDC_CORNERS[i]!
      vec3Set(this._corners[i]!, ndc[0], ndc[1], ndc[2])
      vec3TransformMat4(this._corners[i]!, this._corners[i]!, this._invShadowVP)
    }

    // 4. Write 24 vertex positions (4 per face) from the 8 world-space corners
    const positions = this._geometry.positions
    const normals = this._geometry.normals

    for (let f = 0; f < 6; f++) {
      const face = FACES[f]!
      const normal = FACE_NORMALS[f]!

      for (let v = 0; v < 4; v++) {
        const corner = this._corners[face[v]!]!
        const p = (f * 4 + v) * 3
        positions[p] = corner[0]!
        positions[p + 1] = corner[1]!
        positions[p + 2] = corner[2]!
        normals[p] = normal[0]
        normals[p + 1] = normal[1]
        normals[p + 2] = normal[2]
      }
    }

    this._geometry.needsUpdate = true
  }

  dispose(): void {
    this._geometry.dispose()
  }
}
