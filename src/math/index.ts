// Math – Linear algebra primitives for 3D graphics (vectors, matrices, quaternions).
//
// All types are backed by typed float arrays (Float32Array or Float16Array depending on the
// engine's floatPrecision setting) for direct GPU upload with zero copying. Functions follow
// a "write into output" pattern (e.g. vec3Add(out, a, b)) to avoid allocating new arrays
// every frame, which is critical for performance in a render loop.
//
// Coordinate system: Z-up, right-handed. Forward is +Y, right is +X, up is +Z.
//
// Vec3  – 3D vector (position, direction, scale). Common: add, sub, normalize, cross, dot.
// Vec4  – 4D vector (used for homogeneous coordinates and shader uniforms).
// Quat  – Quaternion [x,y,z,w] for rotation. Avoids gimbal lock unlike Euler angles.
//         Key op: slerp (spherical interpolation for smooth rotation blending).
// Mat4  – 4x4 matrix (column-major) for transforms. Combines position + rotation + scale
//         into one matrix that the GPU multiplies with each vertex.
//         Key ops: compose (from pos/rot/scale), perspective (camera projection),
//         lookAt (camera view), multiply, invert.
// AABB  – Axis-aligned bounding box [minX,minY,minZ, maxX,maxY,maxZ] for culling/raycasting.
// Frustum – Six clip planes extracted from the view-projection matrix for frustum culling.
//
// Voidcore Math — Z-up, right-handed, FloatArray-backed, zero-allocation

import { createFloatArray, createFloatArrayFrom, type FloatArray } from '../float'

export type Vec2 = FloatArray // length 2
export type Vec3 = FloatArray // length 3
export type Vec4 = FloatArray // length 4
export type Quat = FloatArray // length 4 [x,y,z,w]
export type Mat4 = FloatArray // length 16, column-major
export type AABB = FloatArray // length 6 [minX, minY, minZ, maxX, maxY, maxZ]

// ─── Constants ────────────────────────────────────────────────────────

// Note: Object.freeze doesn't work on typed arrays in modern runtimes.
// These are module-level constants — treat as readonly by convention.
export const VEC3_ZERO: Vec3 = new Float32Array([0, 0, 0])
export const VEC3_ONE: Vec3 = new Float32Array([1, 1, 1])
export const VEC3_UP: Vec3 = new Float32Array([0, 0, 1])
export const VEC3_FORWARD: Vec3 = new Float32Array([0, 1, 0])
export const VEC3_RIGHT: Vec3 = new Float32Array([1, 0, 0])
export const QUAT_IDENTITY: Quat = new Float32Array([0, 0, 0, 1])
export const MAT4_IDENTITY: Mat4 = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

// ─── Vec3 ─────────────────────────────────────────────────────────────

export const vec3Create = (): Vec3 => createFloatArray(3)

export const vec3Set = (out: Vec3, x: number, y: number, z: number): Vec3 => {
  out[0] = x
  out[1] = y
  out[2] = z
  return out
}

export const vec3Copy = (out: Vec3, a: Vec3): Vec3 => {
  out[0] = a[0]!
  out[1] = a[1]!
  out[2] = a[2]!
  return out
}

export const vec3Add = (out: Vec3, a: Vec3, b: Vec3): Vec3 => {
  out[0] = a[0]! + b[0]!
  out[1] = a[1]! + b[1]!
  out[2] = a[2]! + b[2]!
  return out
}

export const vec3Sub = (out: Vec3, a: Vec3, b: Vec3): Vec3 => {
  out[0] = a[0]! - b[0]!
  out[1] = a[1]! - b[1]!
  out[2] = a[2]! - b[2]!
  return out
}

export const vec3Scale = (out: Vec3, a: Vec3, s: number): Vec3 => {
  out[0] = a[0]! * s
  out[1] = a[1]! * s
  out[2] = a[2]! * s
  return out
}

export const vec3Negate = (out: Vec3, a: Vec3): Vec3 => {
  out[0] = -a[0]!
  out[1] = -a[1]!
  out[2] = -a[2]!
  return out
}

export const vec3Dot = (a: Vec3, b: Vec3): number => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!

export const vec3Cross = (out: Vec3, a: Vec3, b: Vec3): Vec3 => {
  const ax = a[0]!,
    ay = a[1]!,
    az = a[2]!
  const bx = b[0]!,
    by = b[1]!,
    bz = b[2]!
  out[0] = ay * bz - az * by
  out[1] = az * bx - ax * bz
  out[2] = ax * by - ay * bx
  return out
}

export const vec3Length = (a: Vec3): number => Math.sqrt(a[0]! * a[0]! + a[1]! * a[1]! + a[2]! * a[2]!)

export const vec3LengthSq = (a: Vec3): number => a[0]! * a[0]! + a[1]! * a[1]! + a[2]! * a[2]!

export const vec3Normalize = (out: Vec3, a: Vec3): Vec3 => {
  const len = Math.sqrt(a[0]! * a[0]! + a[1]! * a[1]! + a[2]! * a[2]!)
  if (len > 1e-6) {
    const inv = 1 / len
    out[0] = a[0]! * inv
    out[1] = a[1]! * inv
    out[2] = a[2]! * inv
  } else {
    out[0] = 0
    out[1] = 0
    out[2] = 0
  }
  return out
}

export const vec3Lerp = (out: Vec3, a: Vec3, b: Vec3, t: number): Vec3 => {
  out[0] = a[0]! + (b[0]! - a[0]!) * t
  out[1] = a[1]! + (b[1]! - a[1]!) * t
  out[2] = a[2]! + (b[2]! - a[2]!) * t
  return out
}

export const vec3Min = (out: Vec3, a: Vec3, b: Vec3): Vec3 => {
  out[0] = Math.min(a[0]!, b[0]!)
  out[1] = Math.min(a[1]!, b[1]!)
  out[2] = Math.min(a[2]!, b[2]!)
  return out
}

export const vec3Max = (out: Vec3, a: Vec3, b: Vec3): Vec3 => {
  out[0] = Math.max(a[0]!, b[0]!)
  out[1] = Math.max(a[1]!, b[1]!)
  out[2] = Math.max(a[2]!, b[2]!)
  return out
}

export const vec3TransformMat4 = (out: Vec3, a: Vec3, m: Mat4): Vec3 => {
  const x = a[0]!,
    y = a[1]!,
    z = a[2]!
  const w = m[3]! * x + m[7]! * y + m[11]! * z + m[15]!
  const invW = w !== 0 ? 1 / w : 1
  out[0] = (m[0]! * x + m[4]! * y + m[8]! * z + m[12]!) * invW
  out[1] = (m[1]! * x + m[5]! * y + m[9]! * z + m[13]!) * invW
  out[2] = (m[2]! * x + m[6]! * y + m[10]! * z + m[14]!) * invW
  return out
}

export const vec3TransformQuat = (out: Vec3, a: Vec3, q: Quat): Vec3 => {
  const x = a[0]!,
    y = a[1]!,
    z = a[2]!
  const qx = q[0]!,
    qy = q[1]!,
    qz = q[2]!,
    qw = q[3]!
  const uvx = qy * z - qz * y
  const uvy = qz * x - qx * z
  const uvz = qx * y - qy * x
  const uuvx = qy * uvz - qz * uvy
  const uuvy = qz * uvx - qx * uvz
  const uuvz = qx * uvy - qy * uvx
  out[0] = x + 2 * (qw * uvx + uuvx)
  out[1] = y + 2 * (qw * uvy + uuvy)
  out[2] = z + 2 * (qw * uvz + uuvz)
  return out
}

export const vec3Distance = (a: Vec3, b: Vec3): number => {
  const dx = a[0]! - b[0]!,
    dy = a[1]! - b[1]!,
    dz = a[2]! - b[2]!
  return Math.sqrt(dx * dx + dy * dy + dz * dz)
}

export const vec3DistanceSq = (a: Vec3, b: Vec3): number => {
  const dx = a[0]! - b[0]!,
    dy = a[1]! - b[1]!,
    dz = a[2]! - b[2]!
  return dx * dx + dy * dy + dz * dz
}

// ─── Vec4 ─────────────────────────────────────────────────────────────

export const vec4Create = (): Vec4 => createFloatArray(4)

export const vec4Set = (out: Vec4, x: number, y: number, z: number, w: number): Vec4 => {
  out[0] = x
  out[1] = y
  out[2] = z
  out[3] = w
  return out
}

export const vec4TransformMat4 = (out: Vec4, a: Vec4, m: Mat4): Vec4 => {
  const x = a[0]!,
    y = a[1]!,
    z = a[2]!,
    w = a[3]!
  out[0] = m[0]! * x + m[4]! * y + m[8]! * z + m[12]! * w
  out[1] = m[1]! * x + m[5]! * y + m[9]! * z + m[13]! * w
  out[2] = m[2]! * x + m[6]! * y + m[10]! * z + m[14]! * w
  out[3] = m[3]! * x + m[7]! * y + m[11]! * z + m[15]! * w
  return out
}

// ─── Mat4 (column-major) ──────────────────────────────────────────────

export const mat4Create = (): Mat4 => {
  const out = createFloatArray(16)
  out[0] = 1
  out[5] = 1
  out[10] = 1
  out[15] = 1
  return out
}

export const mat4Identity = (out: Mat4): Mat4 => {
  out[0] = 1
  out[1] = 0
  out[2] = 0
  out[3] = 0
  out[4] = 0
  out[5] = 1
  out[6] = 0
  out[7] = 0
  out[8] = 0
  out[9] = 0
  out[10] = 1
  out[11] = 0
  out[12] = 0
  out[13] = 0
  out[14] = 0
  out[15] = 1
  return out
}

export const mat4Copy = (out: Mat4, a: Mat4): Mat4 => {
  out.set(a)
  return out
}

export const mat4Multiply = (out: Mat4, a: Mat4, b: Mat4): Mat4 => {
  const a00 = a[0]!,
    a01 = a[1]!,
    a02 = a[2]!,
    a03 = a[3]!
  const a10 = a[4]!,
    a11 = a[5]!,
    a12 = a[6]!,
    a13 = a[7]!
  const a20 = a[8]!,
    a21 = a[9]!,
    a22 = a[10]!,
    a23 = a[11]!
  const a30 = a[12]!,
    a31 = a[13]!,
    a32 = a[14]!,
    a33 = a[15]!

  let b0 = b[0]!,
    b1 = b[1]!,
    b2 = b[2]!,
    b3 = b[3]!
  out[0] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30
  out[1] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31
  out[2] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32
  out[3] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33

  b0 = b[4]!
  b1 = b[5]!
  b2 = b[6]!
  b3 = b[7]!
  out[4] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30
  out[5] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31
  out[6] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32
  out[7] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33

  b0 = b[8]!
  b1 = b[9]!
  b2 = b[10]!
  b3 = b[11]!
  out[8] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30
  out[9] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31
  out[10] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32
  out[11] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33

  b0 = b[12]!
  b1 = b[13]!
  b2 = b[14]!
  b3 = b[15]!
  out[12] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30
  out[13] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31
  out[14] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32
  out[15] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33

  return out
}

export const mat4Invert = (out: Mat4, a: Mat4): Mat4 | null => {
  const a00 = a[0]!,
    a01 = a[1]!,
    a02 = a[2]!,
    a03 = a[3]!
  const a10 = a[4]!,
    a11 = a[5]!,
    a12 = a[6]!,
    a13 = a[7]!
  const a20 = a[8]!,
    a21 = a[9]!,
    a22 = a[10]!,
    a23 = a[11]!
  const a30 = a[12]!,
    a31 = a[13]!,
    a32 = a[14]!,
    a33 = a[15]!

  const b00 = a00 * a11 - a01 * a10
  const b01 = a00 * a12 - a02 * a10
  const b02 = a00 * a13 - a03 * a10
  const b03 = a01 * a12 - a02 * a11
  const b04 = a01 * a13 - a03 * a11
  const b05 = a02 * a13 - a03 * a12
  const b06 = a20 * a31 - a21 * a30
  const b07 = a20 * a32 - a22 * a30
  const b08 = a20 * a33 - a23 * a30
  const b09 = a21 * a32 - a22 * a31
  const b10 = a21 * a33 - a23 * a31
  const b11 = a22 * a33 - a23 * a32

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06
  if (Math.abs(det) < 1e-8) return null
  det = 1 / det

  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det

  return out
}

export const mat4Transpose = (out: Mat4, a: Mat4): Mat4 => {
  if (out === a) {
    const a01 = a[1]!,
      a02 = a[2]!,
      a03 = a[3]!
    const a12 = a[6]!,
      a13 = a[7]!,
      a23 = a[11]!
    out[1] = a[4]!
    out[2] = a[8]!
    out[3] = a[12]!
    out[4] = a01
    out[6] = a[9]!
    out[7] = a[13]!
    out[8] = a02
    out[9] = a12
    out[11] = a[14]!
    out[12] = a03
    out[13] = a13
    out[14] = a23
  } else {
    out[0] = a[0]!
    out[1] = a[4]!
    out[2] = a[8]!
    out[3] = a[12]!
    out[4] = a[1]!
    out[5] = a[5]!
    out[6] = a[9]!
    out[7] = a[13]!
    out[8] = a[2]!
    out[9] = a[6]!
    out[10] = a[10]!
    out[11] = a[14]!
    out[12] = a[3]!
    out[13] = a[7]!
    out[14] = a[11]!
    out[15] = a[15]!
  }
  return out
}

export const mat4Compose = (out: Mat4, pos: Vec3, rot: Quat, scl: Vec3): Mat4 => {
  const x = rot[0]!,
    y = rot[1]!,
    z = rot[2]!,
    w = rot[3]!
  const x2 = x + x,
    y2 = y + y,
    z2 = z + z
  const xx = x * x2,
    xy = x * y2,
    xz = x * z2
  const yy = y * y2,
    yz = y * z2,
    zz = z * z2
  const wx = w * x2,
    wy = w * y2,
    wz = w * z2
  const sx = scl[0]!,
    sy = scl[1]!,
    sz = scl[2]!

  out[0] = (1 - (yy + zz)) * sx
  out[1] = (xy + wz) * sx
  out[2] = (xz - wy) * sx
  out[3] = 0
  out[4] = (xy - wz) * sy
  out[5] = (1 - (xx + zz)) * sy
  out[6] = (yz + wx) * sy
  out[7] = 0
  out[8] = (xz + wy) * sz
  out[9] = (yz - wx) * sz
  out[10] = (1 - (xx + yy)) * sz
  out[11] = 0
  out[12] = pos[0]!
  out[13] = pos[1]!
  out[14] = pos[2]!
  out[15] = 1

  return out
}

export const mat4Perspective = (
  out: Mat4,
  fovY: number,
  aspect: number,
  near: number,
  far: number,
  depth: 'zero-to-one' | 'neg-one-to-one',
): Mat4 => {
  const f = 1 / Math.tan(fovY / 2)
  out[0] = f / aspect
  out[1] = 0
  out[2] = 0
  out[3] = 0
  out[4] = 0
  out[5] = f
  out[6] = 0
  out[7] = 0
  out[8] = 0
  out[9] = 0

  if (depth === 'zero-to-one') {
    out[10] = far / (near - far)
    out[11] = -1
    out[12] = 0
    out[13] = 0
    out[14] = (near * far) / (near - far)
  } else {
    out[10] = -(far + near) / (far - near)
    out[11] = -1
    out[12] = 0
    out[13] = 0
    out[14] = -(2 * far * near) / (far - near)
  }

  out[15] = 0
  return out
}

export const mat4Ortho = (
  out: Mat4,
  left: number,
  right: number,
  bottom: number,
  top: number,
  near: number,
  far: number,
): Mat4 => {
  const lr = 1 / (left - right)
  const bt = 1 / (bottom - top)
  const nf = 1 / (near - far)
  out[0] = -2 * lr
  out[1] = 0
  out[2] = 0
  out[3] = 0
  out[4] = 0
  out[5] = -2 * bt
  out[6] = 0
  out[7] = 0
  out[8] = 0
  out[9] = 0
  out[10] = 2 * nf
  out[11] = 0
  out[12] = (left + right) * lr
  out[13] = (top + bottom) * bt
  out[14] = (far + near) * nf
  out[15] = 1
  return out
}

export const mat4OrthoZO = (
  out: Mat4,
  left: number,
  right: number,
  bottom: number,
  top: number,
  near: number,
  far: number,
): Mat4 => {
  const lr = 1 / (left - right)
  const bt = 1 / (bottom - top)
  const nf = 1 / (near - far)
  out[0] = -2 * lr
  out[1] = 0
  out[2] = 0
  out[3] = 0
  out[4] = 0
  out[5] = -2 * bt
  out[6] = 0
  out[7] = 0
  out[8] = 0
  out[9] = 0
  out[10] = nf
  out[11] = 0
  out[12] = (left + right) * lr
  out[13] = (top + bottom) * bt
  out[14] = near * nf
  out[15] = 1
  return out
}

// Z-up lookAt: eye looks at target with Z as up
export const mat4LookAt = (out: Mat4, eye: Vec3, target: Vec3, up: Vec3 = VEC3_UP): Mat4 => {
  let fx = target[0]! - eye[0]!
  let fy = target[1]! - eye[1]!
  let fz = target[2]! - eye[2]!
  let len = Math.sqrt(fx * fx + fy * fy + fz * fz)
  if (len > 1e-6) {
    len = 1 / len
    fx *= len
    fy *= len
    fz *= len
  }

  // right = cross(forward, up)
  let rx = fy * up[2]! - fz * up[1]!
  let ry = fz * up[0]! - fx * up[2]!
  let rz = fx * up[1]! - fy * up[0]!
  len = Math.sqrt(rx * rx + ry * ry + rz * rz)
  if (len > 1e-6) {
    len = 1 / len
    rx *= len
    ry *= len
    rz *= len
  }

  // cameraUp = cross(right, forward)
  const ux = ry * fz - rz * fy
  const uy = rz * fx - rx * fz
  const uz = rx * fy - ry * fx

  out[0] = rx
  out[1] = ux
  out[2] = -fx
  out[3] = 0
  out[4] = ry
  out[5] = uy
  out[6] = -fy
  out[7] = 0
  out[8] = rz
  out[9] = uz
  out[10] = -fz
  out[11] = 0
  out[12] = -(rx * eye[0]! + ry * eye[1]! + rz * eye[2]!)
  out[13] = -(ux * eye[0]! + uy * eye[1]! + uz * eye[2]!)
  out[14] = fx * eye[0]! + fy * eye[1]! + fz * eye[2]!
  out[15] = 1

  return out
}

export const mat4GetTranslation = (out: Vec3, m: Mat4): Vec3 => {
  out[0] = m[12]!
  out[1] = m[13]!
  out[2] = m[14]!
  return out
}

// ─── Quat ─────────────────────────────────────────────────────────────

export const quatCreate = (): Quat => {
  const out = createFloatArray(4)
  out[3] = 1
  return out
}

export const quatIdentity = (out: Quat): Quat => {
  out[0] = 0
  out[1] = 0
  out[2] = 0
  out[3] = 1
  return out
}

export const quatCopy = (out: Quat, a: Quat): Quat => {
  out[0] = a[0]!
  out[1] = a[1]!
  out[2] = a[2]!
  out[3] = a[3]!
  return out
}

export const quatMultiply = (out: Quat, a: Quat, b: Quat): Quat => {
  const ax = a[0]!,
    ay = a[1]!,
    az = a[2]!,
    aw = a[3]!
  const bx = b[0]!,
    by = b[1]!,
    bz = b[2]!,
    bw = b[3]!
  out[0] = ax * bw + aw * bx + ay * bz - az * by
  out[1] = ay * bw + aw * by + az * bx - ax * bz
  out[2] = az * bw + aw * bz + ax * by - ay * bx
  out[3] = aw * bw - ax * bx - ay * by - az * bz
  return out
}

export const quatNormalize = (out: Quat, a: Quat): Quat => {
  const len = Math.sqrt(a[0]! * a[0]! + a[1]! * a[1]! + a[2]! * a[2]! + a[3]! * a[3]!)
  if (len > 1e-6) {
    const inv = 1 / len
    out[0] = a[0]! * inv
    out[1] = a[1]! * inv
    out[2] = a[2]! * inv
    out[3] = a[3]! * inv
  }
  return out
}

export const quatFromAxisAngle = (out: Quat, axis: Vec3, angle: number): Quat => {
  const half = angle * 0.5
  const s = Math.sin(half)
  out[0] = axis[0]! * s
  out[1] = axis[1]! * s
  out[2] = axis[2]! * s
  out[3] = Math.cos(half)
  return out
}

export const quatSlerp = (out: Quat, a: Quat, b: Quat, t: number): Quat => {
  let ax = a[0]!,
    ay = a[1]!,
    az = a[2]!,
    aw = a[3]!
  let bx = b[0]!,
    by = b[1]!,
    bz = b[2]!,
    bw = b[3]!
  let dot = ax * bx + ay * by + az * bz + aw * bw
  if (dot < 0) {
    dot = -dot
    bx = -bx
    by = -by
    bz = -bz
    bw = -bw
  }
  if (dot > 0.9995) {
    out[0] = ax + t * (bx - ax)
    out[1] = ay + t * (by - ay)
    out[2] = az + t * (bz - az)
    out[3] = aw + t * (bw - aw)
    return quatNormalize(out, out)
  }
  const theta = Math.acos(dot)
  const sinTheta = Math.sin(theta)
  const wa = Math.sin((1 - t) * theta) / sinTheta
  const wb = Math.sin(t * theta) / sinTheta
  out[0] = ax * wa + bx * wb
  out[1] = ay * wa + by * wb
  out[2] = az * wa + bz * wb
  out[3] = aw * wa + bw * wb
  return out
}

export const quatInvert = (out: Quat, a: Quat): Quat => {
  const dot = a[0]! * a[0]! + a[1]! * a[1]! + a[2]! * a[2]! + a[3]! * a[3]!
  if (dot > 1e-6) {
    const inv = 1 / dot
    out[0] = -a[0]! * inv
    out[1] = -a[1]! * inv
    out[2] = -a[2]! * inv
    out[3] = a[3]! * inv
  }
  return out
}

export const quatConjugate = (out: Quat, a: Quat): Quat => {
  out[0] = -a[0]!
  out[1] = -a[1]!
  out[2] = -a[2]!
  out[3] = a[3]!
  return out
}

// ─── AABB ─────────────────────────────────────────────────────────────

export const aabbCreate = (): AABB =>
  createFloatArrayFrom([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity])

export const aabbFromPoints = (out: AABB, positions: Float32Array, count: number): AABB => {
  out[0] = Infinity
  out[1] = Infinity
  out[2] = Infinity
  out[3] = -Infinity
  out[4] = -Infinity
  out[5] = -Infinity
  for (let i = 0; i < count; i++) {
    const x = positions[i * 3]!
    const y = positions[i * 3 + 1]!
    const z = positions[i * 3 + 2]!
    if (x < out[0]!) out[0] = x
    if (y < out[1]!) out[1] = y
    if (z < out[2]!) out[2] = z
    if (x > out[3]!) out[3] = x
    if (y > out[4]!) out[4] = y
    if (z > out[5]!) out[5] = z
  }
  return out
}

export const aabbTransform = (out: AABB, a: AABB, m: Mat4): AABB => {
  // Arvo's method: compute transformed AABB directly from matrix columns and extents.
  // For each output axis, decompose the matrix-vector product into min/max contributions.
  // This is ~3x faster than transforming all 8 corners through mat4.
  const m0 = m[0]!,
    m1 = m[1]!,
    m2 = m[2]!
  const m4 = m[4]!,
    m5 = m[5]!,
    m6 = m[6]!
  const m8 = m[8]!,
    m9 = m[9]!,
    m10 = m[10]!
  const tx = m[12]!,
    ty = m[13]!,
    tz = m[14]!

  const minX = a[0]!,
    minY = a[1]!,
    minZ = a[2]!
  const maxX = a[3]!,
    maxY = a[4]!,
    maxZ = a[5]!

  // X-axis contribution
  const xa = m0 * minX,
    xb = m0 * maxX
  const xc = m4 * minY,
    xd = m4 * maxY
  const xe = m8 * minZ,
    xf = m8 * maxZ

  // Y-axis contribution
  const ya = m1 * minX,
    yb = m1 * maxX
  const yc = m5 * minY,
    yd = m5 * maxY
  const ye = m9 * minZ,
    yf = m9 * maxZ

  // Z-axis contribution
  const za = m2 * minX,
    zb = m2 * maxX
  const zc = m6 * minY,
    zd = m6 * maxY
  const ze = m10 * minZ,
    zf = m10 * maxZ

  out[0] = (xa < xb ? xa : xb) + (xc < xd ? xc : xd) + (xe < xf ? xe : xf) + tx
  out[3] = (xa > xb ? xa : xb) + (xc > xd ? xc : xd) + (xe > xf ? xe : xf) + tx
  out[1] = (ya < yb ? ya : yb) + (yc < yd ? yc : yd) + (ye < yf ? ye : yf) + ty
  out[4] = (ya > yb ? ya : yb) + (yc > yd ? yc : yd) + (ye > yf ? ye : yf) + ty
  out[2] = (za < zb ? za : zb) + (zc < zd ? zc : zd) + (ze < zf ? ze : zf) + tz
  out[5] = (za > zb ? za : zb) + (zc > zd ? zc : zd) + (ze > zf ? ze : zf) + tz

  return out
}

export const aabbCenter = (out: Vec3, a: AABB): Vec3 => {
  out[0] = (a[0]! + a[3]!) * 0.5
  out[1] = (a[1]! + a[4]!) * 0.5
  out[2] = (a[2]! + a[5]!) * 0.5
  return out
}

export const aabbSize = (out: Vec3, a: AABB): Vec3 => {
  out[0] = a[3]! - a[0]!
  out[1] = a[4]! - a[1]!
  out[2] = a[5]! - a[2]!
  return out
}

// ─── Frustum ──────────────────────────────────────────────────────────

// 6 planes, each [nx, ny, nz, d] = 24 floats
export const frustumFromViewProjection = (out: FloatArray, vp: Mat4): FloatArray => {
  // Gribb-Hartmann plane extraction
  const m = vp
  // Left
  out[0] = m[3]! + m[0]!
  out[1] = m[7]! + m[4]!
  out[2] = m[11]! + m[8]!
  out[3] = m[15]! + m[12]!
  // Right
  out[4] = m[3]! - m[0]!
  out[5] = m[7]! - m[4]!
  out[6] = m[11]! - m[8]!
  out[7] = m[15]! - m[12]!
  // Bottom
  out[8] = m[3]! + m[1]!
  out[9] = m[7]! + m[5]!
  out[10] = m[11]! + m[9]!
  out[11] = m[15]! + m[13]!
  // Top
  out[12] = m[3]! - m[1]!
  out[13] = m[7]! - m[5]!
  out[14] = m[11]! - m[9]!
  out[15] = m[15]! - m[13]!
  // Near
  out[16] = m[3]! + m[2]!
  out[17] = m[7]! + m[6]!
  out[18] = m[11]! + m[10]!
  out[19] = m[15]! + m[14]!
  // Far
  out[20] = m[3]! - m[2]!
  out[21] = m[7]! - m[6]!
  out[22] = m[11]! - m[10]!
  out[23] = m[15]! - m[14]!

  // Normalize planes
  for (let i = 0; i < 6; i++) {
    const o = i * 4
    const len = Math.sqrt(out[o]! * out[o]! + out[o + 1]! * out[o + 1]! + out[o + 2]! * out[o + 2]!)
    if (len > 1e-6) {
      const inv = 1 / len
      out[o] = out[o]! * inv
      out[o + 1] = out[o + 1]! * inv
      out[o + 2] = out[o + 2]! * inv
      out[o + 3] = out[o + 3]! * inv
    }
  }
  return out
}

export const frustumContainsAABB = (planes: FloatArray, aabb: AABB): boolean => {
  for (let i = 0; i < 6; i++) {
    const o = i * 4
    const nx = planes[o]!,
      ny = planes[o + 1]!,
      nz = planes[o + 2]!,
      d = planes[o + 3]!
    // P-vertex: the corner most in the direction of the normal
    const px = nx >= 0 ? aabb[3]! : aabb[0]!
    const py = ny >= 0 ? aabb[4]! : aabb[1]!
    const pz = nz >= 0 ? aabb[5]! : aabb[2]!
    if (nx * px + ny * py + nz * pz + d < 0) return false
  }
  return true
}
