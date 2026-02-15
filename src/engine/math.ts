// Zero-allocation math for skeletal animation.
// All functions write to `out` at offset `o`.

export function v3Lerp(
  out: Float32Array,
  o: number,
  a: Float32Array,
  ao: number,
  b: Float32Array,
  bo: number,
  t: number,
): void {
  out[o] = a[ao]! + (b[bo]! - a[ao]!) * t
  out[o + 1] = a[ao + 1]! + (b[bo + 1]! - a[ao + 1]!) * t
  out[o + 2] = a[ao + 2]! + (b[bo + 2]! - a[ao + 2]!) * t
}

export function quatSlerp(
  out: Float32Array,
  o: number,
  a: Float32Array,
  ao: number,
  b: Float32Array,
  bo: number,
  t: number,
): void {
  const ax = a[ao]!,
    ay = a[ao + 1]!,
    az = a[ao + 2]!,
    aw = a[ao + 3]!
  let bx = b[bo]!,
    by = b[bo + 1]!,
    bz = b[bo + 2]!,
    bw = b[bo + 3]!

  let dot = ax * bx + ay * by + az * bz + aw * bw
  if (dot < 0) {
    bx = -bx
    by = -by
    bz = -bz
    bw = -bw
    dot = -dot
  }

  if (dot > 0.9995) {
    out[o] = ax + (bx - ax) * t
    out[o + 1] = ay + (by - ay) * t
    out[o + 2] = az + (bz - az) * t
    out[o + 3] = aw + (bw - aw) * t
  } else {
    const theta = Math.acos(dot)
    const sinTheta = Math.sin(theta)
    const wa = Math.sin((1 - t) * theta) / sinTheta
    const wb = Math.sin(t * theta) / sinTheta
    out[o] = ax * wa + bx * wb
    out[o + 1] = ay * wa + by * wb
    out[o + 2] = az * wa + bz * wb
    out[o + 3] = aw * wa + bw * wb
  }

  // Normalize
  const nx = out[o]!,
    ny = out[o + 1]!,
    nz = out[o + 2]!,
    nw = out[o + 3]!
  const len = Math.sqrt(nx * nx + ny * ny + nz * nz + nw * nw) || 1
  out[o] = nx / len
  out[o + 1] = ny / len
  out[o + 2] = nz / len
  out[o + 3] = nw / len
}

export function m4FromQuatTRS(
  out: Float32Array,
  o: number,
  pos: Float32Array,
  po: number,
  quat: Float32Array,
  qo: number,
  scl: Float32Array,
  so: number,
): void {
  const qx = quat[qo]!,
    qy = quat[qo + 1]!,
    qz = quat[qo + 2]!,
    qw = quat[qo + 3]!
  const sx = scl[so]!,
    sy = scl[so + 1]!,
    sz = scl[so + 2]!

  const x2 = qx + qx,
    y2 = qy + qy,
    z2 = qz + qz
  const xx = qx * x2,
    xy = qx * y2,
    xz = qx * z2
  const yy = qy * y2,
    yz = qy * z2,
    zz = qz * z2
  const wx = qw * x2,
    wy = qw * y2,
    wz = qw * z2

  out[o] = (1 - (yy + zz)) * sx
  out[o + 1] = (xy + wz) * sx
  out[o + 2] = (xz - wy) * sx
  out[o + 3] = 0
  out[o + 4] = (xy - wz) * sy
  out[o + 5] = (1 - (xx + zz)) * sy
  out[o + 6] = (yz + wx) * sy
  out[o + 7] = 0
  out[o + 8] = (xz + wy) * sz
  out[o + 9] = (yz - wx) * sz
  out[o + 10] = (1 - (xx + yy)) * sz
  out[o + 11] = 0
  out[o + 12] = pos[po]!
  out[o + 13] = pos[po + 1]!
  out[o + 14] = pos[po + 2]!
  out[o + 15] = 1
}

export function m4Multiply(
  out: Float32Array,
  o: number,
  a: Float32Array,
  ao: number,
  b: Float32Array,
  bo: number,
): void {
  const a00 = a[ao]!,
    a01 = a[ao + 1]!,
    a02 = a[ao + 2]!,
    a03 = a[ao + 3]!
  const a10 = a[ao + 4]!,
    a11 = a[ao + 5]!,
    a12 = a[ao + 6]!,
    a13 = a[ao + 7]!
  const a20 = a[ao + 8]!,
    a21 = a[ao + 9]!,
    a22 = a[ao + 10]!,
    a23 = a[ao + 11]!
  const a30 = a[ao + 12]!,
    a31 = a[ao + 13]!,
    a32 = a[ao + 14]!,
    a33 = a[ao + 15]!

  let b0 = b[bo]!,
    b1 = b[bo + 1]!,
    b2 = b[bo + 2]!,
    b3 = b[bo + 3]!
  out[o] = a00 * b0 + a10 * b1 + a20 * b2 + a30 * b3
  out[o + 1] = a01 * b0 + a11 * b1 + a21 * b2 + a31 * b3
  out[o + 2] = a02 * b0 + a12 * b1 + a22 * b2 + a32 * b3
  out[o + 3] = a03 * b0 + a13 * b1 + a23 * b2 + a33 * b3

  b0 = b[bo + 4]!
  b1 = b[bo + 5]!
  b2 = b[bo + 6]!
  b3 = b[bo + 7]!
  out[o + 4] = a00 * b0 + a10 * b1 + a20 * b2 + a30 * b3
  out[o + 5] = a01 * b0 + a11 * b1 + a21 * b2 + a31 * b3
  out[o + 6] = a02 * b0 + a12 * b1 + a22 * b2 + a32 * b3
  out[o + 7] = a03 * b0 + a13 * b1 + a23 * b2 + a33 * b3

  b0 = b[bo + 8]!
  b1 = b[bo + 9]!
  b2 = b[bo + 10]!
  b3 = b[bo + 11]!
  out[o + 8] = a00 * b0 + a10 * b1 + a20 * b2 + a30 * b3
  out[o + 9] = a01 * b0 + a11 * b1 + a21 * b2 + a31 * b3
  out[o + 10] = a02 * b0 + a12 * b1 + a22 * b2 + a32 * b3
  out[o + 11] = a03 * b0 + a13 * b1 + a23 * b2 + a33 * b3

  b0 = b[bo + 12]!
  b1 = b[bo + 13]!
  b2 = b[bo + 14]!
  b3 = b[bo + 15]!
  out[o + 12] = a00 * b0 + a10 * b1 + a20 * b2 + a30 * b3
  out[o + 13] = a01 * b0 + a11 * b1 + a21 * b2 + a31 * b3
  out[o + 14] = a02 * b0 + a12 * b1 + a22 * b2 + a32 * b3
  out[o + 15] = a03 * b0 + a13 * b1 + a23 * b2 + a33 * b3
}
