#include "math.h"

// --- Trig: minimax polynomial approximations ---

static f32 wrap_angle(f32 x) {
  // Reduce to [-PI, PI]
  x = x - (f32)(int)(x / (2.0f * PI)) * (2.0f * PI);
  if (x > PI) x -= 2.0f * PI;
  if (x < -PI) x += 2.0f * PI;
  return x;
}

f32 vc_sinf(f32 x) {
  x = wrap_angle(x);
  // Minimax 5th-order polynomial for sin on [-PI, PI]
  f32 x2 = x * x;
  f32 x3 = x2 * x;
  f32 x5 = x3 * x2;
  return x - x3 * 0.16666667f + x5 * 0.00833333f - x3 * x2 * x2 * 0.00019841f;
}

f32 vc_cosf(f32 x) {
  return vc_sinf(x + PI * 0.5f);
}

f32 vc_tanf(f32 x) {
  f32 c = vc_cosf(x);
  if (c > -1e-7f && c < 1e-7f) c = 1e-7f;
  return vc_sinf(x) / c;
}

// --- WASM intrinsics ---

f32 vc_sqrtf(f32 x) {
  return __builtin_sqrtf(x);
}

f32 vc_fabsf(f32 x) {
  return __builtin_fabsf(x);
}

f32 vc_fminf(f32 a, f32 b) {
  return __builtin_fminf(a, b);
}

f32 vc_fmaxf(f32 a, f32 b) {
  return __builtin_fmaxf(a, b);
}

// --- Vec3 ---

void v3_set(f32* out, f32 x, f32 y, f32 z) {
  out[0] = x; out[1] = y; out[2] = z;
}

void v3_add(f32* out, const f32* a, const f32* b) {
  out[0] = a[0] + b[0];
  out[1] = a[1] + b[1];
  out[2] = a[2] + b[2];
}

void v3_sub(f32* out, const f32* a, const f32* b) {
  out[0] = a[0] - b[0];
  out[1] = a[1] - b[1];
  out[2] = a[2] - b[2];
}

void v3_scale(f32* out, const f32* v, f32 s) {
  out[0] = v[0] * s;
  out[1] = v[1] * s;
  out[2] = v[2] * s;
}

f32 v3_dot(const f32* a, const f32* b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

f32 v3_length(const f32* v) {
  return vc_sqrtf(v3_dot(v, v));
}

void v3_normalize(f32* out, const f32* v) {
  f32 len = v3_length(v);
  if (len > 1e-8f) {
    f32 inv = 1.0f / len;
    out[0] = v[0] * inv;
    out[1] = v[1] * inv;
    out[2] = v[2] * inv;
  } else {
    out[0] = 0.0f; out[1] = 0.0f; out[2] = 0.0f;
  }
}

void v3_cross(f32* out, const f32* a, const f32* b) {
  f32 x = a[1] * b[2] - a[2] * b[1];
  f32 y = a[2] * b[0] - a[0] * b[2];
  f32 z = a[0] * b[1] - a[1] * b[0];
  out[0] = x; out[1] = y; out[2] = z;
}

// --- Mat4 (column-major) ---
// Layout: m[col*4 + row]
// Col 0: m[0..3], Col 1: m[4..7], Col 2: m[8..11], Col 3: m[12..15]

void m4_identity(f32* out) {
  for (int i = 0; i < 16; i++) out[i] = 0.0f;
  out[0] = 1.0f; out[5] = 1.0f; out[10] = 1.0f; out[15] = 1.0f;
}

void m4_multiply(f32* out, const f32* a, const f32* b) {
  f32 tmp[16];
  for (int col = 0; col < 4; col++) {
    for (int row = 0; row < 4; row++) {
      f32 sum = 0.0f;
      for (int k = 0; k < 4; k++) {
        sum += a[k * 4 + row] * b[col * 4 + k];
      }
      tmp[col * 4 + row] = sum;
    }
  }
  for (int i = 0; i < 16; i++) out[i] = tmp[i];
}

void m4_perspective(f32* out, f32 fov, f32 aspect, f32 near, f32 far) {
  f32 f = 1.0f / vc_tanf(fov * 0.5f);
  f32 range_inv = 1.0f / (near - far);
  for (int i = 0; i < 16; i++) out[i] = 0.0f;
  out[0] = f / aspect;
  out[5] = f;
  out[10] = far * range_inv;
  out[11] = -1.0f;
  out[14] = near * far * range_inv;
}

void m4_look_at(f32* out, const f32* eye, const f32* target, const f32* up) {
  f32 fwd[3], side[3], u[3];
  v3_sub(fwd, target, eye);
  v3_normalize(fwd, fwd);
  v3_cross(side, fwd, up);
  v3_normalize(side, side);
  v3_cross(u, side, fwd);

  for (int i = 0; i < 16; i++) out[i] = 0.0f;
  out[0] = side[0];   out[1] = u[0];   out[2]  = -fwd[0];
  out[4] = side[1];   out[5] = u[1];   out[6]  = -fwd[1];
  out[8] = side[2];   out[9] = u[2];   out[10] = -fwd[2];
  out[12] = -v3_dot(side, eye);
  out[13] = -v3_dot(u, eye);
  out[14] = v3_dot(fwd, eye);
  out[15] = 1.0f;
}

void m4_invert(f32* out, const f32* m) {
  f32 a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  f32 a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  f32 a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  f32 a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];

  f32 b00 = a00*a11 - a01*a10, b01 = a00*a12 - a02*a10;
  f32 b02 = a00*a13 - a03*a10, b03 = a01*a12 - a02*a11;
  f32 b04 = a01*a13 - a03*a11, b05 = a02*a13 - a03*a12;
  f32 b06 = a20*a31 - a21*a30, b07 = a20*a32 - a22*a30;
  f32 b08 = a20*a33 - a23*a30, b09 = a21*a32 - a22*a31;
  f32 b10 = a21*a33 - a23*a31, b11 = a22*a33 - a23*a32;

  f32 det = b00*b11 - b01*b10 + b02*b09 + b03*b08 - b04*b07 + b05*b06;
  if (det > -1e-8f && det < 1e-8f) { m4_identity(out); return; }
  f32 inv_det = 1.0f / det;

  out[0]  = ( a11*b11 - a12*b10 + a13*b09) * inv_det;
  out[1]  = (-a01*b11 + a02*b10 - a03*b09) * inv_det;
  out[2]  = ( a31*b05 - a32*b04 + a33*b03) * inv_det;
  out[3]  = (-a21*b05 + a22*b04 - a23*b03) * inv_det;
  out[4]  = (-a10*b11 + a12*b08 - a13*b07) * inv_det;
  out[5]  = ( a00*b11 - a02*b08 + a03*b07) * inv_det;
  out[6]  = (-a30*b05 + a32*b02 - a33*b01) * inv_det;
  out[7]  = ( a20*b05 - a22*b02 + a23*b01) * inv_det;
  out[8]  = ( a10*b10 - a11*b08 + a13*b06) * inv_det;
  out[9]  = (-a00*b10 + a01*b08 - a03*b06) * inv_det;
  out[10] = ( a30*b04 - a31*b02 + a33*b00) * inv_det;
  out[11] = (-a20*b04 + a21*b02 - a23*b00) * inv_det;
  out[12] = (-a10*b09 + a11*b07 - a12*b06) * inv_det;
  out[13] = ( a00*b09 - a01*b07 + a02*b06) * inv_det;
  out[14] = (-a30*b03 + a31*b01 - a32*b00) * inv_det;
  out[15] = ( a20*b03 - a21*b01 + a22*b00) * inv_det;
}

void m4_from_euler_trs(f32* out, const f32* pos, const f32* euler, const f32* scale) {
  // Euler ZXY order
  f32 sx = vc_sinf(euler[0]), cx = vc_cosf(euler[0]);
  f32 sy = vc_sinf(euler[1]), cy = vc_cosf(euler[1]);
  f32 sz = vc_sinf(euler[2]), cz = vc_cosf(euler[2]);

  // Rotation matrix = Z * X * Y
  f32 r00 = cy * cz + sy * sx * sz;
  f32 r01 = cx * sz;
  f32 r02 = -sy * cz + cy * sx * sz;

  f32 r10 = -cy * sz + sy * sx * cz;
  f32 r11 = cx * cz;
  f32 r12 = sy * sz + cy * sx * cz;

  f32 r20 = sy * cx;
  f32 r21 = -sx;
  f32 r22 = cy * cx;

  // Apply scale and set translation
  out[0]  = r00 * scale[0];
  out[1]  = r01 * scale[0];
  out[2]  = r02 * scale[0];
  out[3]  = 0.0f;
  out[4]  = r10 * scale[1];
  out[5]  = r11 * scale[1];
  out[6]  = r12 * scale[1];
  out[7]  = 0.0f;
  out[8]  = r20 * scale[2];
  out[9]  = r21 * scale[2];
  out[10] = r22 * scale[2];
  out[11] = 0.0f;
  out[12] = pos[0];
  out[13] = pos[1];
  out[14] = pos[2];
  out[15] = 1.0f;
}
