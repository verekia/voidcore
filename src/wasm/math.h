#ifndef MATH_H
#define MATH_H

#include "arena.h"

#define PI 3.14159265358979323846f

// Trig (minimax polynomial, no libc)
f32 vc_sinf(f32 x);
f32 vc_cosf(f32 x);
f32 vc_tanf(f32 x);

// Native WASM intrinsics
f32 vc_sqrtf(f32 x);
f32 vc_fabsf(f32 x);
f32 vc_fminf(f32 a, f32 b);
f32 vc_fmaxf(f32 a, f32 b);

// Vec3 operations (write to out)
void v3_set(f32* out, f32 x, f32 y, f32 z);
void v3_add(f32* out, const f32* a, const f32* b);
void v3_sub(f32* out, const f32* a, const f32* b);
void v3_scale(f32* out, const f32* v, f32 s);
void v3_normalize(f32* out, const f32* v);
void v3_cross(f32* out, const f32* a, const f32* b);
f32  v3_dot(const f32* a, const f32* b);
f32  v3_length(const f32* v);

// Mat4 (column-major, 16 floats)
void m4_identity(f32* out);
void m4_multiply(f32* out, const f32* a, const f32* b);
void m4_perspective(f32* out, f32 fov, f32 aspect, f32 near, f32 far);
void m4_look_at(f32* out, const f32* eye, const f32* target, const f32* up);
void m4_invert(f32* out, const f32* m);
void m4_from_euler_trs(f32* out, const f32* pos, const f32* euler, const f32* scale);

#endif
