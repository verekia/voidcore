#include "cull.h"

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#endif

#define FLAG_VISIBLE 0x02

// Gribb-Hartmann frustum plane extraction from column-major VP matrix
void vc_extract_frustum_planes_impl(const f32* m, f32* planes) {
  // Left:   row3 + row0
  planes[0]  = m[3]  + m[0];
  planes[1]  = m[7]  + m[4];
  planes[2]  = m[11] + m[8];
  planes[3]  = m[15] + m[12];

  // Right:  row3 - row0
  planes[4]  = m[3]  - m[0];
  planes[5]  = m[7]  - m[4];
  planes[6]  = m[11] - m[8];
  planes[7]  = m[15] - m[12];

  // Bottom: row3 + row1
  planes[8]  = m[3]  + m[1];
  planes[9]  = m[7]  + m[5];
  planes[10] = m[11] + m[9];
  planes[11] = m[15] + m[13];

  // Top:    row3 - row1
  planes[12] = m[3]  - m[1];
  planes[13] = m[7]  - m[5];
  planes[14] = m[11] - m[9];
  planes[15] = m[15] - m[13];

  // Near:   row3 + row2
  planes[16] = m[3]  + m[2];
  planes[17] = m[7]  + m[6];
  planes[18] = m[11] + m[10];
  planes[19] = m[15] + m[14];

  // Far:    row3 - row2
  planes[20] = m[3]  - m[2];
  planes[21] = m[7]  - m[6];
  planes[22] = m[11] - m[10];
  planes[23] = m[15] - m[14];

  // Normalize each plane
  for (int i = 0; i < 6; i++) {
    f32* p = &planes[i * 4];
    f32 len = vc_sqrtf(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
    if (len > 1e-8f) {
      f32 inv = 1.0f / len;
      p[0] *= inv;
      p[1] *= inv;
      p[2] *= inv;
      p[3] *= inv;
    }
  }
}

u32 vc_frustum_cull_impl(u32 count, const f32* planes, const f32* bspheres,
                         const u32* flags, u32* visible_out) {
  u32 visible_count = 0;

  for (u32 i = 0; i < count; i++) {
    if (!(flags[i] & FLAG_VISIBLE)) continue;

    const f32* sphere = &bspheres[i * 4];
    f32 cx = sphere[0], cy = sphere[1], cz = sphere[2], radius = sphere[3];

    int inside = 1;
    for (int p = 0; p < 6; p++) {
      const f32* plane = &planes[p * 4];
      f32 dist = plane[0] * cx + plane[1] * cy + plane[2] * cz + plane[3];
      if (dist < -radius) {
        inside = 0;
        break;
      }
    }

    if (inside) {
      visible_out[visible_count++] = i;
    }
  }

  return visible_count;
}

// WASM exports
__attribute__((export_name("vc_extract_frustum_planes")))
void vc_extract_frustum_planes(u32 vp_offset, u32 planes_offset) {
  vc_extract_frustum_planes_impl(
    WASM_PTR(const f32, vp_offset),
    WASM_PTR(f32, planes_offset)
  );
}

__attribute__((export_name("vc_frustum_cull")))
u32 vc_frustum_cull(u32 count, u32 planes_offset, u32 bspheres_offset,
                    u32 flags_offset, u32 visible_out_offset) {
  return vc_frustum_cull_impl(
    count,
    WASM_PTR(const f32, planes_offset),
    WASM_PTR(const f32, bspheres_offset),
    WASM_PTR(const u32, flags_offset),
    WASM_PTR(u32, visible_out_offset)
  );
}
