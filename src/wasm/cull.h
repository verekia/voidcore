#ifndef CULL_H
#define CULL_H

#include "math.h"

// Extract 6 frustum planes from VP matrix (Gribb-Hartmann)
// planes: 6 × 4 floats (a, b, c, d) = 24 floats
void vc_extract_frustum_planes_impl(const f32* vp, f32* planes);

// Frustum cull entities. Returns visible count.
// Writes visible entity indices to visible_out.
u32 vc_frustum_cull_impl(u32 count, const f32* planes, const f32* bspheres,
                         const u32* flags, u32* visible_out);

#endif
