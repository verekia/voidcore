#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "math.h"

// Transform computation is handled in voidcore.c:
// - SoA storage allocated in vc_init()
// - vc_compute_world_matrices() loops entities and calls m4_from_euler_trs()
// - m4_from_euler_trs() defined in math.c

#endif
