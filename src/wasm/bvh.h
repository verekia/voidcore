#ifndef BVH_H
#define BVH_H

#include "math.h"

// BVH persistent memory allocator (separate from per-frame arena)
void bvh_arena_init(u32 base, u32 size);
u32 bvh_arena_alloc(u32 bytes, u32 align);
void bvh_arena_reset(void);

#endif
