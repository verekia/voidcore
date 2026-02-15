#ifndef SORT_H
#define SORT_H

#include "math.h"

// Pack sort key: [geometryId:16 | entityIndex:16]
void vc_build_sort_keys_impl(u32 count, const u32* visible_indices,
                              const u32* geometry_ids, u32* sort_keys);

// Radix sort (8-bit, 4 passes for 32-bit keys)
// Sorts both keys and indices in tandem
void vc_sort_draw_calls_impl(u32 count, u32* sort_keys, u32* indices);

#endif
