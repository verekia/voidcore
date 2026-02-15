#include "sort.h"
#include "arena.h"

void vc_build_sort_keys_impl(u32 count, const u32* visible_indices,
                              const u32* geometry_ids, u32* sort_keys) {
  for (u32 i = 0; i < count; i++) {
    u32 idx = visible_indices[i];
    u32 geoId = geometry_ids[idx];
    // Pack: geometry ID in upper 16 bits, entity index in lower 16
    sort_keys[i] = (geoId << 16) | (idx & 0xFFFF);
  }
}

void vc_sort_draw_calls_impl(u32 count, u32* sort_keys, u32* indices) {
  if (count <= 1) return;

  // Allocate temp buffers from arena
  u32 temp_keys_offset = arena_alloc(count * 4, 4);
  u32 temp_indices_offset = arena_alloc(count * 4, 4);
  if (temp_keys_offset == 0 || temp_indices_offset == 0) return;

  u32* temp_keys = (u32*)((unsigned char*)0 + temp_keys_offset);
  u32* temp_indices = (u32*)((unsigned char*)0 + temp_indices_offset);

  // 4-pass radix sort (8 bits per pass)
  u32* src_keys = sort_keys;
  u32* src_idx = indices;
  u32* dst_keys = temp_keys;
  u32* dst_idx = temp_indices;

  for (int pass = 0; pass < 4; pass++) {
    int shift = pass * 8;

    // Count occurrences
    u32 counts[256];
    for (int i = 0; i < 256; i++) counts[i] = 0;

    for (u32 i = 0; i < count; i++) {
      u32 byte = (src_keys[i] >> shift) & 0xFF;
      counts[byte]++;
    }

    // Prefix sum
    u32 total = 0;
    for (int i = 0; i < 256; i++) {
      u32 c = counts[i];
      counts[i] = total;
      total += c;
    }

    // Scatter
    for (u32 i = 0; i < count; i++) {
      u32 byte = (src_keys[i] >> shift) & 0xFF;
      u32 dst_pos = counts[byte]++;
      dst_keys[dst_pos] = src_keys[i];
      dst_idx[dst_pos] = src_idx[i];
    }

    // Swap src/dst
    u32* tmp;
    tmp = src_keys; src_keys = dst_keys; dst_keys = tmp;
    tmp = src_idx; src_idx = dst_idx; dst_idx = tmp;
  }

  // After 4 passes (even number), result is back in original buffers
  // (src_keys == sort_keys, src_idx == indices) — no copy needed
}

// WASM exports
__attribute__((export_name("vc_build_sort_keys")))
void vc_build_sort_keys(u32 count, u32 visible_offset, u32 geo_ids_offset, u32 keys_out_offset) {
  vc_build_sort_keys_impl(
    count,
    (const u32*)((unsigned char*)0 + visible_offset),
    (const u32*)((unsigned char*)0 + geo_ids_offset),
    (u32*)((unsigned char*)0 + keys_out_offset)
  );
}

__attribute__((export_name("vc_sort_draw_calls")))
void vc_sort_draw_calls(u32 count, u32 keys_offset, u32 indices_offset) {
  vc_sort_draw_calls_impl(
    count,
    (u32*)((unsigned char*)0 + keys_offset),
    (u32*)((unsigned char*)0 + indices_offset)
  );
}
