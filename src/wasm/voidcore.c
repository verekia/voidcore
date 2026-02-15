#include "math.h"
#include "arena.h"
#include "bvh.h"

extern unsigned char __heap_base;

#define MAX_ENTITIES 50000

// SoA array byte offsets (set during init)
static u32 positions_ptr;
static u32 euler_rotations_ptr;
static u32 scales_ptr;
static u32 world_matrices_ptr;
static u32 colors_ptr;
static u32 flags_ptr;
static u32 bspheres_ptr;
static u32 geometry_ids_ptr;
static u32 sort_keys_ptr;
static u32 visible_indices_ptr;

// Per-frame arena
static u32 frame_arena_base;
static u32 frame_arena_size;

// Flags
#define FLAG_DIRTY    0x01
#define FLAG_VISIBLE  0x02

// Helper to access WASM linear memory as f32/u32 arrays
#define F32_AT(byte_offset) WASM_PTR(f32, byte_offset)
#define U32_AT(byte_offset) WASM_PTR(u32, byte_offset)

__attribute__((export_name("vc_init")))
u32 vc_init(u32 memory_pages) {
  (void)memory_pages;
  u32 heap = (u32)&__heap_base;
  // Align to 16 bytes
  heap = (heap + 15) & ~15;

  u32 offset = heap;

  // Allocate SoA arrays
  positions_ptr = offset;         offset += MAX_ENTITIES * 3 * 4;  // f32[150000]
  euler_rotations_ptr = offset;   offset += MAX_ENTITIES * 3 * 4;  // f32[150000]
  scales_ptr = offset;            offset += MAX_ENTITIES * 3 * 4;  // f32[150000]
  world_matrices_ptr = offset;    offset += MAX_ENTITIES * 16 * 4; // f32[800000]
  colors_ptr = offset;            offset += MAX_ENTITIES * 4 * 4;  // f32[200000]
  flags_ptr = offset;             offset += MAX_ENTITIES * 4;      // u32[50000]
  bspheres_ptr = offset;          offset += MAX_ENTITIES * 4 * 4;  // f32[200000]
  geometry_ids_ptr = offset;      offset += MAX_ENTITIES * 4;      // u32[50000]
  sort_keys_ptr = offset;         offset += MAX_ENTITIES * 4;      // u32[50000]
  visible_indices_ptr = offset;   offset += MAX_ENTITIES * 4;      // u32[50000]

  // Initialize default scales to (1,1,1) and flags to DIRTY|VISIBLE
  f32* sc = F32_AT(scales_ptr);
  u32* fl = U32_AT(flags_ptr);
  for (u32 i = 0; i < MAX_ENTITIES; i++) {
    sc[i * 3 + 0] = 1.0f;
    sc[i * 3 + 1] = 1.0f;
    sc[i * 3 + 2] = 1.0f;
    fl[i] = FLAG_DIRTY | FLAG_VISIBLE;
  }

  // Initialize default colors to white
  f32* col = F32_AT(colors_ptr);
  for (u32 i = 0; i < MAX_ENTITIES; i++) {
    col[i * 4 + 0] = 1.0f;
    col[i * 4 + 1] = 1.0f;
    col[i * 4 + 2] = 1.0f;
    col[i * 4 + 3] = 1.0f;
  }

  // Scratch region starts after SoA arrays (JS uses this for camera/planes/raycast temp data)
  // Reserve 4KB for scratch: camera (192) + planes (96) + ray result (16) + inv matrix (64) + padding
  u32 scratch_start = offset;
  offset += 4096;

  // Set up BVH persistent arena (after scratch, before frame arena)
  u32 bvh_start = (offset + 15) & ~15;
  u32 bvh_end = 16 * 1024 * 1024;
  bvh_arena_init(bvh_start, bvh_end - bvh_start);

  // Set up per-frame arena in upper half of memory (16MB..32MB)
  frame_arena_base = 16 * 1024 * 1024;
  frame_arena_size = 16 * 1024 * 1024;
  arena_init(frame_arena_base, frame_arena_size);

  return scratch_start;
}

// --- SoA pointer getters ---

__attribute__((export_name("vc_get_positions_ptr")))
u32 vc_get_positions_ptr(void) { return positions_ptr; }

__attribute__((export_name("vc_get_euler_rotations_ptr")))
u32 vc_get_euler_rotations_ptr(void) { return euler_rotations_ptr; }

__attribute__((export_name("vc_get_scales_ptr")))
u32 vc_get_scales_ptr(void) { return scales_ptr; }

__attribute__((export_name("vc_get_world_matrices_ptr")))
u32 vc_get_world_matrices_ptr(void) { return world_matrices_ptr; }

__attribute__((export_name("vc_get_colors_ptr")))
u32 vc_get_colors_ptr(void) { return colors_ptr; }

__attribute__((export_name("vc_get_flags_ptr")))
u32 vc_get_flags_ptr(void) { return flags_ptr; }

__attribute__((export_name("vc_get_bspheres_ptr")))
u32 vc_get_bspheres_ptr(void) { return bspheres_ptr; }

__attribute__((export_name("vc_get_geometry_ids_ptr")))
u32 vc_get_geometry_ids_ptr(void) { return geometry_ids_ptr; }

__attribute__((export_name("vc_get_sort_keys_ptr")))
u32 vc_get_sort_keys_ptr(void) { return sort_keys_ptr; }

__attribute__((export_name("vc_get_visible_indices_ptr")))
u32 vc_get_visible_indices_ptr(void) { return visible_indices_ptr; }

// --- Matrix operations via byte offsets ---

__attribute__((export_name("vc_perspective")))
void vc_perspective(u32 byte_offset, f32 fov, f32 aspect, f32 near, f32 far) {
  m4_perspective(F32_AT(byte_offset), fov, aspect, near, far);
}

__attribute__((export_name("vc_look_at")))
void vc_look_at(u32 byte_offset,
                f32 ex, f32 ey, f32 ez,
                f32 tx, f32 ty, f32 tz,
                f32 ux, f32 uy, f32 uz) {
  f32 eye[3] = {ex, ey, ez};
  f32 tgt[3] = {tx, ty, tz};
  f32 up[3]  = {ux, uy, uz};
  m4_look_at(F32_AT(byte_offset), eye, tgt, up);
}

__attribute__((export_name("vc_m4_multiply")))
void vc_m4_multiply(u32 out_offset, u32 a_offset, u32 b_offset) {
  m4_multiply(F32_AT(out_offset), F32_AT(a_offset), F32_AT(b_offset));
}

__attribute__((export_name("vc_compute_world_matrices")))
u32 vc_compute_world_matrices(u32 count) {
  f32* pos = F32_AT(positions_ptr);
  f32* rot = F32_AT(euler_rotations_ptr);
  f32* sc  = F32_AT(scales_ptr);
  f32* wm  = F32_AT(world_matrices_ptr);
  u32* fl  = U32_AT(flags_ptr);
  u32 processed = 0;

  for (u32 i = 0; i < count; i++) {
    if (fl[i] & FLAG_DIRTY) {
      m4_from_euler_trs(&wm[i * 16], &pos[i * 3], &rot[i * 3], &sc[i * 3]);
      fl[i] &= ~FLAG_DIRTY;
      processed++;
    }
  }
  return processed;
}

__attribute__((export_name("vc_m4_invert")))
void vc_m4_invert_export(u32 out_offset, u32 m_offset) {
  m4_invert(F32_AT(out_offset), F32_AT(m_offset));
}

__attribute__((export_name("vc_frame_reset")))
void vc_frame_reset(void) {
  arena_reset();
}
