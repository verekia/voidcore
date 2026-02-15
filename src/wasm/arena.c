#include "arena.h"

static u32 arena_base;
static u32 arena_offset;
static u32 arena_size;

void arena_init(u32 base, u32 size) {
  arena_base = base;
  arena_offset = 0;
  arena_size = size;
}

u32 arena_alloc(u32 bytes, u32 align) {
  u32 current = arena_base + arena_offset;
  u32 aligned = (current + align - 1) & ~(align - 1);
  u32 next = aligned + bytes;
  if (next - arena_base > arena_size) return 0;
  arena_offset = next - arena_base;
  return aligned;
}

void arena_reset(void) {
  arena_offset = 0;
}
