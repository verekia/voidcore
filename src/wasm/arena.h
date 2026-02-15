#ifndef ARENA_H
#define ARENA_H

typedef float f32;
typedef unsigned int u32;
typedef int i32;

// Safe WASM linear memory access: integer-to-pointer cast avoids null pointer
// arithmetic UB that can cause LTO to emit 'unreachable' instructions.
#define WASM_PTR(type, byte_offset) ((type*)(void*)(__UINTPTR_TYPE__)(byte_offset))

void arena_init(u32 base, u32 size);
u32 arena_alloc(u32 bytes, u32 align);
void arena_reset(void);

#endif
