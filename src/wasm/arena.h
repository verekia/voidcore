#ifndef ARENA_H
#define ARENA_H

typedef float f32;
typedef unsigned int u32;
typedef int i32;

void arena_init(u32 base, u32 size);
u32 arena_alloc(u32 bytes, u32 align);
void arena_reset(void);

#endif
