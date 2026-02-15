# VoidCore

High-performance WebGPU/WebGL game engine. C core compiled to WASM (zero dependencies — no Emscripten, no libc). JS handles GPU API calls and scene management. Shared memory via typed array views — no serialization.

## Architecture

```
Game Code (JS) → Scene/Mesh/Camera (JS) → Computation (C/WASM) → GPU Rendering (JS)
                                                                    ├─ WebGPU (primary)
                                                                    └─ WebGL2 (fallback)
```

- **C/WASM owns**: SoA entity storage, TRS→world matrix, frustum culling, draw sorting, all math
- **JS owns**: GPU device/pipelines/buffers/draw calls, Scene/Mesh/Camera classes, geometry generation, glTF/GLB import
- **Coordinate system**: Z-up, right-handed (Blender convention)
- **Matrix format**: Column-major mat4, Euler ZXY rotation order

## Build

Requires Homebrew LLVM and Binaryen (Apple clang lacks wasm32 target):

```sh
brew install llvm binaryen
```

Commands:

```sh
bun run build:wasm    # Compile C → WASM (make -C src/wasm → public/voidcore.wasm)
bun run dev           # Start Next.js dev server (default port 3003)
bun run typecheck     # Type-check with tsgo
bun run lint          # OXLint
bun run format:check  # OXFmt
bun run all           # lint + format + test + typecheck
```

## Directory Structure

```
src/
  wasm/                     C source → compiled to public/voidcore.wasm
    voidcore.c              Init, SoA layout, exported WASM functions
    math.c / math.h         Vec3/Mat4, sin/cos (minimax polynomial), WASM SIMD
    transform.c / .h        m4_from_euler_trs
    cull.c / .h             Frustum plane extraction + bounding sphere culling
    sort.c / .h             Radix sort for draw call ordering
    arena.c / arena.h       Bump allocator using __heap_base
    Makefile                clang --target=wasm32 + wasm-opt
  engine/                   TypeScript engine layer
    wasm.ts                 WASM loader, typed array views, SoA byte offsets
    scene.ts                Scene class (entity mgmt, render orchestration)
    mesh.ts                 Mesh class (writes directly to WASM SoA memory)
    camera.ts               Camera (eye/target/up, VP matrix via WASM)
    gpu.ts                  createRenderer() factory (picks WebGPU or WebGL)
    renderer.ts             WebGPU renderer + Renderer interface
    webgl-renderer.ts       WebGL2 fallback renderer
    geometry.ts             createBoxGeometry, createSphereGeometry
    gltf.ts                 glTF 2.0 / GLB loader with optional Draco mesh compression
    shaders.ts              WGSL shaders (WebGPU)
    webgl-shaders.ts        GLSL 300 es shaders (WebGL2)
    index.ts                Barrel exports
  main.ts                   Demo app (1000 entities, orbit camera, stats overlay)
public/
  voidcore.wasm             Build artifact (gitignored)
  draco-1.5.7/              Draco WASM decoder (used by gltf.ts for compressed meshes)
```

## WASM Memory Layout (32 MB fixed, no growth)

```
[__heap_base..~7.5MB]   SoA arrays (50k max entities):
                          positions, eulerRotations, scales, worldMatrices,
                          colors, flags, bSpheres, geometryIds, sortKeys, visibleIndices
[after SoA..~8MB]       Scratch area (camera matrices, frustum planes)
[16MB..32MB]            Per-frame arena (bump allocator, reset each frame)
```

JS creates `WebAssembly.Memory({ initial: 512 })` (32 MB). WASM imports it via `--import-memory`. Typed array views (`Float32Array`, `Uint32Array`) are created once and never detach.

## JS↔WASM Interface

C exports functions with `__attribute__((export_name("vc_...")))`. JS calls them and reads/writes results directly via shared typed array views at byte offsets.

Key exports: `vc_init`, `vc_compute_world_matrices`, `vc_perspective`, `vc_look_at`, `vc_m4_multiply`, `vc_extract_frustum_planes`, `vc_frustum_cull`, `vc_build_sort_keys`, `vc_sort_draw_calls`, `vc_frame_reset`, `vc_get_*_ptr` (SoA pointer getters).

## Rendering

- **Lighting**: Lambert diffuse (directional + ambient), per-entity unlit flag
- **Geometry**: Box and sphere primitives + glTF/GLB import (with optional Draco compression), vertex format `[px, py, pz, nx, ny, nz]` (24 bytes/vertex), supports Uint16 and Uint32 indices
- **WebGPU**: MSAA 4x, depth24plus, dynamic offset model uniforms (256-byte aligned)
- **WebGL2**: UBOs for camera/model/light, `antialias: true` canvas
- **Bind groups**: Group 0 = camera (view + projection), Group 1 = model (world + color + flags), Group 2 = light (direction + color + ambient)

## Entity Flags (bitfield in u32)

- `0x01` FLAG_DIRTY — needs world matrix recomputation
- `0x02` FLAG_VISIBLE — included in rendering
- `0x04` FLAG_UNLIT — skip lighting calculation

## Render Loop (scene.render())

1. `vc_compute_world_matrices` — TRS→mat4 for dirty entities
2. `camera.update` — view/projection/VP via WASM
3. Update bounding spheres
4. `vc_extract_frustum_planes` + `vc_frustum_cull` — bounding sphere test
5. `vc_build_sort_keys` + `vc_sort_draw_calls` — radix sort by geometry ID
6. Build draw entity list from visible indices
7. `renderer.draw` — submit GPU draw calls
8. `vc_frame_reset` — reset per-frame arena
