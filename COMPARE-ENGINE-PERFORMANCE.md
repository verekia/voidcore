# VoidCore vs V1V2: Performance Comparison

## Context

VoidCore has a C/WASM core for computation (matrix math, frustum culling, radix sort). V1V2 is 100% JS with the same SoA + zero-allocation architecture. Both engines use WebGPU (primary) and WebGL2 (fallback).

For typical scenes (300–1000 entities, up to 2000 draw calls), both engines perform similarly. Both are significantly faster than Three.js, which struggles beyond ~500 draw calls.

## Why Both Engines Beat Three.js

The performance gap with Three.js comes from architectural decisions, not WASM vs JS:

- **No per-frame allocations.** Three.js creates `Matrix4`, `Vector3`, `Quaternion` objects constantly, causing GC pressure and cache misses. Both engines use pre-allocated TypedArrays with offset-based math.
- **No scene graph traversal.** Three.js walks a tree of `Object3D` nodes with recursive matrix multiplication. Both engines have flat SoA arrays.
- **No material system overhead.** Three.js has a shader program cache, uniform diffing, and material hashing. Both engines use a handful of hardcoded pipelines (static, skinned, textured).
- **No abstraction tax.** Three.js wraps every GPU concept in classes with getters, setters, change tracking, and event dispatchers. Both engines talk to WebGPU/WebGL almost directly.

## Why WASM Doesn't Matter Much at This Scale

At 300–1000 entities, CPU-side work (matrix math, culling, sorting) takes well under 1ms per frame. The 16ms frame budget is dominated by GPU work (buffer uploads, draw call submission, shader execution), which both engines handle identically in JS.

Where WASM theoretically wins — tight loops over contiguous memory, custom minimax sin/cos, no GC — is fighting over microseconds at this entity count. V8/SpiderMonkey JIT-compile TypedArray math to near-native code, and the JS-to-WASM boundary crossing for small calls (perspective, lookAt, multiply) partially offsets the raw throughput advantage.

The WASM core would start to matter at 10k+ entities, but even then the biggest wins come from algorithmic choices that can be implemented in pure JS.

## Performance Tricks to Port from VoidCore to V1V2

All four optimizations below are algorithmic — none require WASM.

### 1. Dirty Flags (Highest Impact)

**VoidCore:** `vc_compute_world_matrices` checks `FLAG_DIRTY` per entity and skips clean ones. Game code calls `mesh.setDirty()` when position/rotation/scale changes.

**V1V2:** Recomputes every world matrix every frame via `m4FromTRS` for all entities, regardless of whether they moved.

**Impact:** At 1000 entities where ~50 move per frame, this eliminates 950 unnecessary `m4FromTRS` calls, each involving 6 trig functions and ~60 multiplies.

**Implementation:** Add a `dirty: Uint8Array(maxEntities)` flag array. Set on position/rotation/scale mutation. Skip `m4FromTRS` when clean. Clear after computing.

### 2. Direct SoA Writes (Eliminates Copy Pass)

**VoidCore:** `Mesh.position` is a `Float32Array.subarray()` pointing directly into the SoA buffer. When game code writes `mesh.position[0] = 5`, it writes straight into the data the renderer consumes. Zero copy.

**V1V2:** Each Mesh owns its own `Float32Array(3)` for position, rotation, scale, color. Every frame, a sync loop copies all per-Mesh arrays into the SoA TypedArrays — even if nothing changed.

**Impact:** Eliminates an O(n) copy pass over every entity's position, scale, color, alpha, and flags each frame. Combined with dirty flags, unchanged entities require zero per-frame work.

**Implementation:** Have Mesh objects hold subarray views into the scene's SoA TypedArrays instead of owning separate small arrays.

### 3. Single-Pass Frustum Culling (Eliminates Redundant Tests)

**VoidCore:** Frustum culling runs once per frame via `vc_frustum_cull`, producing a compact `visibleIndices` array. All downstream passes (draw entity list building, sorting) iterate only the visible set.

**V1V2:** Frustum-tests each entity inline inside each pipeline pass in `drawScene()` — unlit, opaque static, opaque textured, opaque skinned, transparent. An entity can be tested against 6 frustum planes up to 5 times per frame.

**Impact:** Reduces frustum culling work by up to 5x. More importantly, produces a single visible index list that all subsequent logic can share.

**Implementation:** Build a `visibleIndices: Uint32Array` once per frame before draw submission. All pipeline passes iterate this list instead of the full entity set.

### 4. Draw Call Sorting by Geometry ID (Reduces GPU State Changes)

**VoidCore:** Packs `(geometryId << 16) | entityIndex` into sort keys and radix-sorts them. This groups draw calls by geometry, minimizing vertex/index buffer rebinds on the GPU.

**V1V2:** Iterates entities in insertion order within each pipeline category. No attempt to group same-geometry entities together.

**Impact:** At 1000 entities with 10 geometry types, reduces buffer rebinds from ~1000 to ~10. The GPU-side savings depend on the driver, but fewer state changes is universally beneficial.

**Implementation:** A counting sort on geometry ID is simpler than radix sort for this use case. With max ~256 geometry types, it's O(n), and can be zero-alloc with pre-allocated bucket arrays.

## Priority Order

1. **Dirty flags** — eliminates the most redundant CPU work
2. **Direct SoA writes** — eliminates the Mesh-to-SoA copy pass
3. **Single-pass frustum culling** — removes redundant per-pass sphere tests
4. **Draw call sorting** — reduces GPU state changes

The first two eliminate work proportional to entity count. The last two reduce constant factors. All four are straightforward to implement in pure TypeScript with zero new dependencies.
