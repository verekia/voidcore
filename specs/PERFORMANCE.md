# Voidcore — Performance

## Target

**2000 draw calls at 60fps on recent mobile phones.**

- 16.6ms frame budget
- Per-draw JS overhead target: <2-3 microseconds
- Zero GC-triggered frame drops in the render loop

## Performance Strategy (Ranked by Impact)

### 1. Draw Call Sorting (Highest Impact)

**64-bit radix sort by pipeline > material > depth.**

State changes are the dominant CPU-side cost:

- Pipeline (shader) switches: ~50μs each on mobile
- Material/texture binds: ~10μs each
- Buffer offset changes: ~1μs each

With 2000 draws and ~20 unique pipelines, unsorted submission causes ~200 pipeline switches. After sorting: ~20. That's **~9ms saved on mobile**.

Overall state change reduction: **90-99%** (from ~2000 to ~10-200).

The sort is O(n) radix sort over 64-bit keys. For 2000 draws: ~0.05ms.

### 2. Frustum Culling

**AABB P-vertex test against 6 frustum planes.**

Culling invisible objects prevents wasted GPU work. Typical culling ratio: 30-70% of objects invisible per frame (cameras rarely see the whole world).

For 2000 objects, brute-force AABB culling: ~0.1-0.2ms. BVH-accelerated culling available for >5000 objects.

### 3. WebGPU-First

**WebGPU primary, WebGL2 fallback.**

WebGPU has fundamentally lower driver overhead:

- Immutable pipelines (no per-draw state validation)
- Render bundles for static geometry
- Explicit command recording
- Better GPU utilization

On devices with WebGPU, draw call submission is **~30-50% cheaper** than WebGL2.

### 4. Dirty Flags for Incremental Updates

**Update world matrices only for dirty nodes.**

Typical frame: 2-5% of nodes have moved. Dirty flags skip 95%+ of matrix recomputations — ~40× fewer computations vs "update everything every frame" (the Three.js approach).

### 5. State Cache (WebGL2)

**Full GL state tracking, skip redundant API calls.**

Before every GL call, check a state cache. Skip if unchanged. Eliminates 40-60% of redundant API calls when draws are sorted. Essential for making WebGL2 performance approach WebGPU levels.

### 6. Zero Per-Frame Allocations

**No `new`, no array creation, no closures in hot paths.**

- Pre-allocate all math scratch variables at module scope
- Frame-scoped scratch pool with pointer reset (no deallocation, no GC)
- Pre-allocate sort key arrays, draw command arrays, traversal stacks
- Reuse typed arrays across frames
- Target: zero GC collections triggered by the render loop

Even a minor GC pause (2-5ms) eats 12-30% of the frame budget. Eliminating allocations in the render path prevents this entirely.

### 7. Shader Warm-up

**Pre-compile common shader variants during loading.**

Shader compilation on first use causes frame hitches (50-200ms on mobile). Pre-warming during asset loading eliminates this:

- Lambert opaque + shadow + color texture
- Lambert opaque + shadow + vertex colors
- Lambert opaque + shadow + material index
- Lambert transparent
- Basic opaque
- Shadow depth-only
- Shadow depth-only + skinning

## No Automatic Instancing

When per-draw overhead is <2-3μs, 2000 unique draws is feasible without instancing. Instancing adds complexity (instance buffers, batching logic, material compatibility checks) without meaningful benefit at this scale.

The prompt explicitly states "no instancing." Games needing instancing for thousands of identical objects can add it manually in the future.

## Uniform Upload Strategy

**Dynamic offsets on a shared buffer.**

Per-object data: world matrix = 64 bytes (padded to 256 for alignment); skinned meshes add 32 bone matrices = 2KB. Total per-frame upload (non-skinned):

```
2000 objects × 256 bytes = ~512KB via single buffer write
```

Dynamic offsets avoid per-object bind group creation. One bind group for slot 2, different offset per draw call.

## Render Bundles (WebGPU)

Available for static geometry, not the primary optimization:

- Typical split: ~1800 static + ~200 dynamic objects
- Static objects recorded into a render bundle, replayed with near-zero JS cost
- Dynamic objects use normal draw submission
- Bundles invalidated on scene structure changes

Not the primary strategy — the sort-based renderer achieves the target on both backends.

## Data Layout: SoA for Hot Paths

Contiguous typed arrays for renderer-facing data:

```typescript
// Hot data: contiguous, cache-friendly
worldMatrices: Float32Array(maxObjects * 16) // 16 floats per mat4
aabbPool: Float32Array(maxObjects * 6) // 6 floats per AABB
visibilityFlags: Uint8Array(maxObjects) // 1 byte per object
sortKeys: BigUint64Array(maxObjects) // 8 bytes per key

// Public API: wrapper objects index into arrays
mesh.position.set(1, 2, 3) // Writes to internal pool at node's index
```

Iterating 2000 contiguous AABBs for frustum culling is ~10× faster than chasing 2000 heap-allocated objects.

## Render List: Rebuild Every Frame

For 2000 objects: collect visible meshes + radix sort completes in <0.3ms total. Rebuilding is always correct and avoids the complexity of maintaining a persistent sorted list (invalidation on add/remove/visibility change).

## Mobile-Specific Optimizations

### Tile-Based GPU Awareness

Mobile GPUs (Apple, Qualcomm Adreno, ARM Mali) use tile-based rendering:

- **MSAA is nearly free** on tile-based GPUs (resolve happens on-chip)
- **Minimize render target switches** — each forces an expensive tile store/load
- **Half-resolution bloom** reduces memory bandwidth (dominant mobile cost)
- **1024px shadow maps** by default (configurable to 2048 for desktop)
- **9-sample Poisson PCF** provides good quality at reasonable cost

### Memory Bandwidth

Mobile GPUs are bandwidth-constrained. Key mitigations:

- Separate vertex attribute buffers (only fetch what's needed)
- Uint16 index buffers for small meshes
- Compressed textures (ASTC on mobile, reduces bandwidth 4-8×)
- Half-resolution bloom chain

## Frame Budget Breakdown

### CPU Budget (~6ms)

| Phase                            | Cost       |
| -------------------------------- | ---------- |
| Matrix updates (dirty only)      | 0.2-0.5ms  |
| Frustum culling (AABB)           | 0.1-0.2ms  |
| Sort + render list rebuild       | 0.05-0.2ms |
| Draw submission (2000 × 2-3μs)   | 4.0-6.0ms  |
| Animation update (10 characters) | 0.2-0.5ms  |
| Overlay update (20 elements)     | 0.2-0.3ms  |
| Controls + misc                  | 0.1-0.2ms  |

### GPU Budget (~10ms on mobile)

| Phase                        | Cost       |
| ---------------------------- | ---------- |
| Shadow maps (3 CSM, 1024px)  | ~1.5ms     |
| Opaque geometry (2000 draws) | 3.0-5.0ms  |
| Transparent geometry (WBOIT) | 0.3-0.5ms  |
| MSAA resolve                 | 0.2-0.5ms  |
| OIT composite                | 0.1-0.15ms |
| Bloom (5 levels)             | 0.5-1.1ms  |
| Final blit                   | ~0.05ms    |

**JS and GPU overlap:** JS prepares frame N+1 while GPU renders frame N. Effective frame time: `max(JS, GPU)` ≈ 6-10ms → 60fps achievable with headroom.

## Memory Budget

| Resource                          | Budget      |
| --------------------------------- | ----------- |
| JavaScript heap                   | < 50MB      |
| GPU buffers (geometry)            | < 100MB     |
| GPU textures                      | < 200MB     |
| Shadow maps (3 × 1024)            | ~12MB       |
| Bloom chain (1080p, RGBA16F)      | ~5MB        |
| OIT targets (1080p, RGBA16F + R8) | ~18MB       |
| MSAA buffers (4×, 1080p)          | ~32MB       |
| **Total GPU memory**              | **< 400MB** |

## Frame Statistics API

```typescript
const stats = engine.getStats()
{
  fps: number,                // Frames per second
  frameTime: number,          // Total frame time (ms)
  jsTime: number,             // CPU-side time (ms)
  gpuTime: number,            // GPU time if available (ms, requires timer query)
  drawCalls: number,          // Draw calls submitted
  triangles: number,          // Triangles rendered
  stateChanges: number,       // Pipeline + material switches
  pipelineSwitches: number,   // Pipeline switches only
  culledObjects: number,      // Objects culled by frustum
  visibleObjects: number,     // Objects passed to renderer
  shadowDrawCalls: number,    // Draw calls in shadow pass
}
```

GPU timing via:

- WebGPU: `GPUComputePassEncoder.writeTimestamp` / `GPURenderPassEncoder.writeTimestamp`
- WebGL2: `EXT_disjoint_timer_query_webgl2`

## Comparison to Three.js

| Metric            | Voidcore                     | Three.js                                    |
| ----------------- | ---------------------------- | ------------------------------------------- |
| Per-draw overhead | 2-3μs                        | ~15-50μs (5-20× slower)                     |
| GC pressure       | Zero (render loop)           | High (frame spikes from alloc)              |
| Matrix updates    | Dirty-only (2-5% per frame)  | Every object every frame                    |
| Draw call sorting | 64-bit radix sort (O(n))     | No sorting (random state changes)           |
| Uniform upload    | Single UBO + dynamic offsets | Individual `gl.uniform*` calls              |
| Transparency      | WBOIT (no sorting needed)    | Manual sort-based (broken in complex cases) |

## Anti-Patterns Avoided

The following patterns are **banned** in hot paths:

- `new` allocations (use pre-allocated pools)
- `Array.sort` (use radix sort)
- `Map`/`Set` iteration (use flat typed arrays)
- Closure creation (use module-level functions)
- String concatenation for shader keys (use integer bitmasks)
- `gl.getError()` in production (synchronous stall)
- `readPixels` unless explicitly requested (GPU stall)
- `Array.push` in tight loops (use pre-sized arrays with index tracking)
- `for...of` on typed arrays (use indexed `for` loops)
