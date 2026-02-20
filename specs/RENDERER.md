# Voidcore — Renderer

## Backend Architecture

WebGPU-first abstraction with WebGL2 translation layer.

```typescript
const device = await createDevice(canvas, {
  backend: 'auto', // 'auto' | 'webgpu' | 'webgl2'
})
```

Auto-detection sequence:

1. Try `navigator.gpu.requestAdapter()` → request device
2. If unavailable or fails, try `canvas.getContext('webgl2')`
3. If neither works, throw `UnsupportedBackendError`

## Render Pass Pipeline

Every frame executes these passes in fixed order:

```
1. Shadow Pass      — Single shadow map, depth-only rendering
2. Opaque Pass      — Sorted draw calls, 4x MSAA, MRT (color + emissive)
3. Transparent Pass — WBOIT accumulation (RGBA16F + R8), depth read-only
4. MSAA Resolve     — Resolve multisample → single-sample
5. OIT Composite    — Blend transparent result over resolved opaque
6. Bloom            — Downsample emissive chain → upsample → composite
7. Final Blit       — gamma correction → screen
```

## Render Target Layout

Pre-allocated at initialization, recreated on canvas resize:

| Target            | Format      | MSAA | Size       | Purpose                               |
| ----------------- | ----------- | ---- | ---------- | ------------------------------------- |
| Color             | RGBA8       | 4x   | Canvas     | Main scene color                      |
| Emissive          | RGBA8       | 4x   | Canvas     | Bloom input (MRT output 1)            |
| Depth             | Depth24Plus | 4x   | Canvas     | Z-buffer                              |
| OIT Accumulation  | RGBA16F     | 1x   | Canvas     | WBOIT weighted color sum              |
| OIT Revealage     | R8          | 1x   | Canvas     | WBOIT alpha product                   |
| Shadow Map        | Depth24Plus | 1x   | Per config | Single shadow depth texture           |
| Bloom Mips        | RGBA16F     | 1x   | Halving    | Progressive downsample/upsample chain |
| Resolved Color    | RGBA8       | 1x   | Canvas     | Post-MSAA resolve                     |
| Resolved Emissive | RGBA8       | 1x   | Canvas     | Post-MSAA emissive resolve            |

## Bind Group Layout

Three bind groups organized by update frequency:

| Slot | Name         | Update Frequency               | Contents                                                                                                         |
| ---- | ------------ | ------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| 0    | Per-frame    | Once per frame                 | Camera VP matrix, light direction/color/intensity, ambient color/intensity, shadow VP matrix, shadow bias params |
| 1    | Per-material | Per material switch            | Material UBO (palette struct array, base color, opacity, flags), color/AO textures, samplers                     |
| 2    | Per-object   | Per draw call (dynamic offset) | World matrix (16 floats), optional bone matrices                                                                 |

Dynamic offsets on bind group 2 allow a single bind group with different offsets per object, minimizing rebind overhead. Per-frame UBO at slot 0 is ~256-320 bytes. Per-material UBO at slot 1 varies by palette size.

## Draw Call Sorting

### 32-Bit Radix Sort

Sort key layout (most significant bits first):

```
Bits 31-30: Layer (2b)        — opaque=0, transparent=1, overlay=2
Bits 29-22: Pipeline ID (8b)  — shader variant + blend/depth state (up to 256 unique)
Bits 21-10: Material ID (12b) — material + texture binding (up to 4096 unique)
Bits 9-0:   Depth (10b)       — front-to-back (opaque) or back-to-front (transparent)
```

**Why this ordering:** Pipeline switches are the most expensive GPU state change (~50μs each on mobile). They occupy the highest bits so draws sharing a pipeline are adjacent. Material/texture switches (~10μs) are medium cost. Depth ordering within the same material enables early-Z rejection for opaque and visual correctness for transparent.

**Why 32 bits:** Pass bits (shadow/main/post) are unnecessary because those are already separate render passes. The transparent flag is redundant with the layer field. 256 pipelines, 4096 materials, and 1024 depth buckets are well beyond what a 2000-draw-call scene needs (typical scenes use 10-30 pipelines and 10-100 materials). Using a single `Uint32Array` avoids `BigUint64Array` / BigInt, which are 10-100× slower than native 32-bit integer operations in JavaScript.

### Radix Sort Implementation

Single-pass 32-bit radix sort over a `Uint32Array`:

```typescript
// Key composition: all native 32-bit bitwise ops
sortKeys[i] = (layer << 30) | (pipelineId << 22) | (materialId << 10) | depth

// Single-pass radix sort: O(n) for 2000 draws — ~0.05ms
```

Radix sort is O(n) vs O(n log n) for comparison sorts. For 2000 draws, it completes in ~0.05ms. Stable, predictable, no worst-case degradation. Pre-allocated count and output arrays are reused across frames (zero allocations).

**Impact:** State change reduction of 90-99% (from ~2000 state changes unsorted to ~10-200 sorted).

## WebGL2 Backend Details

### Full State Cache

Track all GL state to eliminate redundant calls:

```typescript
interface StateCache {
  program: WebGLProgram | null
  vao: WebGLVertexArrayObject | null
  fbo: WebGLFramebuffer | null
  boundTextures: (WebGLTexture | null)[] // per texture unit
  boundUBOs: (WebGLBuffer | null)[] // per binding point
  uboOffsets: number[] // per binding point
  depthWrite: boolean
  depthFunc: number
  blendEnabled: boolean
  blendSrc: number
  blendDst: number
  cullFace: number
  viewport: [number, number, number, number]
}
```

Before every GL call, check the cache. Skip if state is already set. Eliminates 40-60% of redundant GL calls when draws are sorted by pipeline/material.

### Pipeline as State Bundle

A WebGL2 "pipeline" is a frozen record of: program + blend state + depth state + cull state. Applying a pipeline compares against the state cache and sets only what changed.

### VAO Caching

One VAO per geometry+pipeline combination (because different shaders may use different attribute sets). Lazily created on first use, destroyed when geometry or pipeline is disposed.

### BindGroup Translation

- Bind group 0/1/2 → UBO bindings via `gl.bindBufferRange(GL.UNIFORM_BUFFER, bindingPoint, buffer, offset, size)`
- Textures → `gl.activeTexture(GL.TEXTURE0 + unit)` + `gl.bindTexture(GL.TEXTURE_2D, texture)`
- Samplers → sampler state set on the texture object (WebGL2 has separate sampler objects via `gl.bindSampler`)

## WebGPU Backend Details

### Render Bundles

Available for static geometry but not the primary optimization:

- Pre-record draw commands for meshes that don't change between frames
- Replayed with near-zero JS overhead
- Invalidated when objects are added/removed, materials change, or visibility changes
- Typical split: ~1800 static + ~200 dynamic objects

The sort-based renderer with dynamic offsets is the primary strategy on both backends. Render bundles are an optional acceleration for mostly-static scenes.

### Pipeline Caching

All pipelines cached by hash of their descriptor:

```typescript
// Hash inputs: shader variant bitmask, blend mode, depth state, cull mode, vertex layout, sample count
// Typical scene: 10-30 unique pipelines
const pipelineCache = new Map<number, GPURenderPipeline>()
```

## Shader Management

### Dual-Source WGSL + GLSL

Maintain separate WGSL and GLSL shader source files. No runtime transpilation — both are authored and shipped.

```
src/materials/shaders/
  lambert.vert.glsl
  lambert.frag.glsl
  lambert.vert.wgsl
  lambert.frag.wgsl
  basic.vert.glsl
  basic.frag.glsl
  basic.vert.wgsl
  basic.frag.wgsl
  shadow.vert.glsl / .wgsl
  bloom-downsample.frag.glsl / .wgsl
  bloom-upsample.frag.glsl / .wgsl
  oit-composite.frag.glsl / .wgsl
  fullscreen.vert.glsl / .wgsl
```

### Shader Variants via Feature Flags

```glsl
// GLSL example (WGSL uses equivalent #ifdef / const overrides)
#ifdef HAS_COLOR_TEXTURE
  uniform sampler2D u_colorMap;
#endif

#ifdef HAS_AO_TEXTURE
  uniform sampler2D u_aoMap;
#endif

#ifdef HAS_VERTEX_COLORS
  in vec4 a_color;
#endif

#ifdef HAS_MATERIAL_INDEX
  in float a_materialIndex;
#endif

#ifdef HAS_SKINNING
  in vec4 a_joints;
  in vec4 a_weights;
#endif

#ifdef HAS_EMISSIVE
  layout(location = 1) out vec4 fragEmissive;
#endif

#ifdef SHADOW_RECEIVE
  // Shadow map sampling code
#endif
```

Feature bitmask flags:

```
HAS_COLOR_TEXTURE    = 0x01
HAS_AO_TEXTURE       = 0x02
HAS_VERTEX_COLORS    = 0x04
HAS_MATERIAL_INDEX   = 0x08
HAS_SKINNING         = 0x10
HAS_EMISSIVE         = 0x20
SHADOW_RECEIVE       = 0x40
IS_TRANSPARENT       = 0x80
```

Variants compiled lazily on first use and cached by bitmask. Typical scene uses 10-30 unique variants.

### Shader Warm-up

Pre-compile the most common variants during asset loading to prevent hitching:

- Lambert opaque + shadow receive + color texture
- Lambert opaque + shadow receive + vertex colors
- Lambert opaque + shadow receive + material index
- Lambert transparent
- Basic opaque
- Shadow depth-only
- Shadow depth-only + skinning

## MSAA

4x MSAA by default, configurable (1x, 2x, 4x):

- **WebGPU**: Native multisample render target with resolve target
- **WebGL2**: MSAA renderbuffer attachment, resolved via `gl.blitFramebuffer`
- Resolve happens **before** post-processing (bloom operates on 1x resolved textures)
- On tile-based mobile GPUs, MSAA resolve is nearly free (on-chip data)

## Uniform Upload

Per-frame data (camera, lights) uploaded once per frame to a single UBO (bind group 0).

Per-object data uses a shared buffer with dynamic offsets:

```typescript
// Pre-allocated for 4096 objects
const objectBuffer = device.createBuffer({
  size: 4096 * OBJECT_STRIDE, // OBJECT_STRIDE = 256 bytes (world matrix + padding)
  usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
})

// Each frame: write all visible object data in one upload
device.writeBuffer(objectBuffer, 0, objectDataArray, 0, visibleCount * OBJECT_STRIDE)

// Each draw: just change the offset
pass.setBindGroup(2, objectBindGroup, [objectIndex * OBJECT_STRIDE])
```

For skinned meshes, bone matrices are packed after the world matrix within the same stride (32 bones × 64 bytes = 2KB per skinned mesh). The hard 32-bone limit means the UBO is always sufficient — no texture fallback path.

## Per-Draw Overhead Target

**<2-3 microseconds per draw call** in JavaScript.

For 2000 draws: ~4-6ms total JS overhead for draw submission, leaving ~10ms for GPU work within the 16.6ms frame budget.

## Performance Budget

```
Scene graph update:        0.2-0.5ms  (dirty nodes only)
Frustum culling:           0.1-0.2ms  (AABB P-vertex test, brute force)
Draw call sorting:         0.05-0.1ms (radix sort, O(n))
Uniform uploads:           0.2-0.5ms  (single buffer write)
Shadow pass:               0.5-1.0ms  (GPU, depth-only)
Opaque pass (2000 draws):  3.0-5.0ms  (GPU, sorted, MSAA)
Transparent pass:          0.3-0.5ms  (GPU, WBOIT)
MSAA resolve:              0.2-0.5ms  (GPU)
OIT composite:             0.1-0.15ms (GPU, fullscreen)
Bloom:                     0.5-1.1ms  (GPU, progressive)
Final blit:                0.05ms     (GPU)
─────────────────────────────────────────
JS overhead total:         1.0-2.0ms
GPU work total:            5.9-10.3ms
Total frame time:          6.9-12.3ms
Headroom:                  4.3-9.7ms
```

JS and GPU overlap (JS prepares frame N+1 while GPU renders frame N). Effective frame time: max(JS, GPU).
