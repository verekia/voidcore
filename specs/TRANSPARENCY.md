# Transparency

Sorted alpha-blend transparency with premultiplied alpha. Transparent meshes are drawn back-to-front after all opaques in the same render pass — no extra render targets or composite passes needed.

## Material API

```ts
new LambertMaterial({ color: [0.2, 0.6, 1.0], opacity: 0.7, transparent: true })
new BasicMaterial({ color: [1, 0, 0], opacity: 0.5, transparent: true })
```

- `opacity` (number, default `1.0`) — Alpha value in the range [0, 1].
- `transparent` (boolean, default `false`) — When true, the mesh uses a transparent pipeline (blend enabled, depth writes disabled, back-face culling disabled).
- The glTF loader sets `transparent: true` when `alphaMode === 'BLEND'` or when `baseColorFactor[3] < 1`.

## Sort Key Layout

Meshes are sorted using a 32-bit radix sort key. Opaque and transparent meshes use different bit layouts to optimize for their respective priorities:

```
Bits 31-30: Layer — 0 = opaque, 1 = transparent

Opaque (layer 0) — minimize state changes:
  Bits 29-22: Pipeline ID (shader program variant)
  Bits 21-10: Material ID (uniform group)
  Bits 9-0:   Depth (nearest-first for early-Z)

Transparent (layer 1) — correct depth order:
  Bits 29-20: Depth (farthest-first for back-to-front blending)
  Bits 19-12: Pipeline ID
  Bits 11-0:  Material ID
```

For opaque meshes, state-change minimization is the primary sort criterion — pipeline and material grouping reduces expensive GPU state switches, with depth as a tiebreaker for early-Z benefits.

For transparent meshes, depth ordering is the primary criterion — correct alpha blending requires strict back-to-front rendering. Pipeline and material grouping are demoted to secondary/tertiary criteria.

The `findTransparentStart()` helper scans the sorted key array for the first key with bit 30 set, splitting the draw list into opaque `[0, transparentStart)` and transparent `[transparentStart, meshCount)` ranges.

## Render Pipeline

Both backends draw opaques and transparents in the same render pass (no extra pass needed):

1. **Opaque loop** `[0, transparentStart)` — normal pipeline (depth write on, back-face cull, no blend).
2. **Transparent loop** `[transparentStart, meshCount)` — transparent pipeline:
   - Blend enabled (premultiplied alpha on WebGPU, straight alpha on WebGL2)
   - Depth writes disabled (reads depth buffer but doesn't write to it)
   - Back-face culling disabled (both sides visible)

### WebGPU Pipeline State

Transparent pipelines are separate `GPURenderPipeline` objects created at init (4 variants: basic/lambert × static/skinned):

```
blend: { color: one / one-minus-src-alpha, alpha: one / one-minus-src-alpha }
depthWriteEnabled: false
depthCompare: less-equal
cullMode: none
```

### WebGL2 State

WebGL2 sets blend/depth/cull state dynamically between the opaque and transparent loops:

```
gl.enable(GL_BLEND)
gl.blendFuncSeparate(SRC_ALPHA, ONE_MINUS_SRC_ALPHA, ONE, ONE_MINUS_SRC_ALPHA)
gl.depthMask(false)
gl.disable(GL_CULL_FACE)
```

State is restored after the transparent loop.

## Premultiplied Alpha (WebGPU)

WebGPU shaders output premultiplied color: `rgb * alpha` in both color and emissive MRT targets. The blend factors are `one / one-minus-src-alpha` instead of `src-alpha / one-minus-src-alpha`.

This avoids using the `src-alpha` blend factor, which triggers `VK_ERROR_UNKNOWN` on some Android Vulkan drivers when combined with `textureSampleCompareLevel` (comparison texture sampling used by shadow maps).

WebGL2 uses straight (non-premultiplied) alpha since it doesn't share this driver issue.

## Shader Changes

All four shader variants (lambert, lambert-skinned, basic, basic-skinned) read `material.opacity` for alpha. Fragment outputs:

**WebGPU (premultiplied)**:

```wgsl
out.color = vec4<f32>(finalColor * alpha, alpha);
out.emissive = vec4<f32>(emissive * alpha, alpha);
```

**WebGL2 (straight alpha)**:

```glsl
fragColor = vec4(finalColor, alpha);
fragEmissive = vec4(emissive * alpha, alpha);
```

The `u_opacity` uniform (WebGL2) and `material.opacity` field (WebGPU) are written per-material during the draw loop.

## glTF Import

The loader reads alpha from glTF materials:

- `alphaMode === 'BLEND'` → `transparent: true`
- `baseColorFactor[3] < 1` → `transparent: true`, `opacity = baseColorFactor[3]`

## Limitations

- **Order-dependent artifacts**: Like all sorted alpha-blend approaches, intersecting transparent geometry or complex overlap patterns can produce incorrect results. For most game/viz use cases (windows, particles, UI panels), sorted blending is sufficient and much cheaper than OIT.
- **No per-pixel sorting**: Sorting is per-mesh center, not per-fragment. Large transparent meshes that overlap may not blend correctly at all pixels.
- **Single-pass**: Transparent meshes share the same render pass as opaques. This is efficient but means transparent meshes see the opaque depth buffer as-is (no depth peeling).
