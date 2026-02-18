# Voidcore — Transparency

## Technique

**Weighted Blended Order-Independent Transparency (WBOIT)**

WBOIT renders all transparent objects in a single pass without sorting. This eliminates the most painful transparency problem in real-time graphics: incorrect visual ordering when transparent surfaces overlap.

Three.js requires manual sort order management and frequently produces visual artifacts. WBOIT makes transparency "just work" for the stylized/low-poly aesthetic Voidcore targets.

## How WBOIT Works

Instead of blending transparent fragments in back-to-front order (which requires sorting), WBOIT accumulates all transparent fragments into two buffers using commutative blend operations (order doesn't matter):

1. **Accumulation buffer**: weighted sum of premultiplied colors
2. **Revealage buffer**: product of (1 - alpha) values — how much of the background shows through

A final composite pass reconstructs the approximate blended color.

## Weight Function

McGuire Equation 10 — depth-based weighting that approximates correct ordering:

```glsl
float weight(float depth, float alpha) {
  return alpha * clamp(0.03 / (1e-5 + pow(depth / 200.0, 4.0)), 1e-2, 3e3);
}
```

- Closer transparent surfaces get higher weight → contribute more to final color
- The `/200.0` matches the 200m world range
- `alpha` factor ensures nearly-invisible fragments don't contribute disproportionately
- Clamped to `[0.01, 3000]` to prevent numerical issues

## Render Targets

| Target       | Format  | Size   | Purpose                               |
| ------------ | ------- | ------ | ------------------------------------- |
| Accumulation | RGBA16F | Canvas | Weighted premultiplied color sum      |
| Revealage    | R8      | Canvas | Alpha product (background visibility) |

**No MSAA on OIT buffers.** WBOIT already provides smooth edges through weighted blending, and MSAA on float16 targets is expensive and unnecessary.

## Blend States

### Accumulation Buffer

```
srcFactor: ONE
dstFactor: ONE
equation:  ADD
```

Additive blending accumulates weighted color contributions. Order doesn't matter because addition is commutative.

### Revealage Buffer

```
srcFactor: ZERO
dstFactor: ONE_MINUS_SRC_ALPHA
equation:  ADD
```

Multiplies accumulated alpha: `result = ∏(1 - αᵢ)`. The revealage starts at 1.0 (fully transparent) and decreases as more opaque fragments are accumulated.

## Transparent Pass Configuration

- **Depth test: ON** — transparent objects respect the opaque depth buffer (don't render behind walls)
- **Depth write: OFF** — transparent objects don't write to depth (don't occlude each other)
- Both accumulation and revealage buffers written simultaneously via MRT

## Fragment Shader (Transparent Pass)

```glsl
void main() {
  // Same lighting as opaque (Lambert N·L + ambient + shadow)
  vec4 color = computeLitColor();

  // Compute WBOIT weight from depth and alpha
  float w = weight(gl_FragCoord.z, color.a);

  // Output 0: accumulation (premultiplied color × weight)
  fragAccum = vec4(color.rgb * color.a * w, color.a * w);

  // Output 1: revealage (alpha for multiplicative blending)
  fragReveal = color.a;
}
```

The fragment shader is almost identical to the opaque shader — same lighting, same material index palette, same vertex color blending. The only difference is the output (weighted accumulation instead of direct color).

## Composite Pass

Full-screen pass that blends the OIT result over the resolved opaque image:

```glsl
void main() {
  vec4 accum = texelFetch(u_accumTexture, ivec2(gl_FragCoord.xy), 0);
  float reveal = texelFetch(u_revealTexture, ivec2(gl_FragCoord.xy), 0).r;

  // Early out: no transparent fragments at this pixel
  if (accum.a < 1e-5) {
    fragColor = texelFetch(u_opaqueTexture, ivec2(gl_FragCoord.xy), 0);
    return;
  }

  // Reconstruct average transparent color
  vec3 averageColor = accum.rgb / max(accum.a, 1e-5);

  // Composite: transparent over opaque
  vec3 opaqueColor = texelFetch(u_opaqueTexture, ivec2(gl_FragCoord.xy), 0).rgb;
  fragColor = vec4(averageColor * (1.0 - reveal) + opaqueColor * reveal, 1.0);
}
```

## Render Pipeline Position

```
1. Shadow pass           (depth-only, 3 cascades)
2. Opaque pass           (MSAA, MRT: color + emissive)
3. Transparent pass      (WBOIT accumulation → RGBA16F + R8)
4. MSAA resolve          (opaque color + emissive → 1x)
5. OIT composite         (blend transparent over resolved opaque)
6. Bloom                 (downsample/upsample on emissive)
7. Final blit            (gamma correction, output to screen)
```

The transparent pass runs after opaque but before MSAA resolve. It reads the opaque depth buffer (for depth testing) but writes to separate non-MSAA targets. The OIT composite happens after MSAA resolve so it blends against the clean resolved color.

## Material API

```typescript
// Fully transparent material
const glass = createLambertMaterial({
  color: [0.5, 0.7, 1.0],
  transparent: true,
  opacity: 0.4,
})

// Per-entry transparency via palette
const material = createLambertMaterial({
  palette: [
    { color: [1, 1, 1], opacity: 1.0 }, // Opaque white
    { color: [1, 0, 0], opacity: 0.3 }, // Transparent red
    { color: [0, 0.5, 1], opacity: 0.6 }, // Translucent blue
  ],
})
```

### Automatic Routing

Materials with `transparent: true`, `opacity < 1.0`, or palette entries with `opacity < 1.0` are automatically routed to the WBOIT pass via the sort key's transparent bit (bit 59).

The sort key system ensures:

1. All opaque objects are drawn first (fills the depth buffer)
2. All transparent objects are drawn second (reads depth, writes to OIT buffers)
3. No manual sorting needed by the user

### Mixed Opaque/Transparent Palettes

When a material has palette entries with mixed opacity (some 1.0, some <1.0), the mesh is rendered in the transparent pass. Palette entries with full opacity will have `weight` proportional to `alpha = 1.0`, which produces correct results in WBOIT (high weight effectively makes them opaque in the accumulation).

For best performance, separate fully opaque and transparent meshes when possible — opaque meshes benefit from early-Z rejection and don't need the OIT overhead.

## WebGL2 Compatibility

Required extensions for WBOIT:

- **`EXT_color_buffer_float`** — render to RGBA16F targets
- **`EXT_float_blend`** — additive blending on float targets

Both are widely supported on WebGL2 devices (95%+ coverage).

**Fallback (if extensions unavailable):** Simple alpha blending with back-to-front sorting (sorted by the depth component of the sort key). This is the "traditional" approach and may show ordering artifacts with complex overlap, but it's better than no transparency.

## Known Limitations

WBOIT is an approximation. The following limitations are acceptable for the stylized/low-poly target:

- **Similar depths with different colors**: very thin overlapping surfaces at nearly the same depth may show subtle blending errors (the weight function can't distinguish them)
- **High alpha overlap (>8 layers)**: extremely dense overlap can saturate the accumulation buffer, producing washed-out results
- **No refraction/volumetric**: WBOIT doesn't support light bending or volumetric scattering (not needed for the target aesthetic)
- **No per-pixel correct ordering**: it's an approximation — for 99% of stylized/low-poly use cases, the visual result is indistinguishable from correct ordering

These tradeoffs eliminate the far worse artifacts of sort-based transparency (z-fighting, popping, per-triangle sorting failures) that plague Three.js.

## Performance Budget

```
Transparent pass (200 transparent objects): ~0.3-0.5ms GPU
OIT composite (fullscreen quad):            ~0.1-0.15ms GPU
──────────────────────────────────────────────────────────
Total OIT overhead:                         ~0.4-0.65ms GPU
```
