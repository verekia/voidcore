# Voidcore — Post-Processing

## Overview

Voidcore's post-processing pipeline consists of three stages after the main scene passes:

1. **OIT Composite** — blend transparent result over resolved opaque (covered in TRANSPARENCY.md)
2. **Bloom** — Unreal-style progressive downsample/upsample driven by MRT emissive
3. **Final Blit** — gamma correction → screen

## Bloom

### Input: MRT Emissive

Bloom is driven by the dedicated emissive render target (MRT output 1 from the opaque pass). **No global brightness threshold pass.** Only geometry with non-zero emissive values contributes to bloom.

This is physically motivated: only light-emitting surfaces glow. An artist sets `emissiveIntensity: 2.0` on a palette entry, and that surface — and only that surface — blooms. There's no risk of white walls or bright reflections accidentally triggering bloom.

### Downsample Chain

**5 levels, starting at half resolution:**

For 1920×1080 canvas:

```
Input:   Resolved emissive (1920×1080)
Level 0: 960×540    (half res, most expensive)
Level 1: 480×270
Level 2: 240×135
Level 3: 120×67
Level 4: 60×33     (widest glow radius)
```

5 levels provide a wide enough glow radius for the stylized aesthetic. Starting at half resolution reduces the cost of the first downsample (most fragment invocations).

### Downsample Filter: 13-Tap Jimenez

The 13-tap kernel combines 4 bilinear samples to approximate a 4×4 box filter with Gaussian weighting. This reduces aliasing during downsampling:

```glsl
vec3 downsample13Tap(sampler2D tex, vec2 uv, vec2 texelSize) {
  // 13 taps arranged in a cross+diamond pattern
  vec3 a = texture(tex, uv + vec2(-2, -2) * texelSize).rgb;
  vec3 b = texture(tex, uv + vec2( 0, -2) * texelSize).rgb;
  vec3 c = texture(tex, uv + vec2( 2, -2) * texelSize).rgb;
  vec3 d = texture(tex, uv + vec2(-2,  0) * texelSize).rgb;
  vec3 e = texture(tex, uv                           ).rgb;
  vec3 f = texture(tex, uv + vec2( 2,  0) * texelSize).rgb;
  vec3 g = texture(tex, uv + vec2(-2,  2) * texelSize).rgb;
  vec3 h = texture(tex, uv + vec2( 0,  2) * texelSize).rgb;
  vec3 i = texture(tex, uv + vec2( 2,  2) * texelSize).rgb;
  vec3 j = texture(tex, uv + vec2(-1, -1) * texelSize).rgb;
  vec3 k = texture(tex, uv + vec2( 1, -1) * texelSize).rgb;
  vec3 l = texture(tex, uv + vec2(-1,  1) * texelSize).rgb;
  vec3 m = texture(tex, uv + vec2( 1,  1) * texelSize).rgb;

  // Weighted combination (weights from Jimenez 2014)
  vec3 result = e * 0.125;
  result += (a + c + g + i) * 0.03125;
  result += (b + d + f + h) * 0.0625;
  result += (j + k + l + m) * 0.125;
  return result;
}
```

### Karis Average (First Downsample Only)

On the first downsample step, apply Karis weighted average to prevent firefly artifacts:

```glsl
// Weight each sample by 1 / (1 + luma) to suppress bright outliers
float karisWeight(vec3 color) {
  float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
  return 1.0 / (1.0 + luma);
}
```

This prevents single bright pixels (e.g., a small emissive light source) from creating disproportionately large bloom halos. Subsequent downsample levels use the standard 13-tap filter without Karis weighting.

### Upsample Filter: 3×3 Tent (9-Tap)

The tent filter provides smooth upsampling with additive blending — each upsample step adds the wider glow from the level below:

```glsl
vec3 upsample9Tap(sampler2D tex, vec2 uv, vec2 texelSize) {
  vec3 result = texture(tex, uv + vec2(-1, -1) * texelSize).rgb * 1.0
              + texture(tex, uv + vec2( 0, -1) * texelSize).rgb * 2.0
              + texture(tex, uv + vec2( 1, -1) * texelSize).rgb * 1.0
              + texture(tex, uv + vec2(-1,  0) * texelSize).rgb * 2.0
              + texture(tex, uv                            ).rgb * 4.0
              + texture(tex, uv + vec2( 1,  0) * texelSize).rgb * 2.0
              + texture(tex, uv + vec2(-1,  1) * texelSize).rgb * 1.0
              + texture(tex, uv + vec2( 0,  1) * texelSize).rgb * 2.0
              + texture(tex, uv + vec2( 1,  1) * texelSize).rgb * 1.0;
  return result / 16.0;
}
```

The upsample chain runs from the smallest level back to half-resolution. Each level is composited additively onto the previous, accumulating progressively wider glow:

```
Upsample 4→3: Level 3 += upsample(Level 4)
Upsample 3→2: Level 2 += upsample(Level 3)
Upsample 2→1: Level 1 += upsample(Level 2)
Upsample 1→0: Level 0 += upsample(Level 1)
```

### Bloom Composite

The final bloom result (Level 0, half-resolution) is composited with the scene color using the bloom intensity:

```glsl
vec3 finalColor = sceneColor + bloomColor * u_bloomIntensity;
```

This additive blend makes emissive surfaces appear to glow without affecting non-emissive areas.

### Render Target Format

**RGBA16F** for the entire bloom chain. HDR precision preserves emissive values >1.0 through the pipeline. RGBA8 would clamp bright values and produce flat, unnatural glow.

### Full-Screen Geometry

All full-screen passes (downsample, upsample, composite, blit) use a single oversized triangle:

```glsl
// Vertex shader — no vertex buffer needed, uses gl_VertexIndex / gl_VertexID
vec2 position = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
v_uv = position;
```

A single triangle that covers the full viewport is more efficient than a quad (2 triangles) because there's no diagonal edge causing redundant fragment work.

## Final Blit

### Gamma Correction

Applied after bloom compositing:

```glsl
fragColor.rgb = pow(fragColor.rgb, vec3(1.0 / 2.2));
```

On WebGPU with an sRGB surface format, gamma correction is handled by the hardware. On WebGL2, it's applied in the shader.

## MSAA Integration

MSAA is resolved **before** post-processing:

```
Opaque pass (4x MSAA) → Resolve → [1x Color, 1x Emissive]
                                          ↓
Bloom operates on resolved 1x emissive texture
Final blit operates on resolved 1x color + bloom
```

OIT buffers are always 1x (no MSAA on transparent pass — WBOIT already provides smooth edges through weighted blending, and MSAA on float targets is expensive).

## Configuration

```typescript
const engine = await createEngine(canvas, {
  bloom: {
    enabled: true,
    intensity: 0.5, // Bloom mix strength (0 = no bloom, 1 = full bloom)
    levels: 5, // Downsample levels (default 5)
  },
  antialias: true, // 4x MSAA (default true)
})

// Boolean shorthand uses defaults:
const engine = await createEngine(canvas, { bloom: true })
```

## Performance Budget

```
Bloom downsample (5 levels):     ~0.3-0.5ms GPU
Bloom upsample (5 levels):      ~0.2-0.4ms GPU
Bloom composite:                 ~0.05ms GPU
Final blit (gamma):              ~0.05ms GPU
────────────────────────────────────────────────
Total post-processing:           ~0.6-1.0ms GPU
```

The bloom chain is the dominant cost. Each downsample/upsample level is a full-screen pass at progressively smaller resolutions, so the total cost is bounded by the half-resolution first level.
