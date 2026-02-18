# Voidcore — Materials

## Core Material Types

Two materials, matching the stylized/low-poly target aesthetic:

### BasicMaterial (Unlit)

No lighting calculations. Output = base color × vertex color × texture. For UI elements, debug visualization, skybox-type objects, and glTF assets with `KHR_materials_unlit`.

```typescript
const basic = createBasicMaterial({
  color: [1, 0, 0], // Base color RGB, default [1,1,1]
  colorTexture: texture, // Optional color map
  vertexColors: false, // Use vertex color attribute
  transparent: false, // Route to WBOIT pass
  opacity: 1.0, // Global opacity
})
```

Fragment shader (simplified):

```glsl
vec3 color = u_color;
#ifdef HAS_COLOR_TEXTURE
  color *= texture(u_colorMap, v_uv).rgb;
#endif
#ifdef HAS_VERTEX_COLORS
  color *= v_color.rgb;
#endif
fragColor = vec4(color, u_opacity);
```

### LambertMaterial (Diffuse Lit)

Lambert N·L diffuse shading with ambient. For all lit objects in the scene.

```typescript
const lambert = createLambertMaterial({
  color: [0.8, 0.8, 0.8],       // Base color RGB, default [1,1,1]
  colorTexture: texture,          // Optional color map
  aoTexture: aoTexture,           // Optional ambient occlusion map
  vertexColors: false,            // Use vertex color attribute
  receiveShadow: true,            // Sample shadow map
  transparent: false,             // Route to WBOIT pass
  opacity: 1.0,                   // Global opacity
  palette: [...],                 // Optional material index palette (up to 32 entries)
})
```

Fragment shader core (simplified):

```glsl
vec3 ambient = u_ambientColor * u_ambientIntensity;
float NdotL = max(dot(normal, u_lightDirection), 0.0);
vec3 diffuse = u_lightColor * u_lightIntensity * NdotL;
vec3 litColor = baseColor * (ambient + diffuse * shadow);
vec3 finalColor = litColor + emissive;  // Emissive is additive, unaffected by lighting
```

This is ~15 ALU ops per fragment vs ~80+ for Cook-Torrance PBR. Sufficient for stylized games and leaves significant GPU headroom.

## Material Index Palette

The engine's distinguishing feature for stylized games. A per-vertex `_materialindex` attribute selects a palette entry, enabling complex visual effects with a **single draw call**.

### Palette Structure

Each palette entry contains:

```typescript
interface PaletteEntry {
  color: [number, number, number] // RGB, default [1,1,1]
  opacity: number // Default 1.0
  emissive: [number, number, number] // RGB, default [0,0,0]
  emissiveIntensity: number // Default 0.0
}
```

Up to **32 entries** per material. Stored as a struct array in a UBO (bind group 1).

### Usage Example

```typescript
const characterMaterial = createLambertMaterial({
  palette: [
    { color: [0.9, 0.7, 0.6] }, // 0: skin
    { color: [0.2, 0.3, 0.8] }, // 1: shirt
    { color: [0.1, 0.1, 0.1] }, // 2: hair
    { color: [1, 0.8, 0.2], emissive: [1, 0.8, 0.2], emissiveIntensity: 2.0 }, // 3: glowing eyes
    { color: [0.5, 0.5, 0.5], opacity: 0.4 }, // 4: semi-transparent visor
  ],
  receiveShadow: true,
})
```

A single character mesh with `_materialindex` vertex attribute values 0-4 gets different colors, glow effects, and transparency per surface region — all in one draw call.

### UBO Layout

```
struct PaletteEntry {
  color: vec4       // xyz = RGB, w = opacity
  emissive: vec4    // xyz = RGB, w = emissiveIntensity
}

// UBO: array of 32 PaletteEntry = 32 × 32 bytes = 1024 bytes
```

### Shader Integration

```glsl
#ifdef HAS_MATERIAL_INDEX
  int idx = int(a_materialIndex);
  vec3 baseColor = u_palette[idx].color.rgb;
  float alpha = u_palette[idx].color.a;
  vec3 emissive = u_palette[idx].emissive.rgb * u_palette[idx].emissive.a;
#else
  vec3 baseColor = u_color;
  float alpha = u_opacity;
  vec3 emissive = u_emissive * u_emissiveIntensity;
#endif
```

## Vertex Color Combining

Multiplicative blending:

```glsl
vec3 surfaceColor = materialColor * vertexColor.rgb;
float alpha = materialOpacity * vertexColor.a;
```

Vertex colors tint the material's base color rather than replacing it. This allows the same geometry to be recolored per-material while vertex colors provide per-vertex variation.

When using the palette system with vertex colors, the palette entry color is used as the base, and vertex colors multiply on top.

## Emissive Output

MRT (Multiple Render Targets) with a separate emissive render target for bloom:

```glsl
// Fragment shader dual outputs (opaque pass):
layout(location = 0) out vec4 fragColor;      // Lit scene color → Color RT
layout(location = 1) out vec4 fragEmissive;   // Emissive-only → Emissive RT (bloom source)
```

Emissive values are written to a separate render target that feeds directly into the bloom downsample chain. **No global brightness threshold pass** — bloom is driven purely by explicit emissive output. Only surfaces with non-zero emissive glow. This is physically motivated and gives artists precise control.

The `emissiveIntensity` value controls how bright the bloom is. Values >1.0 produce stronger glow.

## Shader Variant System

Feature-flag bitmask with lazy compilation and caching:

### Feature Flags

```typescript
const enum ShaderFeature {
  HAS_COLOR_TEXTURE = 0x01,
  HAS_AO_TEXTURE = 0x02,
  HAS_VERTEX_COLORS = 0x04,
  HAS_MATERIAL_INDEX = 0x08,
  HAS_SKINNING = 0x10,
  HAS_EMISSIVE = 0x20,
  SHADOW_RECEIVE = 0x40,
  IS_TRANSPARENT = 0x80,
}
```

### Compilation and Caching

```typescript
const shaderCache = new Map<number, CompiledShader>()

const getShader = (materialType: 'basic' | 'lambert', featureMask: number): CompiledShader => {
  const key = (materialType === 'lambert' ? 0x100 : 0) | featureMask
  let shader = shaderCache.get(key)
  if (!shader) {
    shader = compileShaderVariant(materialType, featureMask)
    shaderCache.set(key, shader)
  }
  return shader
}
```

The feature mask is derived from the material's configuration and the mesh's attributes:

```typescript
const computeFeatureMask = (material: Material, geometry: Geometry, mesh: Mesh): number => {
  let mask = 0
  if (material.colorTexture) mask |= ShaderFeature.HAS_COLOR_TEXTURE
  if (material.aoTexture) mask |= ShaderFeature.HAS_AO_TEXTURE
  if (material.vertexColors && geometry.hasAttribute('color')) mask |= ShaderFeature.HAS_VERTEX_COLORS
  if (material.palette && geometry.hasAttribute('materialIndex')) mask |= ShaderFeature.HAS_MATERIAL_INDEX
  if (mesh instanceof SkinnedMesh) mask |= ShaderFeature.HAS_SKINNING
  if (material.hasEmissive()) mask |= ShaderFeature.HAS_EMISSIVE
  if (mesh.receiveShadow) mask |= ShaderFeature.SHADOW_RECEIVE
  if (material.transparent) mask |= ShaderFeature.IS_TRANSPARENT
  return mask
}
```

Typical scene: 10-30 unique shader variants.

## Default Textures

When no texture is provided, a **1×1 white pixel** texture is bound. This avoids shader branching for textured vs untextured paths — the texture sample returns white, which multiplies with the base color to produce the same result. Created once at engine initialization and shared across all materials.

## Transparency Routing

Materials with `transparent: true` or `opacity < 1.0` (or palette entries with opacity < 1.0) are automatically routed to the WBOIT pass. The transparent bit is encoded in the 64-bit sort key so transparent objects are always drawn after all opaque geometry.

## Pipeline State Mapping

Each unique combination of (material type + feature mask + blend mode + depth state) maps to an immutable pipeline:

- **Opaque**: depth write ON, depth test LESS, blend OFF, cull back faces
- **Transparent**: depth write OFF, depth test LESS, WBOIT blend states (additive accumulation + multiplicative revealage), cull back faces
- **Shadow**: depth write ON, depth test LESS, no color write, cull front faces (render back faces)

The material ID is encoded in the sort key so draws with the same material are adjacent, minimizing texture rebinds.

## Texture Handling

- Separate texture + sampler objects on WebGPU
- Combined texture+sampler on WebGL2
- Compressed format support detected at init: ASTC, BC7, ETC2
- Fallback to RGBA8 when no compressed format available
- Mipmaps: pre-computed from KTX2, runtime `generateMipmap` for standard images
- Default sampler: linear filtering, repeat wrapping
