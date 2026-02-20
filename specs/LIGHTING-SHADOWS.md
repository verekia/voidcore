# Voidcore — Lighting & Shadows

## Lighting Model

Lambert diffuse shading (N·L) with ambient — no specular component:

```glsl
vec3 ambient = u_ambientColor * u_ambientIntensity;
float NdotL = max(dot(normal, u_lightDirection), 0.0);
vec3 diffuse = u_lightColor * u_lightIntensity * NdotL;
vec3 litColor = baseColor * (ambient + diffuse * shadow);
vec3 finalColor = litColor + emissive;    // Emissive is additive, unaffected by lighting
```

**Why Lambert only:**

- ~15 ALU ops per fragment vs ~80+ for Cook-Torrance PBR
- Sufficient for stylized/low-poly games (the target aesthetic)
- No need for metallic/roughness textures, environment probes, or IBL
- Can be extended later to PBR without architectural changes (same light data, same shadow maps)

## Light Types

### Directional Light (Scene Node)

A regular scene graph node. Direction derived from the node's world position (treated as a direction vector — like the sun, infinitely far away). Shadow configuration lives on the light itself.

```typescript
const sun = new DirectionalLight({
  color: [1, 1, 0.95], // Warm white
  intensity: 1.0,
  castShadow: true,
  shadowMapSize: 200, // World-space ortho box size (default 200)
  shadowNear: 1, // Near clip plane (default 1)
  shadowFar: 300, // Far clip / eye offset distance (default 300)
  shadowBias: 0.001, // Constant depth bias (default 0.001)
  shadowSlopeBias: 0.005, // Slope-dependent bias (default 0.005)
})
sun.setPosition(50, 50, 100) // Direction only — shadow volume is always at world origin
scene.add(sun)
```

Because it's a scene node, the light can be:

- Parented to other nodes
- Animated (rotating sun for day/night cycle)
- Toggled via `light.visible = false`

### Ambient Light (Scene Property)

No spatial properties — stored directly on the Scene rather than as a node:

```typescript
scene.ambientLight = { color: [0.4, 0.45, 0.5], intensity: 0.3 }
```

Ambient provides a base illumination level so shadowed areas aren't pure black. The bluish tint simulates sky-colored ambient for outdoor scenes.

## Light Data Upload

All light data packed into the per-frame UBO (bind group 0). One upload per frame, zero per-draw overhead:

```
Per-frame UBO layout (bind group 0) — WebGPU (192 bytes / 48 floats):
  mat4  viewProjectionMatrix      // 64 bytes (float 0–15)
  vec3  lightDirection            // 12 bytes (float 16–18)
  f32   lightIntensity            // 4 bytes  (float 19)
  vec3  lightColor                // 12 bytes (float 20–22)
  f32   ambientIntensity          // 4 bytes  (float 23)
  vec3  ambientColor              // 12 bytes (float 24–26)
  f32   shadowEnabled             // 4 bytes  (float 27)
  mat4  shadowVP                  // 64 bytes (float 28–43, aligned to 16-byte boundary)
  f32   constantBias              // 4 bytes  (float 44)
  f32   slopeBias                 // 4 bytes  (float 45)
  f32   invMapSize                // 4 bytes  (float 46, = 1.0 / shadowResolution)
  f32   padding                   // 4 bytes  (float 47)
  ─────────────────────────────
  Total: 192 bytes
```

WebGL2 layout is similar but uses std140 alignment (208 bytes / 52 floats), with the shadow VP matrix at float 32–47 due to mat4 alignment padding.

## Shadow Map

### Single Shadow Map

A single depth texture covering a fixed orthographic volume centered at the world origin, oriented along the light direction. Camera-independent — shadow quality is consistent regardless of where the camera is looking. Sufficient quality for small-to-medium low-poly worlds.

### Shadow Map Resolution

**Default: 2048×2048**. Configurable via `resolution`.

```
2048 × 2048 × 4 bytes (Depth24Plus) = ~16MB GPU memory
```

### Shadow Map Storage

**Single 2D depth texture** (`depth24plus`):

```typescript
// WebGPU
const shadowMap = device.createTexture({
  format: 'depth24plus',
  size: { width: mapSize, height: mapSize },
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
})

// WebGL2
// Depth texture attached to a dedicated FBO
```

### Shadow Matrix Computation

The shadow volume is a fixed orthographic box centered at the world origin, oriented along the light direction. The camera has no involvement — shadow quality is consistent regardless of camera position.

1. Center = world origin `[0, 0, 0]`
2. Eye = `lightDir × shadowFar` (placed behind the scene along light direction)
3. Build light view matrix via `mat4LookAt(eye, center, up)`
4. Transform center to light space, snap XY to texel grid (`shadowMapSize / shadowResolution`)
5. Build orthographic projection: `snappedXY ± halfSize`, near = `shadowNear`, far = `shadowFar × 2`
6. `shadowVP = lightProj × lightView`

```typescript
const center = [0, 0, 0] // World origin
const eye = [lightDir[0] * shadowFar, lightDir[1] * shadowFar, lightDir[2] * shadowFar]
const lightView = mat4LookAt(lightViewMatrix, eye, center, VEC3_UP)

// Texel snapping to prevent shadow shimmer
const texelSize = shadowMapSize / shadowResolution
const snappedX = Math.floor(centerInLightSpace.x / texelSize) * texelSize
const snappedY = Math.floor(centerInLightSpace.y / texelSize) * texelSize

const halfSize = shadowMapSize / 2
const lightProj = mat4OrthoZO(
  // [0,1] depth for WebGPU, [-1,1] for WebGL2
  lightProjMatrix,
  snappedX - halfSize,
  snappedX + halfSize,
  snappedY - halfSize,
  snappedY + halfSize,
  shadowNear,
  shadowFar * 2,
)
shadowVP = mat4Multiply(result, lightProj, lightView)
```

### Texel Snapping

Prevents shadow shimmer by snapping the light projection to texel boundaries. Since the shadow volume is fixed at the world origin (camera-independent), shimmer is minimal — snapping handles sub-texel alignment.

### PCF Filtering

3×3 grid PCF (9 samples):

```wgsl
fn pcf9(uv: vec2<f32>, d: f32, ts: f32) -> f32 {
  var s = 0.0;
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2(-ts, -ts), d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2(0.0, -ts), d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2( ts, -ts), d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2(-ts,  0.0), d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv,                   d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2( ts,  0.0), d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2(-ts,  ts), d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2(0.0,  ts), d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2( ts,  ts), d);
  return s / 9.0;
}
```

Hardware comparison samplers on both backends:

- WebGPU: `sampler_comparison` with `compare: 'less'` and `textureSampleCompareLevel()`
- WebGL2: `sampler2DShadow` with `TEXTURE_COMPARE_MODE`

### Shadow Bias

Combined depth bias + slope-scaled bias:

```glsl
float bias = constantBias + slopeBias * (1.0 - NdotL);
```

- `constantBias` (default 0.001): prevents acne on surfaces facing the light
- `slopeBias` (default 0.005): increases bias for surfaces at grazing angles (where acne is worst)

Additional acne reduction: **front-face culling during shadow pass** — render back faces to the depth map, so the depth sample is from the back face, naturally offsetting from the front face.

### Shadow Pass Rendering

Single depth-only pass:

1. Set render target to the shadow depth texture
2. Clear depth to 1.0
3. Set the shadow VP matrix as the camera
4. **Light-space frustum culling**: project mesh center to light space, skip if outside bounds (with 0.3 padding margin)
5. Render camera-visible meshes with `castShadow = true`
6. Also render shadow-only casters (outside camera frustum but inside shadow frustum)
7. **Position-only vertex shader** (minimal attribute fetching)
   - Exception: skinned meshes also need joint/weight attributes for skeletal transform
8. **Front-face culling**: render back faces to reduce acne
9. Disable color writes (depth-only)

### Per-Mesh Shadow Control

```typescript
mesh.castShadow = true // Include in shadow pass (default false)
mesh.receiveShadow = true // Sample shadow map in fragment shader (default false)
```

Objects with `castShadow: false` are skipped during shadow pass rendering. Objects with `receiveShadow: false` get `shadow = 1.0` (fully lit) in the fragment shader.

## Configuration API

Shadow configuration is split between the renderer (GPU texture) and the light (scene-level):

```typescript
// Renderer config: only texture resolution (GPU resource, set at init)
const engine = await Engine.create(canvas, {
  shadows: {
    enabled: true,
    resolution: 2048, // Shadow map texture size in pixels (default 2048)
  },
})

// Boolean shorthand uses all defaults:
const engine = await Engine.create(canvas, { shadows: true })

// Light config: ortho box size, depth range, and bias (scene-level, can change at runtime)
const light = new DirectionalLight({
  castShadow: true,
  shadowMapSize: 200, // World-space ortho box size (default 200)
  shadowNear: 1, // Near clip plane (default 1)
  shadowFar: 300, // Far clip / eye offset distance (default 300)
  shadowBias: 0.001, // Constant depth bias (default 0.001)
  shadowSlopeBias: 0.005, // Slope-dependent bias (default 0.005)
})
```

### Shadow Baking

For static scenes where neither the light nor shadow-casting objects move, the shadow map can be frozen after the first render:

```typescript
// Imperative
engine.shadowsBaked = true   // Freeze
engine.shadowsBaked = false  // Resume real-time

// React
<BakeShadows />  // Mount to freeze, unmount to resume
```

## Performance Budget

| Pass                       | Cost (2048 resolution) |
| -------------------------- | ---------------------- |
| Shadow depth pass          | ~0.8ms GPU             |
| PCF sampling (opaque pass) | ~0.3ms GPU             |
| **Total shadow overhead**  | **~1.1ms GPU**         |

Shadow pass is cheap because it uses depth-only rendering (no color writes, no lighting, minimal fragment work).
