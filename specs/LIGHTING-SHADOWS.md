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

A regular scene graph node. Direction derived from the node's world rotation (default: pointing down -Z in local space, transformed by world rotation).

```typescript
const sun = createDirectionalLight({
  color: [1, 1, 0.95], // Warm white
  intensity: 1.0,
  castShadow: true,
})
sun.position.set(50, 50, 100) // Position affects shadow map framing
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
Per-frame UBO layout (bind group 0):
  mat4  viewProjectionMatrix      // 64 bytes
  vec4  lightDirection            // 16 bytes (xyz = direction, w = unused)
  vec4  lightColor                // 16 bytes (xyz = color, w = intensity)
  vec4  ambientColor              // 16 bytes (xyz = color, w = intensity)
  mat4  shadowMatrix[3]           // 192 bytes (3 cascade VP matrices)
  vec4  cascadeSplits             // 16 bytes (xyz = split distances, w = unused)
  vec4  shadowParams              // 16 bytes (x = bias, y = slopeBias, z = blendRange, w = mapSize)
  vec4  cameraPosition            // 16 bytes (xyz = position, w = near)
  vec4  cameraParams              // 16 bytes (x = far, yzw = unused)
  ─────────────────────────────
  Total: ~368 bytes
```

## Cascaded Shadow Maps

### 3-Cascade CSM

3 cascades for 200×200m low-poly worlds:

```
Cascade 0:  ~0.1m – 12m    (high detail near camera)
Cascade 1:  ~12m – 50m     (mid range)
Cascade 2:  ~50m – 200m    (far range)
```

### Shadow Map Resolution

**Default: 1024×1024 per cascade** (mobile-first). Configurable to 2048 for desktop.

```
3 cascades × 1024 × 1024 × 4 bytes (Depth24Plus) = ~12MB GPU memory
3 cascades × 2048 × 2048 × 4 bytes                = ~48MB GPU memory
```

### Shadow Map Storage

**Texture 2D Array** with 3 layers (`TEXTURE_2D_ARRAY`):

```typescript
const shadowMap = device.createTexture({
  format: 'depth24plus',
  size: { width: mapSize, height: mapSize, depthOrArrayLayers: 3 },
  usage: TextureUsage.RENDER_ATTACHMENT | TextureUsage.SAMPLED,
})
```

Advantages over a texture atlas:

- Array index in shader instead of UV offset calculations
- Better GPU cache behavior (each cascade is a separate layer)
- No texel bleed between cascades
- Supported on all WebGL2 (`TEXTURE_2D_ARRAY`) and WebGPU devices

### Cascade Split Scheme

Logarithmic-weighted split with lambda = 0.7:

```typescript
const lambda = 0.7 // More resolution near camera
const near = camera.near
const far = shadowConfig.maxDistance // 200m default

for (let i = 0; i < cascadeCount; i++) {
  const t = (i + 1) / cascadeCount
  const log = near * Math.pow(far / near, t)
  const linear = near + (far - near) * t
  splits[i] = lambda * log + (1 - lambda) * linear
}
```

Lambda 0.7 concentrates more shadow resolution near the camera where players look most. For a 200m world with 3 cascades:

- Split 0: ~12m
- Split 1: ~50m
- Split 2: 200m

### Shadow Matrix Computation

For each cascade:

1. Compute the frustum slice (near plane to split distance) in view space
2. Transform the 8 frustum corners to world space
3. Compute a tight AABB around the frustum corners in light space
4. Build an orthographic projection from the AABB
5. Apply texel snapping to prevent shimmer

```typescript
// Per cascade:
const frustumCorners = getFrustumSliceCorners(camera, splitNear, splitFar)
const lightView = mat4LookAt(lightViewMatrix, lightPosition, lightTarget, VEC3_UP)
const lightSpaceAABB = computeAABBInLightSpace(frustumCorners, lightView)
const lightProj = mat4Ortho(
  lightProjMatrix,
  lightSpaceAABB.minX,
  lightSpaceAABB.maxX,
  lightSpaceAABB.minY,
  lightSpaceAABB.maxY,
  lightSpaceAABB.minZ - backExtend,
  lightSpaceAABB.maxZ,
)
cascadeShadowMatrix[i] = mat4Multiply(result, lightProj, lightView)
```

### Texel Snapping

Prevent shadow shimmer/swimming during camera movement:

```typescript
const texelSize = cascadeWorldSize / cascadeResolution
lightMatrix[12] = Math.floor(lightMatrix[12] / texelSize) * texelSize
lightMatrix[13] = Math.floor(lightMatrix[13] / texelSize) * texelSize
```

Snaps the light projection to texel boundaries so the shadow map doesn't sub-pixel-shift when the camera moves.

### PCF Filtering

3×3 Poisson disk PCF (9 samples):

```glsl
const vec2 poissonDisk[9] = vec2[](
  vec2(-0.94, -0.34), vec2(0.94, 0.34),
  vec2(-0.34, 0.94),  vec2(0.34, -0.94),
  vec2(-0.54, -0.54), vec2(0.54, 0.54),
  vec2(-0.07, -0.83), vec2(0.07, 0.83),
  vec2(0.0, 0.0)
);

float shadow = 0.0;
for (int i = 0; i < 9; i++) {
  vec2 offset = poissonDisk[i] * texelSize;
  shadow += textureSampleCompare(shadowMap, shadowSampler,
    shadowUV + offset, cascadeIndex, depth);
}
shadow /= 9.0;
```

Poisson disk breaks up grid aliasing patterns. Hardware comparison samplers on both backends:

- WebGL2: `sampler2DShadow` with `TEXTURE_COMPARE_MODE`
- WebGPU: `sampler_comparison` with `textureSampleCompareLevel()`

### Cascade Blending

Smooth transition between cascades to eliminate visible seams:

```glsl
float blendFactor = smoothstep(splitEnd - blendRange, splitEnd, viewDepth);
float shadow = mix(cascadeShadow[i], cascadeShadow[i + 1], blendFactor);
```

Blend zone: ~2-5 meters (configurable via `blendRange`). The `smoothstep` produces a gradual transition that is invisible in practice.

### Shadow Bias

Combined depth bias + slope-scaled bias:

```glsl
float bias = constantBias + slopeBias * (1.0 - NdotL);
bias = clamp(bias, 0.0, maxBias);
```

- `constantBias` (default 0.001): prevents acne on surfaces facing the light
- `slopeBias` (default 0.005): increases bias for surfaces at grazing angles (where acne is worst)
- Clamped to prevent peter-panning (shadows detaching from objects)

Additional acne reduction: **front-face culling during shadow pass** — render back faces to the depth map, so the depth sample is from the back face, naturally offsetting from the front face.

### Shadow Pass Rendering

For each of the 3 cascades:

1. Set render target to the cascade's texture array layer
2. Clear depth to 1.0
3. Set the cascade's shadow VP matrix as the camera
4. **Per-cascade frustum culling**: only render objects whose world AABB intersects the cascade's light frustum
5. Render with **position-only vertex shader** (minimal attribute fetching)
   - Exception: skinned meshes also need joint/weight attributes for skeletal transform
6. **Front-face culling**: render back faces to reduce acne
7. Extend light projection Z range backwards ~50-100m to catch shadow casters behind camera

### Per-Mesh Shadow Control

```typescript
mesh.castShadow = true // Include in shadow pass (default true)
mesh.receiveShadow = true // Sample shadow map in fragment shader (default true)
```

Objects with `castShadow: false` are skipped during shadow pass rendering. Objects with `receiveShadow: false` get `shadow = 1.0` (fully lit) in the fragment shader.

## Configuration API

```typescript
const engine = await createEngine(canvas, {
  shadows: {
    enabled: true,
    mapSize: 1024, // Per cascade (1024 default, 2048 for desktop)
    cascadeCount: 3, // Number of cascades
    maxDistance: 200, // Shadow far plane in world units
    lambda: 0.7, // Logarithmic split weight (0=linear, 1=logarithmic)
    bias: 0.001, // Constant depth bias
    slopeBias: 0.005, // Slope-scaled depth bias
    cascadeBlend: true, // Smooth cascade transitions
    blendRange: 2.0, // Blend zone in meters
  },
})

// Boolean shorthand uses all defaults:
const engine = await createEngine(canvas, { shadows: true })
```

## Performance Budget

| Pass                       | Cost (1024 resolution) |
| -------------------------- | ---------------------- |
| Shadow cascade 0           | ~0.5ms GPU             |
| Shadow cascade 1           | ~0.4ms GPU             |
| Shadow cascade 2           | ~0.3ms GPU             |
| PCF sampling (opaque pass) | ~0.3ms GPU             |
| **Total shadow overhead**  | **~1.5ms GPU**         |

Shadow pass is cheaper per cascade because it uses depth-only rendering (no color writes, no lighting, minimal fragment work). Cascade 2 is cheapest because fewer objects are visible at long range.
