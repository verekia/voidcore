// WGSL Shaders – GPU programs for the WebGPU renderer.
//
// Functionally identical to the GLSL shaders but written in WGSL (WebGPU Shading Language).
// See webgl-shaders.ts for detailed explanations of each shader's purpose.
//
// Shadows use a single depth texture covering the full camera frustum with PCF filtering.
//
// Transparency uses sorted back-to-front alpha blending with premultiplied alpha. Transparent
// meshes reuse the same shaders as opaque — all fragment outputs are premultiplied (rgb * alpha).
// The pipeline uses blend factors one / one-minus-src-alpha (avoiding src-alpha, which triggers
// VK_ERROR_UNKNOWN on some Android Vulkan drivers when combined with comparison sampling).
//
// Key WGSL differences from GLSL:
//   - Uses `@group(N) @binding(M)` instead of UBO layout bindings
//   - Structs instead of interface blocks for uniform data
//   - `var<uniform>` for uniform buffers, `var` for locals
//   - Built-in `@builtin(vertex_index)` replaces `gl_VertexID`
//   - Explicit return types on all functions
//
// Outlined Lambert meshes use a combined geometry buffer with doubled vertices: the first half
// contains original vertices (normal.w = 0), the second half contains outline vertices with
// smooth normals (normal.w = 1). The Lambert shader reads normal.w to distinguish main vs
// outline fragments. Outline vertices are inflated along their normal by the thickness stored
// in object.outlineColorAndThickness.w, and the fragment shader uses front_facing to discard
// front-facing outline triangles (keeping only back-facing silhouette fragments).
//
// Vertex-color (VC) variants read per-vertex color (unorm8x4) and emissive (float16x4)
// attributes, replacing the old palette uniform system. Non-VC shaders use material.baseColor
// directly with no emissive. The bakePalette() utility resolves palette entries into vertex
// attributes before rendering. VC variants also sample the AO map (group 3) so that meshes
// using bakePalette() can still benefit from ambient occlusion textures. VC shaders also
// support per-material tiled AO via a 2D array texture (group 3, bindings 3-5): the layer
// index is packed in color.a, intensity in emissive.a, and per-layer tiling scales in a
// uniform buffer. Tiling uses world-space XY coordinates; the tiled AO multiplies the
// world AO for a combined result. Per-material tiled normal maps use a second 2D array
// texture (group 3, binding 6) with per-vertex data in a float16x4 attribute (location 7):
// layerIndex/255 in x, intensity in y, scale in z. Normal mapping uses a cotangent frame
// (screen-space derivatives) for static meshes and triplanar projection for skinned meshes.

// ─── Shared WGSL blocks ──────────────────────────────────────────────

const FRAME_UNIFORMS = `
struct FrameUniforms {
  viewProjection: mat4x4<f32>,
  lightDir: vec3<f32>,
  lightIntensity: f32,
  lightColor: vec3<f32>,
  ambientIntensity: f32,
  ambientColor: vec3<f32>,
  shadowEnabled: f32,
  shadowVP: mat4x4<f32>,
  constantBias: f32,
  slopeBias: f32,
  invMapSize: f32,
  _pad0: f32,
};`

const MATERIAL_UNIFORMS = `
struct MaterialUniforms {
  baseColor: vec3<f32>,
  opacity: f32,
  receiveShadow: f32,
  aoIntensity: f32,
  emissiveBrightness: f32,
  _pad0: f32,
};`

const OBJECT_UNIFORMS = `
struct ObjectUniforms {
  worldMatrix: mat4x4<f32>,
  normalMatrix: mat4x4<f32>,
  outlineColorAndThickness: vec4<f32>,
};`

const SKINNED_OBJECT_UNIFORMS = `
struct ObjectUniforms {
  worldMatrix: mat4x4<f32>,
  normalMatrix: mat4x4<f32>,
  outlineColorAndThickness: vec4<f32>,
  boneMatrices: array<mat4x4<f32>, 32>,
};`

const FRAME_BINDINGS = `
@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var shadowMap: texture_depth_2d;
@group(0) @binding(2) var shadowSampler: sampler_comparison;`

const MATERIAL_BINDING = `
@group(1) @binding(0) var<uniform> material: MaterialUniforms;`

const OBJECT_BINDING = `
@group(2) @binding(0) var<uniform> object: ObjectUniforms;`

const PCF_AND_SHADOW = `
fn pcf9(uv: vec2<f32>, d: f32, ts: f32) -> f32 {
  var s = 0.0;
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(-ts, -ts), d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(0.0, -ts), d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(ts, -ts), d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(-ts, 0.0), d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv, d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(ts, 0.0), d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(-ts, ts), d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(0.0, ts), d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(ts, ts), d);
  return s / 9.0;
}

fn sampleShadow(worldPos: vec3<f32>, NdotL: f32) -> f32 {
  if (frame.shadowEnabled < 0.5) {
    return 1.0;
  }

  let lightClip = frame.shadowVP * vec4<f32>(worldPos, 1.0);
  let uv = lightClip.xy * vec2<f32>(0.5, -0.5) + 0.5;
  let depth = lightClip.z;

  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || depth < 0.0 || depth > 1.0) {
    return 1.0;
  }

  let bias = frame.constantBias + frame.slopeBias * (1.0 - NdotL);
  return pcf9(uv, depth - bias, frame.invMapSize);
}`

// ─── Shadow depth shaders (vertex-only, no fragment) ─────────────────

export const SHADOW_DEPTH_WGSL = /* wgsl */ `
struct ShadowUniforms {
  lightVP: mat4x4<f32>,
};
${OBJECT_UNIFORMS}

@group(0) @binding(0) var<uniform> shadow: ShadowUniforms;
@group(1) @binding(0) var<uniform> object: ObjectUniforms;

@vertex
fn vs_main(@location(0) a_position: vec3<f32>) -> @builtin(position) vec4<f32> {
  return shadow.lightVP * object.worldMatrix * vec4<f32>(a_position, 1.0);
}
`

export const SHADOW_DEPTH_SKINNED_WGSL = /* wgsl */ `
struct ShadowUniforms {
  lightVP: mat4x4<f32>,
};
${SKINNED_OBJECT_UNIFORMS}

@group(0) @binding(0) var<uniform> shadow: ShadowUniforms;
@group(1) @binding(0) var<uniform> object: ObjectUniforms;

@vertex
fn vs_main(
  @location(0) a_position: vec3<f32>,
  @location(1) a_joints: vec4<u32>,
  @location(2) a_weights: vec4<f32>,
) -> @builtin(position) vec4<f32> {
  let skinMatrix =
    a_weights.x * object.boneMatrices[a_joints.x] +
    a_weights.y * object.boneMatrices[a_joints.y] +
    a_weights.z * object.boneMatrices[a_joints.z] +
    a_weights.w * object.boneMatrices[a_joints.w];
  return shadow.lightVP * skinMatrix * vec4<f32>(a_position, 1.0);
}
`

// ─── Lambert shaders (with shadow sampling) ──────────────────────────

export const LAMBERT_WGSL = /* wgsl */ `
${FRAME_UNIFORMS}
${MATERIAL_UNIFORMS}
${OBJECT_UNIFORMS}

${FRAME_BINDINGS}
${MATERIAL_BINDING}
${OBJECT_BINDING}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) @interpolate(flat) outlineFlag: f32,
};

@vertex
fn vs_main(
  @location(0) a_position: vec3<f32>,
  @location(1) a_normal: vec4<f32>,
  @location(2) a_uv: vec2<f32>,
) -> VertexOutput {
  var out: VertexOutput;
  let flag = a_normal.w;
  out.outlineFlag = flag;
  var pos = a_position;
  if (flag > 0.5) {
    pos = pos + normalize(a_normal.xyz) * object.outlineColorAndThickness.w;
  }
  let worldPos = object.worldMatrix * vec4<f32>(pos, 1.0);
  out.worldPos = worldPos.xyz;
  out.normal = normalize((object.normalMatrix * vec4<f32>(a_normal.xyz, 0.0)).xyz);
  out.uv = a_uv;
  out.position = frame.viewProjection * worldPos;
  return out;
}

${PCF_AND_SHADOW}

struct FragmentOutput {
  @location(0) color: vec4<f32>,
  @location(1) emissive: vec4<f32>,
};

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) front_facing: bool) -> FragmentOutput {
  var out: FragmentOutput;

  if (in.outlineFlag > 0.5) {
    if (front_facing || object.outlineColorAndThickness.w <= 0.0) { discard; }
    out.color = vec4<f32>(object.outlineColorAndThickness.xyz, 1.0);
    out.emissive = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    return out;
  }

  let normal = normalize(in.normal);
  let baseColor = material.baseColor;
  let alpha = material.opacity;

  let ambient = frame.ambientColor * frame.ambientIntensity;
  let NdotL = max(dot(normal, frame.lightDir), 0.0);
  var shadow = 1.0;
  if (material.receiveShadow > 0.5) {
    shadow = sampleShadow(in.worldPos, NdotL);
  }
  let diffuse = frame.lightColor * frame.lightIntensity * NdotL * shadow;

  let finalColor = baseColor * (ambient + diffuse);

  out.color = vec4<f32>(finalColor * alpha, alpha);
  out.emissive = vec4<f32>(0.0, 0.0, 0.0, alpha);
  return out;
}
`

export const LAMBERT_SKINNED_WGSL = /* wgsl */ `
${FRAME_UNIFORMS}
${MATERIAL_UNIFORMS}
${SKINNED_OBJECT_UNIFORMS}

${FRAME_BINDINGS}
${MATERIAL_BINDING}
${OBJECT_BINDING}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) @interpolate(flat) outlineFlag: f32,
};

@vertex
fn vs_main(
  @location(0) a_position: vec3<f32>,
  @location(1) a_normal: vec4<f32>,
  @location(2) a_uv: vec2<f32>,
  @location(3) a_joints: vec4<u32>,
  @location(4) a_weights: vec4<f32>,
) -> VertexOutput {
  var out: VertexOutput;
  let flag = a_normal.w;
  out.outlineFlag = flag;

  let skinMatrix =
    a_weights.x * object.boneMatrices[a_joints.x] +
    a_weights.y * object.boneMatrices[a_joints.y] +
    a_weights.z * object.boneMatrices[a_joints.z] +
    a_weights.w * object.boneMatrices[a_joints.w];

  let skinnedPos = skinMatrix * vec4<f32>(a_position, 1.0);
  let skinnedNorm = normalize((skinMatrix * vec4<f32>(a_normal.xyz, 0.0)).xyz);

  var worldPos = skinnedPos.xyz;
  if (flag > 0.5) {
    worldPos = worldPos + skinnedNorm * object.outlineColorAndThickness.w;
  }
  out.worldPos = worldPos;
  out.normal = skinnedNorm;
  out.uv = a_uv;
  out.position = frame.viewProjection * vec4<f32>(worldPos, 1.0);
  return out;
}

${PCF_AND_SHADOW}

struct FragmentOutput {
  @location(0) color: vec4<f32>,
  @location(1) emissive: vec4<f32>,
};

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) front_facing: bool) -> FragmentOutput {
  var out: FragmentOutput;

  if (in.outlineFlag > 0.5) {
    if (front_facing || object.outlineColorAndThickness.w <= 0.0) { discard; }
    out.color = vec4<f32>(object.outlineColorAndThickness.xyz, 1.0);
    out.emissive = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    return out;
  }

  let normal = normalize(in.normal);
  let baseColor = material.baseColor;
  let alpha = material.opacity;

  let ambient = frame.ambientColor * frame.ambientIntensity;
  let NdotL = max(dot(normal, frame.lightDir), 0.0);
  var shadow = 1.0;
  if (material.receiveShadow > 0.5) {
    shadow = sampleShadow(in.worldPos, NdotL);
  }
  let diffuse = frame.lightColor * frame.lightIntensity * NdotL * shadow;

  let finalColor = baseColor * (ambient + diffuse);

  out.color = vec4<f32>(finalColor * alpha, alpha);
  out.emissive = vec4<f32>(0.0, 0.0, 0.0, alpha);
  return out;
}
`

// ─── Vertex-color Lambert shaders ────────────────────────────────────

export const LAMBERT_VC_WGSL = /* wgsl */ `
${FRAME_UNIFORMS}
${MATERIAL_UNIFORMS}
${OBJECT_UNIFORMS}

${FRAME_BINDINGS}
${MATERIAL_BINDING}
${OBJECT_BINDING}
@group(3) @binding(0) var colorMapTexture: texture_2d<f32>;
@group(3) @binding(1) var aoMapTexture: texture_2d<f32>;
@group(3) @binding(2) var mapSampler: sampler;
@group(3) @binding(3) var tiledAoArray: texture_2d_array<f32>;
@group(3) @binding(4) var tiledAoSampler: sampler;
@group(3) @binding(5) var<uniform> tiledAoScales: array<vec4<f32>, 4>;
@group(3) @binding(6) var tiledNormalArray: texture_2d_array<f32>;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) @interpolate(flat) outlineFlag: f32,
  @location(4) color: vec4<f32>,
  @location(5) vcEmissive: vec4<f32>,
  @location(6) tiledNormalInfo: vec4<f32>,
};

@vertex
fn vs_main(
  @location(0) a_position: vec3<f32>,
  @location(1) a_normal: vec4<f32>,
  @location(2) a_uv: vec2<f32>,
  @location(3) a_color: vec4<f32>,
  @location(6) a_emissive: vec4<f32>,
  @location(7) a_tiledNormal: vec4<f32>,
) -> VertexOutput {
  var out: VertexOutput;
  let flag = a_normal.w;
  out.outlineFlag = flag;
  var pos = a_position;
  if (flag > 0.5) {
    pos = pos + normalize(a_normal.xyz) * object.outlineColorAndThickness.w;
  }
  let worldPos = object.worldMatrix * vec4<f32>(pos, 1.0);
  out.worldPos = worldPos.xyz;
  out.normal = normalize((object.normalMatrix * vec4<f32>(a_normal.xyz, 0.0)).xyz);
  out.uv = a_uv;
  out.color = a_color;
  out.vcEmissive = a_emissive;
  out.tiledNormalInfo = a_tiledNormal;
  out.position = frame.viewProjection * worldPos;
  return out;
}

${PCF_AND_SHADOW}

fn getTiledAoScale(index: u32) -> f32 {
  return tiledAoScales[index / 4u][index % 4u];
}

struct FragmentOutput {
  @location(0) color: vec4<f32>,
  @location(1) emissive: vec4<f32>,
};

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) front_facing: bool) -> FragmentOutput {
  var out: FragmentOutput;

  // Sample textures before any non-uniform branching (WGSL requires uniform control flow for textureSample)
  let aoSample = textureSample(aoMapTexture, mapSampler, in.uv).r;

  // Tiled AO: decode layer index from color.a, sample array texture using mesh UVs
  let tiledAoLayerRaw = u32(round(in.color.a * 255.0));
  let tiledAoIntensity = in.vcEmissive.a;
  // Sample at layer 0 for uniform control flow, result discarded when layer=0
  let tiledAoLayer = select(0u, tiledAoLayerRaw - 1u, tiledAoLayerRaw > 0u);
  let tiledAoScale = getTiledAoScale(tiledAoLayer);
  let tiledAoSample = textureSample(tiledAoArray, tiledAoSampler, in.uv * tiledAoScale, tiledAoLayer).r;
  let tiledAo = select(1.0, mix(1.0, tiledAoSample, tiledAoIntensity), tiledAoLayerRaw > 0u);

  // Tiled normals: decode layer, sample before non-uniform branch
  let tnLayerRaw = u32(round(in.tiledNormalInfo.x * 255.0));
  let tnIntensity = in.tiledNormalInfo.y;
  let tnScale = in.tiledNormalInfo.z;
  let tnLayer = select(0u, tnLayerRaw - 1u, tnLayerRaw > 0u);
  let tnUV = in.uv * tnScale;
  let tnSample = textureSample(tiledNormalArray, tiledAoSampler, tnUV, tnLayer).rgb;
  // Also compute screen-space derivatives before the discard branch
  let dWorldPosDx = dpdx(in.worldPos);
  let dWorldPosDy = dpdy(in.worldPos);
  let dUVdx = dpdx(tnUV);
  let dUVdy = dpdy(tnUV);

  if (in.outlineFlag > 0.5) {
    if (front_facing || object.outlineColorAndThickness.w <= 0.0) { discard; }
    out.color = vec4<f32>(object.outlineColorAndThickness.xyz, 1.0);
    out.emissive = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    return out;
  }

  var normal = normalize(in.normal);

  // Apply tiled normal map via cotangent frame
  if (tnLayerRaw > 0u) {
    // Decode tangent-space normal from [0,1] to [-1,1]
    let tnNormal = tnSample * 2.0 - 1.0;
    // Build cotangent frame from screen-space derivatives
    let dp1 = dWorldPosDx;
    let dp2 = dWorldPosDy;
    let duv1 = dUVdx;
    let duv2 = dUVdy;
    let dp2perp = cross(dp2, normal);
    let dp1perp = cross(normal, dp1);
    let T = dp2perp * duv1.x + dp1perp * duv2.x;
    let B = dp2perp * duv1.y + dp1perp * duv2.y;
    let invmax = inverseSqrt(max(dot(T, T), dot(B, B)));
    let tbn_T = T * invmax;
    let tbn_B = B * invmax;
    let perturbedNormal = normalize(tbn_T * tnNormal.x + tbn_B * tnNormal.y + normal * tnNormal.z);
    normal = normalize(mix(normal, perturbedNormal, tnIntensity));
  }

  let baseColor = material.baseColor * in.color.rgb;
  let alpha = material.opacity;
  let emissive = in.vcEmissive.rgb;

  let ao = mix(1.0, aoSample, material.aoIntensity) * tiledAo;
  let ambient = frame.ambientColor * frame.ambientIntensity * ao;
  let NdotL = max(dot(normal, frame.lightDir), 0.0);
  var shadow = 1.0;
  if (material.receiveShadow > 0.5) {
    shadow = sampleShadow(in.worldPos, NdotL);
  }
  let diffuse = frame.lightColor * frame.lightIntensity * NdotL * shadow;

  let litColor = baseColor * (ambient + diffuse);
  let brightness = dot(emissive, vec3<f32>(0.299, 0.587, 0.114));
  let screenEmissive = mix(emissive, vec3<f32>(brightness), saturate(brightness) * material.emissiveBrightness);
  let finalColor = litColor + screenEmissive;

  out.color = vec4<f32>(finalColor * alpha, alpha);
  out.emissive = vec4<f32>(emissive * alpha, alpha);
  return out;
}
`

export const LAMBERT_SKINNED_VC_WGSL = /* wgsl */ `
${FRAME_UNIFORMS}
${MATERIAL_UNIFORMS}
${SKINNED_OBJECT_UNIFORMS}

${FRAME_BINDINGS}
${MATERIAL_BINDING}
${OBJECT_BINDING}
@group(3) @binding(0) var colorMapTexture: texture_2d<f32>;
@group(3) @binding(1) var aoMapTexture: texture_2d<f32>;
@group(3) @binding(2) var mapSampler: sampler;
@group(3) @binding(3) var tiledAoArray: texture_2d_array<f32>;
@group(3) @binding(4) var tiledAoSampler: sampler;
@group(3) @binding(5) var<uniform> tiledAoScales: array<vec4<f32>, 4>;
@group(3) @binding(6) var tiledNormalArray: texture_2d_array<f32>;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) @interpolate(flat) outlineFlag: f32,
  @location(4) color: vec4<f32>,
  @location(5) vcEmissive: vec4<f32>,
  @location(6) tiledNormalInfo: vec4<f32>,
};

@vertex
fn vs_main(
  @location(0) a_position: vec3<f32>,
  @location(1) a_normal: vec4<f32>,
  @location(2) a_uv: vec2<f32>,
  @location(3) a_color: vec4<f32>,
  @location(4) a_joints: vec4<u32>,
  @location(5) a_weights: vec4<f32>,
  @location(6) a_emissive: vec4<f32>,
  @location(7) a_tiledNormal: vec4<f32>,
) -> VertexOutput {
  var out: VertexOutput;
  let flag = a_normal.w;
  out.outlineFlag = flag;

  let skinMatrix =
    a_weights.x * object.boneMatrices[a_joints.x] +
    a_weights.y * object.boneMatrices[a_joints.y] +
    a_weights.z * object.boneMatrices[a_joints.z] +
    a_weights.w * object.boneMatrices[a_joints.w];

  let skinnedPos = skinMatrix * vec4<f32>(a_position, 1.0);
  let skinnedNorm = normalize((skinMatrix * vec4<f32>(a_normal.xyz, 0.0)).xyz);

  var worldPos = skinnedPos.xyz;
  if (flag > 0.5) {
    worldPos = worldPos + skinnedNorm * object.outlineColorAndThickness.w;
  }
  out.worldPos = worldPos;
  out.normal = skinnedNorm;
  out.uv = a_uv;
  out.color = a_color;
  out.vcEmissive = a_emissive;
  out.tiledNormalInfo = a_tiledNormal;
  out.position = frame.viewProjection * vec4<f32>(worldPos, 1.0);
  return out;
}

${PCF_AND_SHADOW}

fn getTiledAoScale(index: u32) -> f32 {
  return tiledAoScales[index / 4u][index % 4u];
}

struct FragmentOutput {
  @location(0) color: vec4<f32>,
  @location(1) emissive: vec4<f32>,
};

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) front_facing: bool) -> FragmentOutput {
  var out: FragmentOutput;

  // Sample textures before any non-uniform branching (WGSL requires uniform control flow for textureSample)
  let aoSample = textureSample(aoMapTexture, mapSampler, in.uv).r;

  // Tiled AO: decode layer index from color.a, triplanar sample array texture
  let tiledAoLayerRaw = u32(round(in.color.a * 255.0));
  let tiledAoIntensity = in.vcEmissive.a;
  let tiledAoLayer = select(0u, tiledAoLayerRaw - 1u, tiledAoLayerRaw > 0u);
  let tiledAoScale = getTiledAoScale(tiledAoLayer);
  // Triplanar sampling: project from all 3 axes to avoid stretching on vertical surfaces
  let taoXY = textureSample(tiledAoArray, tiledAoSampler, in.worldPos.xy * tiledAoScale, tiledAoLayer).r;
  let taoXZ = textureSample(tiledAoArray, tiledAoSampler, in.worldPos.xz * tiledAoScale, tiledAoLayer).r;
  let taoYZ = textureSample(tiledAoArray, tiledAoSampler, in.worldPos.yz * tiledAoScale, tiledAoLayer).r;
  let triN = abs(normalize(in.normal));
  let tiledAoSample = taoXY * triN.z + taoXZ * triN.y + taoYZ * triN.x;
  let tiledAo = select(1.0, mix(1.0, tiledAoSample, tiledAoIntensity), tiledAoLayerRaw > 0u);

  // Tiled normals: triplanar sample before non-uniform branch
  let tnLayerRaw = u32(round(in.tiledNormalInfo.x * 255.0));
  let tnIntensity = in.tiledNormalInfo.y;
  let tnScale = in.tiledNormalInfo.z;
  let tnLayer = select(0u, tnLayerRaw - 1u, tnLayerRaw > 0u);
  // Triplanar normal sampling from 3 axes
  let tnXY = textureSample(tiledNormalArray, tiledAoSampler, in.worldPos.xy * tnScale, tnLayer).rgb;
  let tnXZ = textureSample(tiledNormalArray, tiledAoSampler, in.worldPos.xz * tnScale, tnLayer).rgb;
  let tnYZ = textureSample(tiledNormalArray, tiledAoSampler, in.worldPos.yz * tnScale, tnLayer).rgb;

  if (in.outlineFlag > 0.5) {
    if (front_facing || object.outlineColorAndThickness.w <= 0.0) { discard; }
    out.color = vec4<f32>(object.outlineColorAndThickness.xyz, 1.0);
    out.emissive = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    return out;
  }

  var normal = normalize(in.normal);

  // Apply tiled normal map via triplanar + fixed per-axis TBN
  if (tnLayerRaw > 0u) {
    let triW = abs(normal);
    // Decode tangent-space normals from [0,1] to [-1,1]
    let nXY = tnXY * 2.0 - 1.0;
    let nXZ = tnXZ * 2.0 - 1.0;
    let nYZ = tnYZ * 2.0 - 1.0;
    // Fixed TBN per projection axis (Z-up coordinate system)
    // XY projection: T=+X, B=+Y, N=+Z
    let pXY = normalize(vec3<f32>(nXY.x, nXY.y, nXY.z) * vec3<f32>(1.0, 1.0, sign(normal.z + 0.001)));
    // XZ projection: T=+X, B=+Z, N=+Y
    let pXZ = normalize(vec3<f32>(nXZ.x, nXZ.z, nXZ.y) * vec3<f32>(1.0, sign(normal.y + 0.001), 1.0));
    // YZ projection: T=+Y, B=+Z, N=+X
    let pYZ = normalize(vec3<f32>(nYZ.z, nYZ.x, nYZ.y) * vec3<f32>(sign(normal.x + 0.001), 1.0, 1.0));
    let blended = normalize(pXY * triW.z + pXZ * triW.y + pYZ * triW.x);
    normal = normalize(mix(normal, blended, tnIntensity));
  }

  let baseColor = material.baseColor * in.color.rgb;
  let alpha = material.opacity;
  let emissive = in.vcEmissive.rgb;

  let ao = mix(1.0, aoSample, material.aoIntensity) * tiledAo;
  let ambient = frame.ambientColor * frame.ambientIntensity * ao;
  let NdotL = max(dot(normal, frame.lightDir), 0.0);
  var shadow = 1.0;
  if (material.receiveShadow > 0.5) {
    shadow = sampleShadow(in.worldPos, NdotL);
  }
  let diffuse = frame.lightColor * frame.lightIntensity * NdotL * shadow;

  let litColor = baseColor * (ambient + diffuse);
  let brightness = dot(emissive, vec3<f32>(0.299, 0.587, 0.114));
  let screenEmissive = mix(emissive, vec3<f32>(brightness), saturate(brightness) * material.emissiveBrightness);
  let finalColor = litColor + screenEmissive;

  out.color = vec4<f32>(finalColor * alpha, alpha);
  out.emissive = vec4<f32>(emissive * alpha, alpha);
  return out;
}
`

// ─── Textured Lambert shaders (with color map + AO map sampling) ─────

export const LAMBERT_TEXTURED_WGSL = /* wgsl */ `
${FRAME_UNIFORMS}
${MATERIAL_UNIFORMS}
${OBJECT_UNIFORMS}

${FRAME_BINDINGS}
${MATERIAL_BINDING}
${OBJECT_BINDING}
@group(3) @binding(0) var colorMapTexture: texture_2d<f32>;
@group(3) @binding(1) var aoMapTexture: texture_2d<f32>;
@group(3) @binding(2) var mapSampler: sampler;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) @interpolate(flat) outlineFlag: f32,
};

@vertex
fn vs_main(
  @location(0) a_position: vec3<f32>,
  @location(1) a_normal: vec4<f32>,
  @location(2) a_uv: vec2<f32>,
) -> VertexOutput {
  var out: VertexOutput;
  let flag = a_normal.w;
  out.outlineFlag = flag;
  var pos = a_position;
  if (flag > 0.5) {
    pos = pos + normalize(a_normal.xyz) * object.outlineColorAndThickness.w;
  }
  let worldPos = object.worldMatrix * vec4<f32>(pos, 1.0);
  out.worldPos = worldPos.xyz;
  out.normal = normalize((object.normalMatrix * vec4<f32>(a_normal.xyz, 0.0)).xyz);
  out.uv = a_uv;
  out.position = frame.viewProjection * worldPos;
  return out;
}

${PCF_AND_SHADOW}

struct FragmentOutput {
  @location(0) color: vec4<f32>,
  @location(1) emissive: vec4<f32>,
};

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) front_facing: bool) -> FragmentOutput {
  var out: FragmentOutput;

  // Sample textures before any non-uniform branching (WGSL requires uniform control flow for textureSample)
  let texColor = textureSample(colorMapTexture, mapSampler, in.uv).rgb;
  let aoSample = textureSample(aoMapTexture, mapSampler, in.uv).r;

  if (in.outlineFlag > 0.5) {
    if (front_facing || object.outlineColorAndThickness.w <= 0.0) { discard; }
    out.color = vec4<f32>(object.outlineColorAndThickness.xyz, 1.0);
    out.emissive = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    return out;
  }

  let normal = normalize(in.normal);
  var baseColor = material.baseColor * texColor;
  let alpha = material.opacity;

  let ao = mix(1.0, aoSample, material.aoIntensity);
  let ambient = frame.ambientColor * frame.ambientIntensity * ao;

  let NdotL = max(dot(normal, frame.lightDir), 0.0);
  var shadow = 1.0;
  if (material.receiveShadow > 0.5) {
    shadow = sampleShadow(in.worldPos, NdotL);
  }
  let diffuse = frame.lightColor * frame.lightIntensity * NdotL * shadow;

  let finalColor = baseColor * (ambient + diffuse);

  out.color = vec4<f32>(finalColor * alpha, alpha);
  out.emissive = vec4<f32>(0.0, 0.0, 0.0, alpha);
  return out;
}
`

// ─── Basic shaders (unlit, shadow bindings declared for shared BGL) ──

export const BASIC_SKINNED_WGSL = /* wgsl */ `
${FRAME_UNIFORMS}
${MATERIAL_UNIFORMS}
${SKINNED_OBJECT_UNIFORMS}

${FRAME_BINDINGS}
${MATERIAL_BINDING}
${OBJECT_BINDING}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(
  @location(0) a_position: vec3<f32>,
  @location(1) a_normal: vec3<f32>,
  @location(2) a_uv: vec2<f32>,
  @location(3) a_joints: vec4<u32>,
  @location(4) a_weights: vec4<f32>,
) -> VertexOutput {
  var out: VertexOutput;

  let skinMatrix =
    a_weights.x * object.boneMatrices[a_joints.x] +
    a_weights.y * object.boneMatrices[a_joints.y] +
    a_weights.z * object.boneMatrices[a_joints.z] +
    a_weights.w * object.boneMatrices[a_joints.w];

  let skinnedPos = skinMatrix * vec4<f32>(a_position, 1.0);
  out.position = frame.viewProjection * skinnedPos;
  out.uv = a_uv;
  return out;
}

struct FragmentOutput {
  @location(0) color: vec4<f32>,
  @location(1) emissive: vec4<f32>,
};

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
  var out: FragmentOutput;
  out.color = vec4<f32>(material.baseColor * material.opacity, material.opacity);
  out.emissive = vec4<f32>(0.0, 0.0, 0.0, material.opacity);
  return out;
}
`

export const BASIC_WGSL = /* wgsl */ `
${FRAME_UNIFORMS}
${MATERIAL_UNIFORMS}
${OBJECT_UNIFORMS}

${FRAME_BINDINGS}
${MATERIAL_BINDING}
${OBJECT_BINDING}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(
  @location(0) a_position: vec3<f32>,
  @location(1) a_normal: vec3<f32>,
  @location(2) a_uv: vec2<f32>,
) -> VertexOutput {
  var out: VertexOutput;
  out.position = frame.viewProjection * object.worldMatrix * vec4<f32>(a_position, 1.0);
  out.uv = a_uv;
  return out;
}

struct FragmentOutput {
  @location(0) color: vec4<f32>,
  @location(1) emissive: vec4<f32>,
};

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
  var out: FragmentOutput;
  out.color = vec4<f32>(material.baseColor * material.opacity, material.opacity);
  out.emissive = vec4<f32>(0.0, 0.0, 0.0, material.opacity);
  return out;
}
`

// ─── Post-processing shaders (unchanged) ─────────────────────────────

export const BLOOM_DOWN_WGSL = /* wgsl */ `
struct Params {
  texelSize: vec2<f32>,
  useKarisAverage: f32,
};

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var srcTexture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: Params;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var out: VertexOutput;
  let x = f32((vertexIndex & 1u) << 2u);
  let y = f32((vertexIndex & 2u) << 1u);
  out.uv = vec2<f32>(x * 0.5, 1.0 - y * 0.5);
  out.position = vec4<f32>(x - 1.0, y - 1.0, 0.0, 1.0);
  return out;
}

fn karisWeight(c: vec3<f32>) -> f32 {
  let luma = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
  return 1.0 / (1.0 + luma);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let ts = params.texelSize;

  // 13-tap Jimenez downsample
  let a = textureSample(srcTexture, texSampler, in.uv + vec2<f32>(-2.0, -2.0) * ts).rgb;
  let b = textureSample(srcTexture, texSampler, in.uv + vec2<f32>( 0.0, -2.0) * ts).rgb;
  let c = textureSample(srcTexture, texSampler, in.uv + vec2<f32>( 2.0, -2.0) * ts).rgb;
  let d = textureSample(srcTexture, texSampler, in.uv + vec2<f32>(-2.0,  0.0) * ts).rgb;
  let e = textureSample(srcTexture, texSampler, in.uv).rgb;
  let f = textureSample(srcTexture, texSampler, in.uv + vec2<f32>( 2.0,  0.0) * ts).rgb;
  let g = textureSample(srcTexture, texSampler, in.uv + vec2<f32>(-2.0,  2.0) * ts).rgb;
  let h = textureSample(srcTexture, texSampler, in.uv + vec2<f32>( 0.0,  2.0) * ts).rgb;
  let i = textureSample(srcTexture, texSampler, in.uv + vec2<f32>( 2.0,  2.0) * ts).rgb;
  let j = textureSample(srcTexture, texSampler, in.uv + vec2<f32>(-1.0, -1.0) * ts).rgb;
  let k = textureSample(srcTexture, texSampler, in.uv + vec2<f32>( 1.0, -1.0) * ts).rgb;
  let l = textureSample(srcTexture, texSampler, in.uv + vec2<f32>(-1.0,  1.0) * ts).rgb;
  let m = textureSample(srcTexture, texSampler, in.uv + vec2<f32>( 1.0,  1.0) * ts).rgb;

  var result: vec3<f32>;
  if (params.useKarisAverage > 0.5) {
    // Karis average for first downsample to prevent fireflies
    let g0 = (a + b + d + e) * 0.25;
    let g1 = (b + c + e + f) * 0.25;
    let g2 = (d + e + g + h) * 0.25;
    let g3 = (e + f + h + i) * 0.25;
    let g4 = (j + k + l + m) * 0.25;
    let w0 = karisWeight(g0);
    let w1 = karisWeight(g1);
    let w2 = karisWeight(g2);
    let w3 = karisWeight(g3);
    let w4 = karisWeight(g4);
    let wSum = w0 + w1 + w2 + w3 + w4;
    result = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / wSum;
  } else {
    result = e * 0.125;
    result += (a + c + g + i) * 0.03125;
    result += (b + d + f + h) * 0.0625;
    result += (j + k + l + m) * 0.125;
  }

  return vec4<f32>(result, 1.0);
}
`

export const BLOOM_UP_WGSL = /* wgsl */ `
struct Params {
  texelSize: vec2<f32>,
};

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var srcTexture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: Params;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var out: VertexOutput;
  let x = f32((vertexIndex & 1u) << 2u);
  let y = f32((vertexIndex & 2u) << 1u);
  out.uv = vec2<f32>(x * 0.5, 1.0 - y * 0.5);
  out.position = vec4<f32>(x - 1.0, y - 1.0, 0.0, 1.0);
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let ts = params.texelSize;
  // 9-tap tent filter
  let result = textureSample(srcTexture, texSampler, in.uv + vec2<f32>(-1.0, -1.0) * ts).rgb * 1.0
             + textureSample(srcTexture, texSampler, in.uv + vec2<f32>( 0.0, -1.0) * ts).rgb * 2.0
             + textureSample(srcTexture, texSampler, in.uv + vec2<f32>( 1.0, -1.0) * ts).rgb * 1.0
             + textureSample(srcTexture, texSampler, in.uv + vec2<f32>(-1.0,  0.0) * ts).rgb * 2.0
             + textureSample(srcTexture, texSampler, in.uv).rgb * 4.0
             + textureSample(srcTexture, texSampler, in.uv + vec2<f32>( 1.0,  0.0) * ts).rgb * 2.0
             + textureSample(srcTexture, texSampler, in.uv + vec2<f32>(-1.0,  1.0) * ts).rgb * 1.0
             + textureSample(srcTexture, texSampler, in.uv + vec2<f32>( 0.0,  1.0) * ts).rgb * 2.0
             + textureSample(srcTexture, texSampler, in.uv + vec2<f32>( 1.0,  1.0) * ts).rgb * 1.0;
  return vec4<f32>(result / 16.0, 1.0);
}
`

export const BLIT_WGSL = /* wgsl */ `
struct Params {
  bloomIntensity: f32,
};

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var sceneTexture: texture_2d<f32>;
@group(0) @binding(2) var bloomTexture: texture_2d<f32>;
@group(0) @binding(3) var<uniform> params: Params;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var out: VertexOutput;
  let x = f32((vertexIndex & 1u) << 2u);
  let y = f32((vertexIndex & 2u) << 1u);
  out.uv = vec2<f32>(x * 0.5, 1.0 - y * 0.5);
  out.position = vec4<f32>(x - 1.0, y - 1.0, 0.0, 1.0);
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let scene = textureSample(sceneTexture, texSampler, in.uv).rgb;
  let bloom = textureSample(bloomTexture, texSampler, in.uv).rgb;
  var color = scene + bloom * params.bloomIntensity;
  return vec4<f32>(color, 1.0);
}
`
