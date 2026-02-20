// WGSL Shaders – GPU programs for the WebGPU renderer.
//
// Functionally identical to the GLSL shaders but written in WGSL (WebGPU Shading Language).
// See webgl-shaders.ts for detailed explanations of each shader's purpose.
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

// ─── Shadow depth shaders (vertex-only, no fragment) ─────────────────

export const SHADOW_DEPTH_WGSL = /* wgsl */ `
struct ShadowUniforms {
  lightVP: mat4x4<f32>,
};

struct ObjectUniforms {
  worldMatrix: mat4x4<f32>,
  normalMatrix: mat4x4<f32>,
};

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

struct ObjectUniforms {
  worldMatrix: mat4x4<f32>,
  normalMatrix: mat4x4<f32>,
  boneMatrices: array<mat4x4<f32>, 32>,
};

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
struct FrameUniforms {
  viewProjection: mat4x4<f32>,
  lightDir: vec3<f32>,
  lightIntensity: f32,
  lightColor: vec3<f32>,
  ambientIntensity: f32,
  ambientColor: vec3<f32>,
  shadowEnabled: f32,
  cascadeVP0: mat4x4<f32>,
  cascadeVP1: mat4x4<f32>,
  cascadeVP2: mat4x4<f32>,
  cascadeSplits: vec3<f32>,
  _pad0: f32,
  constantBias: f32,
  slopeBias: f32,
  invMapSize: f32,
  blendRange: f32,
};

struct PaletteEntry {
  color: vec4<f32>,
  emissive: vec4<f32>,
};

struct MaterialUniforms {
  baseColor: vec3<f32>,
  opacity: f32,
  hasPalette: f32,
  receiveShadow: f32,
  aoIntensity: f32,
  palette: array<PaletteEntry, 32>,
};

struct ObjectUniforms {
  worldMatrix: mat4x4<f32>,
  normalMatrix: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var shadowMap: texture_depth_2d_array;
@group(0) @binding(2) var shadowSampler: sampler_comparison;
@group(1) @binding(0) var<uniform> material: MaterialUniforms;
@group(2) @binding(0) var<uniform> object: ObjectUniforms;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) @interpolate(flat) materialIndex: i32,
};

@vertex
fn vs_main(
  @location(0) a_position: vec3<f32>,
  @location(1) a_normal: vec3<f32>,
  @location(2) a_uv: vec2<f32>,
  @location(3) a_materialIndex: f32,
) -> VertexOutput {
  var out: VertexOutput;
  let worldPos = object.worldMatrix * vec4<f32>(a_position, 1.0);
  out.worldPos = worldPos.xyz;
  out.normal = normalize((object.normalMatrix * vec4<f32>(a_normal, 0.0)).xyz);
  out.uv = a_uv;
  out.materialIndex = i32(a_materialIndex);
  out.position = frame.viewProjection * worldPos;
  return out;
}

fn pcf9(uv: vec2<f32>, cascade: i32, d: f32, ts: f32) -> f32 {
  var s = 0.0;
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(-ts, -ts), cascade, d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(0.0, -ts), cascade, d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(ts, -ts), cascade, d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(-ts, 0.0), cascade, d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv, cascade, d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(ts, 0.0), cascade, d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(-ts, ts), cascade, d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(0.0, ts), cascade, d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(ts, ts), cascade, d);
  return s / 9.0;
}

fn sampleShadow(worldPos: vec3<f32>, NdotL: f32) -> f32 {
  if (frame.shadowEnabled < 0.5) {
    return 1.0;
  }

  let viewDepth = abs((frame.viewProjection * vec4<f32>(worldPos, 1.0)).w);

  var cascade: i32 = 2;
  if (viewDepth < frame.cascadeSplits.x) {
    cascade = 0;
  } else if (viewDepth < frame.cascadeSplits.y) {
    cascade = 1;
  }

  var lightVP: mat4x4<f32>;
  if (cascade == 0) {
    lightVP = frame.cascadeVP0;
  } else if (cascade == 1) {
    lightVP = frame.cascadeVP1;
  } else {
    lightVP = frame.cascadeVP2;
  }

  let lightClip = lightVP * vec4<f32>(worldPos, 1.0);
  let uv = lightClip.xy * vec2<f32>(0.5, -0.5) + 0.5;
  let depth = lightClip.z;

  // Outside shadow map bounds → lit (prevents edge-clamping artifacts)
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || depth < 0.0 || depth > 1.0) {
    return 1.0;
  }

  let bias = frame.constantBias + frame.slopeBias * (1.0 - NdotL);
  let ts = frame.invMapSize;
  var shadow = pcf9(uv, cascade, depth - bias, ts);

  if (cascade < 2) {
    let splitDist = select(frame.cascadeSplits.y, frame.cascadeSplits.x, cascade == 0);
    let blendZone = splitDist * frame.blendRange;
    if (viewDepth > splitDist - blendZone) {
      var nextVP: mat4x4<f32>;
      if (cascade == 0) {
        nextVP = frame.cascadeVP1;
      } else {
        nextVP = frame.cascadeVP2;
      }
      let nextClip = nextVP * vec4<f32>(worldPos, 1.0);
      let nextUV = nextClip.xy * vec2<f32>(0.5, -0.5) + 0.5;
      var nextShadow = 1.0;
      if (nextUV.x >= 0.0 && nextUV.x <= 1.0 && nextUV.y >= 0.0 && nextUV.y <= 1.0 && nextClip.z >= 0.0 && nextClip.z <= 1.0) {
        nextShadow = pcf9(nextUV, cascade + 1, nextClip.z - bias, ts);
      }
      let t = smoothstep(splitDist - blendZone, splitDist, viewDepth);
      shadow = mix(shadow, nextShadow, t);
    }
  }

  return shadow;
}

struct FragmentOutput {
  @location(0) color: vec4<f32>,
  @location(1) emissive: vec4<f32>,
};

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
  var out: FragmentOutput;
  let normal = normalize(in.normal);
  var baseColor = material.baseColor;
  var alpha = material.opacity;
  var emissive = vec3<f32>(0.0, 0.0, 0.0);

  if (material.hasPalette > 0.5) {
    let idx = clamp(in.materialIndex, 0, 31);
    baseColor = material.palette[idx].color.rgb;
    alpha = material.palette[idx].color.a;
    emissive = material.palette[idx].emissive.rgb * material.palette[idx].emissive.a;
  }

  let ambient = frame.ambientColor * frame.ambientIntensity;
  let NdotL = max(dot(normal, frame.lightDir), 0.0);
  var shadow = 1.0;
  if (material.receiveShadow > 0.5) {
    shadow = sampleShadow(in.worldPos, NdotL);
  }
  let diffuse = frame.lightColor * frame.lightIntensity * NdotL * shadow;

  let litColor = baseColor * (ambient + diffuse);
  let finalColor = litColor + emissive;

  out.color = vec4<f32>(finalColor * alpha, alpha);
  out.emissive = vec4<f32>(emissive * alpha, alpha);
  return out;
}
`

export const LAMBERT_SKINNED_WGSL = /* wgsl */ `
struct FrameUniforms {
  viewProjection: mat4x4<f32>,
  lightDir: vec3<f32>,
  lightIntensity: f32,
  lightColor: vec3<f32>,
  ambientIntensity: f32,
  ambientColor: vec3<f32>,
  shadowEnabled: f32,
  cascadeVP0: mat4x4<f32>,
  cascadeVP1: mat4x4<f32>,
  cascadeVP2: mat4x4<f32>,
  cascadeSplits: vec3<f32>,
  _pad0: f32,
  constantBias: f32,
  slopeBias: f32,
  invMapSize: f32,
  blendRange: f32,
};

struct PaletteEntry {
  color: vec4<f32>,
  emissive: vec4<f32>,
};

struct MaterialUniforms {
  baseColor: vec3<f32>,
  opacity: f32,
  hasPalette: f32,
  receiveShadow: f32,
  aoIntensity: f32,
  palette: array<PaletteEntry, 32>,
};

struct ObjectUniforms {
  worldMatrix: mat4x4<f32>,
  normalMatrix: mat4x4<f32>,
  boneMatrices: array<mat4x4<f32>, 32>,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var shadowMap: texture_depth_2d_array;
@group(0) @binding(2) var shadowSampler: sampler_comparison;
@group(1) @binding(0) var<uniform> material: MaterialUniforms;
@group(2) @binding(0) var<uniform> object: ObjectUniforms;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) @interpolate(flat) materialIndex: i32,
};

@vertex
fn vs_main(
  @location(0) a_position: vec3<f32>,
  @location(1) a_normal: vec3<f32>,
  @location(2) a_uv: vec2<f32>,
  @location(3) a_materialIndex: f32,
  @location(4) a_joints: vec4<u32>,
  @location(5) a_weights: vec4<f32>,
) -> VertexOutput {
  var out: VertexOutput;

  let skinMatrix =
    a_weights.x * object.boneMatrices[a_joints.x] +
    a_weights.y * object.boneMatrices[a_joints.y] +
    a_weights.z * object.boneMatrices[a_joints.z] +
    a_weights.w * object.boneMatrices[a_joints.w];

  let skinnedPos = skinMatrix * vec4<f32>(a_position, 1.0);
  out.worldPos = skinnedPos.xyz;

  let skinnedNorm = skinMatrix * vec4<f32>(a_normal, 0.0);
  out.normal = normalize(skinnedNorm.xyz);
  out.uv = a_uv;
  out.materialIndex = i32(a_materialIndex);
  out.position = frame.viewProjection * skinnedPos;
  return out;
}

fn pcf9(uv: vec2<f32>, cascade: i32, d: f32, ts: f32) -> f32 {
  var s = 0.0;
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(-ts, -ts), cascade, d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(0.0, -ts), cascade, d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(ts, -ts), cascade, d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(-ts, 0.0), cascade, d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv, cascade, d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(ts, 0.0), cascade, d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(-ts, ts), cascade, d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(0.0, ts), cascade, d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(ts, ts), cascade, d);
  return s / 9.0;
}

fn sampleShadow(worldPos: vec3<f32>, NdotL: f32) -> f32 {
  if (frame.shadowEnabled < 0.5) {
    return 1.0;
  }

  let viewDepth = abs((frame.viewProjection * vec4<f32>(worldPos, 1.0)).w);

  var cascade: i32 = 2;
  if (viewDepth < frame.cascadeSplits.x) {
    cascade = 0;
  } else if (viewDepth < frame.cascadeSplits.y) {
    cascade = 1;
  }

  var lightVP: mat4x4<f32>;
  if (cascade == 0) {
    lightVP = frame.cascadeVP0;
  } else if (cascade == 1) {
    lightVP = frame.cascadeVP1;
  } else {
    lightVP = frame.cascadeVP2;
  }

  let lightClip = lightVP * vec4<f32>(worldPos, 1.0);
  let uv = lightClip.xy * vec2<f32>(0.5, -0.5) + 0.5;
  let depth = lightClip.z;

  // Outside shadow map bounds → lit (prevents edge-clamping artifacts)
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || depth < 0.0 || depth > 1.0) {
    return 1.0;
  }

  let bias = frame.constantBias + frame.slopeBias * (1.0 - NdotL);
  let ts = frame.invMapSize;
  var shadow = pcf9(uv, cascade, depth - bias, ts);

  if (cascade < 2) {
    let splitDist = select(frame.cascadeSplits.y, frame.cascadeSplits.x, cascade == 0);
    let blendZone = splitDist * frame.blendRange;
    if (viewDepth > splitDist - blendZone) {
      var nextVP: mat4x4<f32>;
      if (cascade == 0) {
        nextVP = frame.cascadeVP1;
      } else {
        nextVP = frame.cascadeVP2;
      }
      let nextClip = nextVP * vec4<f32>(worldPos, 1.0);
      let nextUV = nextClip.xy * vec2<f32>(0.5, -0.5) + 0.5;
      var nextShadow = 1.0;
      if (nextUV.x >= 0.0 && nextUV.x <= 1.0 && nextUV.y >= 0.0 && nextUV.y <= 1.0 && nextClip.z >= 0.0 && nextClip.z <= 1.0) {
        nextShadow = pcf9(nextUV, cascade + 1, nextClip.z - bias, ts);
      }
      let t = smoothstep(splitDist - blendZone, splitDist, viewDepth);
      shadow = mix(shadow, nextShadow, t);
    }
  }

  return shadow;
}

struct FragmentOutput {
  @location(0) color: vec4<f32>,
  @location(1) emissive: vec4<f32>,
};

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
  var out: FragmentOutput;
  let normal = normalize(in.normal);
  var baseColor = material.baseColor;
  var alpha = material.opacity;
  var emissive = vec3<f32>(0.0, 0.0, 0.0);

  if (material.hasPalette > 0.5) {
    let idx = clamp(in.materialIndex, 0, 31);
    baseColor = material.palette[idx].color.rgb;
    alpha = material.palette[idx].color.a;
    emissive = material.palette[idx].emissive.rgb * material.palette[idx].emissive.a;
  }

  let ambient = frame.ambientColor * frame.ambientIntensity;
  let NdotL = max(dot(normal, frame.lightDir), 0.0);
  var shadow = 1.0;
  if (material.receiveShadow > 0.5) {
    shadow = sampleShadow(in.worldPos, NdotL);
  }
  let diffuse = frame.lightColor * frame.lightIntensity * NdotL * shadow;

  let litColor = baseColor * (ambient + diffuse);
  let finalColor = litColor + emissive;

  out.color = vec4<f32>(finalColor * alpha, alpha);
  out.emissive = vec4<f32>(emissive * alpha, alpha);
  return out;
}
`

// ─── Textured Lambert shaders (with color map + AO map sampling) ─────

export const LAMBERT_TEXTURED_WGSL = /* wgsl */ `
struct FrameUniforms {
  viewProjection: mat4x4<f32>,
  lightDir: vec3<f32>,
  lightIntensity: f32,
  lightColor: vec3<f32>,
  ambientIntensity: f32,
  ambientColor: vec3<f32>,
  shadowEnabled: f32,
  cascadeVP0: mat4x4<f32>,
  cascadeVP1: mat4x4<f32>,
  cascadeVP2: mat4x4<f32>,
  cascadeSplits: vec3<f32>,
  _pad0: f32,
  constantBias: f32,
  slopeBias: f32,
  invMapSize: f32,
  blendRange: f32,
};

struct PaletteEntry {
  color: vec4<f32>,
  emissive: vec4<f32>,
};

struct MaterialUniforms {
  baseColor: vec3<f32>,
  opacity: f32,
  hasPalette: f32,
  receiveShadow: f32,
  aoIntensity: f32,
  palette: array<PaletteEntry, 32>,
};

struct ObjectUniforms {
  worldMatrix: mat4x4<f32>,
  normalMatrix: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var shadowMap: texture_depth_2d_array;
@group(0) @binding(2) var shadowSampler: sampler_comparison;
@group(1) @binding(0) var<uniform> material: MaterialUniforms;
@group(2) @binding(0) var<uniform> object: ObjectUniforms;
@group(3) @binding(0) var colorMapTexture: texture_2d<f32>;
@group(3) @binding(1) var aoMapTexture: texture_2d<f32>;
@group(3) @binding(2) var mapSampler: sampler;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) @interpolate(flat) materialIndex: i32,
};

@vertex
fn vs_main(
  @location(0) a_position: vec3<f32>,
  @location(1) a_normal: vec3<f32>,
  @location(2) a_uv: vec2<f32>,
  @location(3) a_materialIndex: f32,
) -> VertexOutput {
  var out: VertexOutput;
  let worldPos = object.worldMatrix * vec4<f32>(a_position, 1.0);
  out.worldPos = worldPos.xyz;
  out.normal = normalize((object.normalMatrix * vec4<f32>(a_normal, 0.0)).xyz);
  out.uv = a_uv;
  out.materialIndex = i32(a_materialIndex);
  out.position = frame.viewProjection * worldPos;
  return out;
}

fn pcf9(uv: vec2<f32>, cascade: i32, d: f32, ts: f32) -> f32 {
  var s = 0.0;
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(-ts, -ts), cascade, d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(0.0, -ts), cascade, d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(ts, -ts), cascade, d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(-ts, 0.0), cascade, d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv, cascade, d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(ts, 0.0), cascade, d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(-ts, ts), cascade, d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(0.0, ts), cascade, d);
  s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2<f32>(ts, ts), cascade, d);
  return s / 9.0;
}

fn sampleShadow(worldPos: vec3<f32>, NdotL: f32) -> f32 {
  if (frame.shadowEnabled < 0.5) {
    return 1.0;
  }

  let viewDepth = abs((frame.viewProjection * vec4<f32>(worldPos, 1.0)).w);

  var cascade: i32 = 2;
  if (viewDepth < frame.cascadeSplits.x) {
    cascade = 0;
  } else if (viewDepth < frame.cascadeSplits.y) {
    cascade = 1;
  }

  var lightVP: mat4x4<f32>;
  if (cascade == 0) {
    lightVP = frame.cascadeVP0;
  } else if (cascade == 1) {
    lightVP = frame.cascadeVP1;
  } else {
    lightVP = frame.cascadeVP2;
  }

  let lightClip = lightVP * vec4<f32>(worldPos, 1.0);
  let uv = lightClip.xy * vec2<f32>(0.5, -0.5) + 0.5;
  let depth = lightClip.z;

  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || depth < 0.0 || depth > 1.0) {
    return 1.0;
  }

  let bias = frame.constantBias + frame.slopeBias * (1.0 - NdotL);
  let ts = frame.invMapSize;
  var shadow = pcf9(uv, cascade, depth - bias, ts);

  if (cascade < 2) {
    let splitDist = select(frame.cascadeSplits.y, frame.cascadeSplits.x, cascade == 0);
    let blendZone = splitDist * frame.blendRange;
    if (viewDepth > splitDist - blendZone) {
      var nextVP: mat4x4<f32>;
      if (cascade == 0) {
        nextVP = frame.cascadeVP1;
      } else {
        nextVP = frame.cascadeVP2;
      }
      let nextClip = nextVP * vec4<f32>(worldPos, 1.0);
      let nextUV = nextClip.xy * vec2<f32>(0.5, -0.5) + 0.5;
      var nextShadow = 1.0;
      if (nextUV.x >= 0.0 && nextUV.x <= 1.0 && nextUV.y >= 0.0 && nextUV.y <= 1.0 && nextClip.z >= 0.0 && nextClip.z <= 1.0) {
        nextShadow = pcf9(nextUV, cascade + 1, nextClip.z - bias, ts);
      }
      let t = smoothstep(splitDist - blendZone, splitDist, viewDepth);
      shadow = mix(shadow, nextShadow, t);
    }
  }

  return shadow;
}

struct FragmentOutput {
  @location(0) color: vec4<f32>,
  @location(1) emissive: vec4<f32>,
};

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
  var out: FragmentOutput;
  let normal = normalize(in.normal);
  var baseColor = material.baseColor;
  var alpha = material.opacity;
  var emissive = vec3<f32>(0.0, 0.0, 0.0);

  if (material.hasPalette > 0.5) {
    let idx = clamp(in.materialIndex, 0, 31);
    baseColor = material.palette[idx].color.rgb;
    alpha = material.palette[idx].color.a;
    emissive = material.palette[idx].emissive.rgb * material.palette[idx].emissive.a;
  }

  // Sample texture maps (dummy white textures → identity when no map assigned)
  let texColor = textureSample(colorMapTexture, mapSampler, in.uv).rgb;
  baseColor = baseColor * texColor;

  let ao = mix(1.0, textureSample(aoMapTexture, mapSampler, in.uv).r, material.aoIntensity);
  let ambient = frame.ambientColor * frame.ambientIntensity * ao;

  let NdotL = max(dot(normal, frame.lightDir), 0.0);
  var shadow = 1.0;
  if (material.receiveShadow > 0.5) {
    shadow = sampleShadow(in.worldPos, NdotL);
  }
  let diffuse = frame.lightColor * frame.lightIntensity * NdotL * shadow;

  let litColor = baseColor * (ambient + diffuse);
  let finalColor = litColor + emissive;

  out.color = vec4<f32>(finalColor * alpha, alpha);
  out.emissive = vec4<f32>(emissive * alpha, alpha);
  return out;
}
`

// ─── Basic shaders (unlit, shadow bindings declared for shared BGL) ──

export const BASIC_SKINNED_WGSL = /* wgsl */ `
struct FrameUniforms {
  viewProjection: mat4x4<f32>,
  lightDir: vec3<f32>,
  lightIntensity: f32,
  lightColor: vec3<f32>,
  ambientIntensity: f32,
  ambientColor: vec3<f32>,
  shadowEnabled: f32,
  cascadeVP0: mat4x4<f32>,
  cascadeVP1: mat4x4<f32>,
  cascadeVP2: mat4x4<f32>,
  cascadeSplits: vec3<f32>,
  _pad0: f32,
  constantBias: f32,
  slopeBias: f32,
  invMapSize: f32,
  blendRange: f32,
};

struct PaletteEntry {
  color: vec4<f32>,
  emissive: vec4<f32>,
};

struct MaterialUniforms {
  baseColor: vec3<f32>,
  opacity: f32,
  hasPalette: f32,
  receiveShadow: f32,
  aoIntensity: f32,
  palette: array<PaletteEntry, 32>,
};

struct ObjectUniforms {
  worldMatrix: mat4x4<f32>,
  normalMatrix: mat4x4<f32>,
  boneMatrices: array<mat4x4<f32>, 32>,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var shadowMap: texture_depth_2d_array;
@group(0) @binding(2) var shadowSampler: sampler_comparison;
@group(1) @binding(0) var<uniform> material: MaterialUniforms;
@group(2) @binding(0) var<uniform> object: ObjectUniforms;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(
  @location(0) a_position: vec3<f32>,
  @location(1) a_normal: vec3<f32>,
  @location(2) a_uv: vec2<f32>,
  @location(3) a_materialIndex: f32,
  @location(4) a_joints: vec4<u32>,
  @location(5) a_weights: vec4<f32>,
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
struct FrameUniforms {
  viewProjection: mat4x4<f32>,
  lightDir: vec3<f32>,
  lightIntensity: f32,
  lightColor: vec3<f32>,
  ambientIntensity: f32,
  ambientColor: vec3<f32>,
  shadowEnabled: f32,
  cascadeVP0: mat4x4<f32>,
  cascadeVP1: mat4x4<f32>,
  cascadeVP2: mat4x4<f32>,
  cascadeSplits: vec3<f32>,
  _pad0: f32,
  constantBias: f32,
  slopeBias: f32,
  invMapSize: f32,
  blendRange: f32,
};

struct PaletteEntry {
  color: vec4<f32>,
  emissive: vec4<f32>,
};

struct MaterialUniforms {
  baseColor: vec3<f32>,
  opacity: f32,
  hasPalette: f32,
  receiveShadow: f32,
  aoIntensity: f32,
  palette: array<PaletteEntry, 32>,
};

struct ObjectUniforms {
  worldMatrix: mat4x4<f32>,
  normalMatrix: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var shadowMap: texture_depth_2d_array;
@group(0) @binding(2) var shadowSampler: sampler_comparison;
@group(1) @binding(0) var<uniform> material: MaterialUniforms;
@group(2) @binding(0) var<uniform> object: ObjectUniforms;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(
  @location(0) a_position: vec3<f32>,
  @location(1) a_normal: vec3<f32>,
  @location(2) a_uv: vec2<f32>,
  @location(3) a_materialIndex: f32,
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
