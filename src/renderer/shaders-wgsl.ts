// Inline WGSL shaders for WebGPU

export const LAMBERT_WGSL = /* wgsl */ `
struct FrameUniforms {
  viewProjection: mat4x4<f32>,
  lightDir: vec3<f32>,
  lightIntensity: f32,
  lightColor: vec3<f32>,
  ambientIntensity: f32,
  ambientColor: vec3<f32>,
};

struct PaletteEntry {
  color: vec4<f32>,
  emissive: vec4<f32>,
};

struct MaterialUniforms {
  baseColor: vec3<f32>,
  opacity: f32,
  hasPalette: f32,
  palette: array<PaletteEntry, 32>,
};

struct ObjectUniforms {
  worldMatrix: mat4x4<f32>,
  normalMatrix: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
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
  let diffuse = frame.lightColor * frame.lightIntensity * NdotL;

  let litColor = baseColor * (ambient + diffuse);
  let finalColor = litColor + emissive;

  out.color = vec4<f32>(finalColor, alpha);
  out.emissive = vec4<f32>(emissive, 1.0);
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
};

struct PaletteEntry {
  color: vec4<f32>,
  emissive: vec4<f32>,
};

struct MaterialUniforms {
  baseColor: vec3<f32>,
  opacity: f32,
  hasPalette: f32,
  palette: array<PaletteEntry, 32>,
};

struct ObjectUniforms {
  worldMatrix: mat4x4<f32>,
  normalMatrix: mat4x4<f32>,
  boneMatrices: array<mat4x4<f32>, 32>,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
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
  let worldPos = object.worldMatrix * skinnedPos;
  out.worldPos = worldPos.xyz;

  let skinnedNorm = skinMatrix * vec4<f32>(a_normal, 0.0);
  out.normal = normalize((object.normalMatrix * skinnedNorm).xyz);
  out.uv = a_uv;
  out.materialIndex = i32(a_materialIndex);
  out.position = frame.viewProjection * worldPos;
  return out;
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
  let diffuse = frame.lightColor * frame.lightIntensity * NdotL;

  let litColor = baseColor * (ambient + diffuse);
  let finalColor = litColor + emissive;

  out.color = vec4<f32>(finalColor, alpha);
  out.emissive = vec4<f32>(emissive, 1.0);
  return out;
}
`

export const BASIC_SKINNED_WGSL = /* wgsl */ `
struct FrameUniforms {
  viewProjection: mat4x4<f32>,
  lightDir: vec3<f32>,
  lightIntensity: f32,
  lightColor: vec3<f32>,
  ambientIntensity: f32,
  ambientColor: vec3<f32>,
};

struct PaletteEntry {
  color: vec4<f32>,
  emissive: vec4<f32>,
};

struct MaterialUniforms {
  baseColor: vec3<f32>,
  opacity: f32,
  hasPalette: f32,
  palette: array<PaletteEntry, 32>,
};

struct ObjectUniforms {
  worldMatrix: mat4x4<f32>,
  normalMatrix: mat4x4<f32>,
  boneMatrices: array<mat4x4<f32>, 32>,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
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
  out.position = frame.viewProjection * object.worldMatrix * skinnedPos;
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
  out.color = vec4<f32>(material.baseColor, material.opacity);
  out.emissive = vec4<f32>(0.0, 0.0, 0.0, 1.0);
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
};

struct PaletteEntry {
  color: vec4<f32>,
  emissive: vec4<f32>,
};

struct MaterialUniforms {
  baseColor: vec3<f32>,
  opacity: f32,
  hasPalette: f32,
  palette: array<PaletteEntry, 32>,
};

struct ObjectUniforms {
  worldMatrix: mat4x4<f32>,
  normalMatrix: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
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
  out.color = vec4<f32>(material.baseColor, material.opacity);
  out.emissive = vec4<f32>(0.0, 0.0, 0.0, 1.0);
  return out;
}
`

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
  // Gamma correction
  color = pow(color, vec3<f32>(1.0 / 2.2));
  return vec4<f32>(color, 1.0);
}
`
