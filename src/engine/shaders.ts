// WGSL shaders for WebGPU renderer — single source, no duplicate declarations

export const shaderSource = /* wgsl */ `
struct CameraUniforms {
  view: mat4x4f,
  projection: mat4x4f,
}

struct ModelUniforms {
  world: mat4x4f,
  color: vec4f,
  flags: vec4f, // x = unlit
}

struct LightUniforms {
  direction: vec4f,
  color: vec4f,
  ambient: vec4f,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<uniform> model: ModelUniforms;
@group(2) @binding(0) var<uniform> light: LightUniforms;

struct VertexInput {
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) vertColor: vec3f,
  @location(3) bloom: f32,
}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) worldNormal: vec3f,
  @location(1) color: vec4f,
  @location(2) unlit: f32,
  @location(3) vertColor: vec3f,
  @location(4) bloom: f32,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;

  let worldPos = model.world * vec4f(input.position, 1.0);
  output.position = camera.projection * camera.view * worldPos;

  // Transform normal by upper 3x3 of world matrix
  let normalMat = mat3x3f(
    model.world[0].xyz,
    model.world[1].xyz,
    model.world[2].xyz,
  );
  output.worldNormal = normalize(normalMat * input.normal);
  output.color = model.color;
  output.unlit = model.flags.x;
  output.vertColor = input.vertColor;
  output.bloom = input.bloom;

  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  let normal = normalize(input.worldNormal);
  let lightDir = normalize(-light.direction.xyz);

  // Lambert diffuse
  let NdotL = max(dot(normal, lightDir), 0.0);
  let diffuse = light.color.rgb * NdotL;
  let ambient = light.ambient.rgb;

  var finalColor: vec3f;
  if (input.unlit > 0.5) {
    finalColor = input.color.rgb * input.vertColor;
  } else {
    finalColor = input.color.rgb * input.vertColor * (diffuse + ambient);
  }

  return vec4f(finalColor, input.color.a);
}
`

export const skinnedShaderSource = /* wgsl */ `
struct CameraUniforms {
  view: mat4x4f,
  projection: mat4x4f,
}

struct ModelUniforms {
  world: mat4x4f,
  color: vec4f,
  flags: vec4f,
}

struct LightUniforms {
  direction: vec4f,
  color: vec4f,
  ambient: vec4f,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<uniform> model: ModelUniforms;
@group(2) @binding(0) var<uniform> light: LightUniforms;
@group(3) @binding(0) var<storage, read> jointMatrices: array<mat4x4f>;

struct VertexInput {
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) vertColor: vec3f,
  @location(3) bloom: f32,
  @location(4) joints: vec4u,
  @location(5) weights: vec4f,
}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) worldNormal: vec3f,
  @location(1) color: vec4f,
  @location(2) unlit: f32,
  @location(3) vertColor: vec3f,
  @location(4) bloom: f32,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;

  let skinMat =
    input.weights.x * jointMatrices[input.joints.x] +
    input.weights.y * jointMatrices[input.joints.y] +
    input.weights.z * jointMatrices[input.joints.z] +
    input.weights.w * jointMatrices[input.joints.w];

  let skinnedWorld = model.world * skinMat;
  let worldPos = skinnedWorld * vec4f(input.position, 1.0);
  output.position = camera.projection * camera.view * worldPos;

  let normalMat = mat3x3f(
    skinnedWorld[0].xyz,
    skinnedWorld[1].xyz,
    skinnedWorld[2].xyz,
  );
  output.worldNormal = normalize(normalMat * input.normal);
  output.color = model.color;
  output.unlit = model.flags.x;
  output.vertColor = input.vertColor;
  output.bloom = input.bloom;

  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  let normal = normalize(input.worldNormal);
  let lightDir = normalize(-light.direction.xyz);

  let NdotL = max(dot(normal, lightDir), 0.0);
  let diffuse = light.color.rgb * NdotL;
  let ambient = light.ambient.rgb;

  var finalColor: vec3f;
  if (input.unlit > 0.5) {
    finalColor = input.color.rgb * input.vertColor;
  } else {
    finalColor = input.color.rgb * input.vertColor * (diffuse + ambient);
  }

  return vec4f(finalColor, input.color.a);
}
`

// MRT shaders — output to 2 render targets: scene color + bloom emission

export const mrtShaderSource = /* wgsl */ `
struct CameraUniforms {
  view: mat4x4f,
  projection: mat4x4f,
}

struct ModelUniforms {
  world: mat4x4f,
  color: vec4f,
  flags: vec4f,
}

struct LightUniforms {
  direction: vec4f,
  color: vec4f,
  ambient: vec4f,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<uniform> model: ModelUniforms;
@group(2) @binding(0) var<uniform> light: LightUniforms;

struct VertexInput {
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) vertColor: vec3f,
  @location(3) bloom: f32,
}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) worldNormal: vec3f,
  @location(1) color: vec4f,
  @location(2) unlit: f32,
  @location(3) vertColor: vec3f,
  @location(4) bloom: f32,
}

struct FragmentOutput {
  @location(0) color: vec4f,
  @location(1) bloom: vec4f,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;

  let worldPos = model.world * vec4f(input.position, 1.0);
  output.position = camera.projection * camera.view * worldPos;

  let normalMat = mat3x3f(
    model.world[0].xyz,
    model.world[1].xyz,
    model.world[2].xyz,
  );
  output.worldNormal = normalize(normalMat * input.normal);
  output.color = model.color;
  output.unlit = model.flags.x;
  output.vertColor = input.vertColor;
  output.bloom = input.bloom;

  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> FragmentOutput {
  let normal = normalize(input.worldNormal);
  let lightDir = normalize(-light.direction.xyz);

  let NdotL = max(dot(normal, lightDir), 0.0);
  let diffuse = light.color.rgb * NdotL;
  let ambient = light.ambient.rgb;

  var finalColor: vec3f;
  if (input.unlit > 0.5) {
    finalColor = input.color.rgb * input.vertColor;
  } else {
    finalColor = input.color.rgb * input.vertColor * (diffuse + ambient);
  }

  var out: FragmentOutput;
  out.color = vec4f(finalColor, input.color.a);
  out.bloom = vec4f(finalColor * input.bloom, 1.0);
  return out;
}
`

export const mrtSkinnedShaderSource = /* wgsl */ `
struct CameraUniforms {
  view: mat4x4f,
  projection: mat4x4f,
}

struct ModelUniforms {
  world: mat4x4f,
  color: vec4f,
  flags: vec4f,
}

struct LightUniforms {
  direction: vec4f,
  color: vec4f,
  ambient: vec4f,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<uniform> model: ModelUniforms;
@group(2) @binding(0) var<uniform> light: LightUniforms;
@group(3) @binding(0) var<storage, read> jointMatrices: array<mat4x4f>;

struct VertexInput {
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) vertColor: vec3f,
  @location(3) bloom: f32,
  @location(4) joints: vec4u,
  @location(5) weights: vec4f,
}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) worldNormal: vec3f,
  @location(1) color: vec4f,
  @location(2) unlit: f32,
  @location(3) vertColor: vec3f,
  @location(4) bloom: f32,
}

struct FragmentOutput {
  @location(0) color: vec4f,
  @location(1) bloom: vec4f,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;

  let skinMat =
    input.weights.x * jointMatrices[input.joints.x] +
    input.weights.y * jointMatrices[input.joints.y] +
    input.weights.z * jointMatrices[input.joints.z] +
    input.weights.w * jointMatrices[input.joints.w];

  let skinnedWorld = model.world * skinMat;
  let worldPos = skinnedWorld * vec4f(input.position, 1.0);
  output.position = camera.projection * camera.view * worldPos;

  let normalMat = mat3x3f(
    skinnedWorld[0].xyz,
    skinnedWorld[1].xyz,
    skinnedWorld[2].xyz,
  );
  output.worldNormal = normalize(normalMat * input.normal);
  output.color = model.color;
  output.unlit = model.flags.x;
  output.vertColor = input.vertColor;
  output.bloom = input.bloom;

  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> FragmentOutput {
  let normal = normalize(input.worldNormal);
  let lightDir = normalize(-light.direction.xyz);

  let NdotL = max(dot(normal, lightDir), 0.0);
  let diffuse = light.color.rgb * NdotL;
  let ambient = light.ambient.rgb;

  var finalColor: vec3f;
  if (input.unlit > 0.5) {
    finalColor = input.color.rgb * input.vertColor;
  } else {
    finalColor = input.color.rgb * input.vertColor * (diffuse + ambient);
  }

  var out: FragmentOutput;
  out.color = vec4f(finalColor, input.color.a);
  out.bloom = vec4f(finalColor * input.bloom, 1.0);
  return out;
}
`

// Post-processing shaders

export const fullscreenVertexSource = /* wgsl */ `
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var output: VertexOutput;
  // Fullscreen triangle: 3 vertices cover the entire screen
  let x = f32(i32(vertexIndex & 1u)) * 4.0 - 1.0;
  let y = f32(i32(vertexIndex >> 1u)) * 4.0 - 1.0;
  output.position = vec4f(x, y, 0.0, 1.0);
  output.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
  return output;
}
`

export const downsampleSource = /* wgsl */ `
@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var inputSampler: sampler;

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  // 4-tap bilinear downsample
  let texSize = vec2f(textureDimensions(inputTexture));
  let texelSize = 1.0 / texSize;
  let halfTexel = texelSize * 0.5;

  let s0 = textureSample(inputTexture, inputSampler, uv + vec2f(-halfTexel.x, -halfTexel.y));
  let s1 = textureSample(inputTexture, inputSampler, uv + vec2f( halfTexel.x, -halfTexel.y));
  let s2 = textureSample(inputTexture, inputSampler, uv + vec2f(-halfTexel.x,  halfTexel.y));
  let s3 = textureSample(inputTexture, inputSampler, uv + vec2f( halfTexel.x,  halfTexel.y));

  return (s0 + s1 + s2 + s3) * 0.25;
}
`

export const upsampleSource = /* wgsl */ `
@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var inputSampler: sampler;

struct BloomParams {
  radius: f32,
}

@group(0) @binding(2) var<uniform> params: BloomParams;

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  // 9-tap tent filter for smooth upsample
  let texSize = vec2f(textureDimensions(inputTexture));
  let texelSize = params.radius / texSize;

  var result = textureSample(inputTexture, inputSampler, uv) * 4.0;
  result += textureSample(inputTexture, inputSampler, uv + vec2f(-texelSize.x, 0.0)) * 2.0;
  result += textureSample(inputTexture, inputSampler, uv + vec2f( texelSize.x, 0.0)) * 2.0;
  result += textureSample(inputTexture, inputSampler, uv + vec2f(0.0, -texelSize.y)) * 2.0;
  result += textureSample(inputTexture, inputSampler, uv + vec2f(0.0,  texelSize.y)) * 2.0;
  result += textureSample(inputTexture, inputSampler, uv + vec2f(-texelSize.x, -texelSize.y));
  result += textureSample(inputTexture, inputSampler, uv + vec2f( texelSize.x, -texelSize.y));
  result += textureSample(inputTexture, inputSampler, uv + vec2f(-texelSize.x,  texelSize.y));
  result += textureSample(inputTexture, inputSampler, uv + vec2f( texelSize.x,  texelSize.y));

  return result / 16.0;
}
`

export const compositeSource = /* wgsl */ `
@group(0) @binding(0) var sceneTexture: texture_2d<f32>;
@group(0) @binding(1) var bloomTexture: texture_2d<f32>;
@group(0) @binding(2) var texSampler: sampler;

struct BloomParams {
  intensity: f32,
}

@group(0) @binding(3) var<uniform> params: BloomParams;

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let scene = textureSample(sceneTexture, texSampler, uv);
  let bloom = textureSample(bloomTexture, texSampler, uv);
  return vec4f(scene.rgb + bloom.rgb * params.intensity, scene.a);
}
`
