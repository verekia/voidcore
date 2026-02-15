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
}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) worldNormal: vec3f,
  @location(1) color: vec4f,
  @location(2) unlit: f32,
  @location(3) vertColor: vec3f,
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
