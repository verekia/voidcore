// GLSL 300 es shaders for WebGL2 renderer

export const vertexShaderGLSL = /* glsl */ `#version 300 es
precision highp float;

layout(std140) uniform CameraUniforms {
  mat4 view;
  mat4 projection;
} camera;

layout(std140) uniform ModelUniforms {
  mat4 world;
  vec4 color;
  vec4 flags; // x = unlit
} model;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec3 a_vertColor;

out vec3 v_worldNormal;
out vec4 v_color;
out float v_unlit;
out vec3 v_vertColor;

void main() {
  vec4 worldPos = model.world * vec4(a_position, 1.0);
  gl_Position = camera.projection * camera.view * worldPos;

  mat3 normalMat = mat3(model.world);
  v_worldNormal = normalize(normalMat * a_normal);
  v_color = model.color;
  v_unlit = model.flags.x;
  v_vertColor = a_vertColor;
}
`

export const fragmentShaderGLSL = /* glsl */ `#version 300 es
precision highp float;

layout(std140) uniform LightUniforms {
  vec4 direction;
  vec4 color;
  vec4 ambient;
} light;

in vec3 v_worldNormal;
in vec4 v_color;
in float v_unlit;
in vec3 v_vertColor;

out vec4 fragColor;

void main() {
  vec3 normal = normalize(v_worldNormal);
  vec3 lightDir = normalize(-light.direction.xyz);

  float NdotL = max(dot(normal, lightDir), 0.0);
  vec3 diffuse = light.color.rgb * NdotL;
  vec3 ambient = light.ambient.rgb;

  vec3 finalColor;
  if (v_unlit > 0.5) {
    finalColor = v_color.rgb * v_vertColor;
  } else {
    finalColor = v_color.rgb * v_vertColor * (diffuse + ambient);
  }

  fragColor = vec4(finalColor, v_color.a);
}
`

export const skinnedVertexShaderGLSL = /* glsl */ `#version 300 es
precision highp float;

layout(std140) uniform CameraUniforms {
  mat4 view;
  mat4 projection;
} camera;

layout(std140) uniform ModelUniforms {
  mat4 world;
  vec4 color;
  vec4 flags;
} model;

layout(std140) uniform JointUniforms {
  mat4 joints[128];
} skin;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec3 a_vertColor;
layout(location = 3) in uvec4 a_joints;
layout(location = 4) in vec4 a_weights;

out vec3 v_worldNormal;
out vec4 v_color;
out float v_unlit;
out vec3 v_vertColor;

void main() {
  mat4 skinMat =
    a_weights.x * skin.joints[a_joints.x] +
    a_weights.y * skin.joints[a_joints.y] +
    a_weights.z * skin.joints[a_joints.z] +
    a_weights.w * skin.joints[a_joints.w];

  mat4 skinnedWorld = model.world * skinMat;
  vec4 worldPos = skinnedWorld * vec4(a_position, 1.0);
  gl_Position = camera.projection * camera.view * worldPos;

  mat3 normalMat = mat3(skinnedWorld);
  v_worldNormal = normalize(normalMat * a_normal);
  v_color = model.color;
  v_unlit = model.flags.x;
  v_vertColor = a_vertColor;
}
`
