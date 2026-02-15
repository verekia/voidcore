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
layout(location = 3) in float a_bloom;

out vec3 v_worldNormal;
out vec4 v_color;
out float v_unlit;
out vec3 v_vertColor;
out float v_bloom;

void main() {
  vec4 worldPos = model.world * vec4(a_position, 1.0);
  gl_Position = camera.projection * camera.view * worldPos;

  mat3 normalMat = mat3(model.world);
  v_worldNormal = normalize(normalMat * a_normal);
  v_color = model.color;
  v_unlit = model.flags.x;
  v_vertColor = a_vertColor;
  v_bloom = a_bloom;
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
in float v_bloom;

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
layout(location = 3) in float a_bloom;
layout(location = 4) in uvec4 a_joints;
layout(location = 5) in vec4 a_weights;

out vec3 v_worldNormal;
out vec4 v_color;
out float v_unlit;
out vec3 v_vertColor;
out float v_bloom;

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
  v_bloom = a_bloom;
}
`

// MRT fragment shader — outputs to 2 render targets
export const mrtFragmentShaderGLSL = /* glsl */ `#version 300 es
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
in float v_bloom;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outBloom;

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

  outColor = vec4(finalColor, v_color.a);
  outBloom = vec4(finalColor * v_bloom, 1.0);
}
`

// Textured shaders — same as standard but with AO map sampling via UV coordinates

export const texturedVertexShaderGLSL = /* glsl */ `#version 300 es
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

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec3 a_vertColor;
layout(location = 3) in float a_bloom;
layout(location = 4) in vec2 a_uv;

out vec3 v_worldNormal;
out vec4 v_color;
out float v_unlit;
out vec3 v_vertColor;
out float v_bloom;
out vec2 v_uv;

void main() {
  vec4 worldPos = model.world * vec4(a_position, 1.0);
  gl_Position = camera.projection * camera.view * worldPos;

  mat3 normalMat = mat3(model.world);
  v_worldNormal = normalize(normalMat * a_normal);
  v_color = model.color;
  v_unlit = model.flags.x;
  v_vertColor = a_vertColor;
  v_bloom = a_bloom;
  v_uv = a_uv;
}
`

export const texturedFragmentShaderGLSL = /* glsl */ `#version 300 es
precision highp float;

layout(std140) uniform ModelUniforms {
  mat4 world;
  vec4 color;
  vec4 flags;
} model;

layout(std140) uniform LightUniforms {
  vec4 direction;
  vec4 color;
  vec4 ambient;
} light;

uniform sampler2D u_aoMap;

in vec3 v_worldNormal;
in vec4 v_color;
in float v_unlit;
in vec3 v_vertColor;
in float v_bloom;
in vec2 v_uv;

out vec4 fragColor;

void main() {
  vec3 normal = normalize(v_worldNormal);
  vec3 lightDir = normalize(-light.direction.xyz);

  float NdotL = max(dot(normal, lightDir), 0.0);
  vec3 diffuse = light.color.rgb * NdotL;
  float aoRaw = texture(u_aoMap, v_uv).r;
  float aoIntensity = model.flags.y;
  float ao = clamp(mix(1.0, aoRaw, aoIntensity), 0.0, 1.0);
  vec3 ambient = light.ambient.rgb * ao;

  vec3 finalColor;
  if (v_unlit > 0.5) {
    finalColor = v_color.rgb * v_vertColor;
  } else {
    finalColor = v_color.rgb * v_vertColor * (diffuse * ao + ambient);
  }

  fragColor = vec4(finalColor, v_color.a);
}
`

export const texturedMrtFragmentShaderGLSL = /* glsl */ `#version 300 es
precision highp float;

layout(std140) uniform ModelUniforms {
  mat4 world;
  vec4 color;
  vec4 flags;
} model;

layout(std140) uniform LightUniforms {
  vec4 direction;
  vec4 color;
  vec4 ambient;
} light;

uniform sampler2D u_aoMap;

in vec3 v_worldNormal;
in vec4 v_color;
in float v_unlit;
in vec3 v_vertColor;
in float v_bloom;
in vec2 v_uv;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outBloom;

void main() {
  vec3 normal = normalize(v_worldNormal);
  vec3 lightDir = normalize(-light.direction.xyz);

  float NdotL = max(dot(normal, lightDir), 0.0);
  vec3 diffuse = light.color.rgb * NdotL;
  float aoRaw = texture(u_aoMap, v_uv).r;
  float aoIntensity = model.flags.y;
  float ao = clamp(mix(1.0, aoRaw, aoIntensity), 0.0, 1.0);
  vec3 ambient = light.ambient.rgb * ao;

  vec3 finalColor;
  if (v_unlit > 0.5) {
    finalColor = v_color.rgb * v_vertColor;
  } else {
    finalColor = v_color.rgb * v_vertColor * (diffuse * ao + ambient);
  }

  outColor = vec4(finalColor, v_color.a);
  outBloom = vec4(finalColor * v_bloom, 1.0);
}
`

// Post-processing shaders

export const fullscreenVertexGLSL = /* glsl */ `#version 300 es
precision highp float;

out vec2 v_uv;

void main() {
  // Fullscreen triangle from vertex index
  float x = float(gl_VertexID & 1) * 4.0 - 1.0;
  float y = float(gl_VertexID >> 1) * 4.0 - 1.0;
  gl_Position = vec4(x, y, 0.0, 1.0);
  v_uv = vec2((x + 1.0) * 0.5, (y + 1.0) * 0.5);
}
`

export const downsampleFragmentGLSL = /* glsl */ `#version 300 es
precision highp float;

uniform sampler2D u_input;

in vec2 v_uv;
out vec4 fragColor;

void main() {
  vec2 texSize = vec2(textureSize(u_input, 0));
  vec2 texelSize = 1.0 / texSize;
  vec2 halfTexel = texelSize * 0.5;

  vec4 s0 = texture(u_input, v_uv + vec2(-halfTexel.x, -halfTexel.y));
  vec4 s1 = texture(u_input, v_uv + vec2( halfTexel.x, -halfTexel.y));
  vec4 s2 = texture(u_input, v_uv + vec2(-halfTexel.x,  halfTexel.y));
  vec4 s3 = texture(u_input, v_uv + vec2( halfTexel.x,  halfTexel.y));

  fragColor = (s0 + s1 + s2 + s3) * 0.25;
}
`

export const upsampleFragmentGLSL = /* glsl */ `#version 300 es
precision highp float;

uniform sampler2D u_input;
uniform float u_radius;

in vec2 v_uv;
out vec4 fragColor;

void main() {
  vec2 texSize = vec2(textureSize(u_input, 0));
  vec2 texelSize = u_radius / texSize;

  vec4 result = texture(u_input, v_uv) * 4.0;
  result += texture(u_input, v_uv + vec2(-texelSize.x, 0.0)) * 2.0;
  result += texture(u_input, v_uv + vec2( texelSize.x, 0.0)) * 2.0;
  result += texture(u_input, v_uv + vec2(0.0, -texelSize.y)) * 2.0;
  result += texture(u_input, v_uv + vec2(0.0,  texelSize.y)) * 2.0;
  result += texture(u_input, v_uv + vec2(-texelSize.x, -texelSize.y));
  result += texture(u_input, v_uv + vec2( texelSize.x, -texelSize.y));
  result += texture(u_input, v_uv + vec2(-texelSize.x,  texelSize.y));
  result += texture(u_input, v_uv + vec2( texelSize.x,  texelSize.y));

  fragColor = result / 16.0;
}
`

export const compositeFragmentGLSL = /* glsl */ `#version 300 es
precision highp float;

uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float u_intensity;

in vec2 v_uv;
out vec4 fragColor;

void main() {
  vec4 scene = texture(u_scene, v_uv);
  vec4 bloom = texture(u_bloom, v_uv);
  fragColor = vec4(scene.rgb + bloom.rgb * u_intensity, scene.a);
}
`
