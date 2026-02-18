// Inline GLSL shaders for WebGL2

export const LAMBERT_VERT = `#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in float a_materialIndex;

uniform mat4 u_viewProjection;
uniform mat4 u_worldMatrix;
uniform mat4 u_normalMatrix;

out vec3 v_worldPos;
out vec3 v_normal;
out vec2 v_uv;
flat out int v_materialIndex;

void main() {
  vec4 worldPos = u_worldMatrix * vec4(a_position, 1.0);
  v_worldPos = worldPos.xyz;
  v_normal = normalize((u_normalMatrix * vec4(a_normal, 0.0)).xyz);
  v_uv = a_uv;
  v_materialIndex = int(a_materialIndex);
  gl_Position = u_viewProjection * worldPos;
}
`

export const LAMBERT_FRAG = `#version 300 es
precision highp float;

in vec3 v_worldPos;
in vec3 v_normal;
in vec2 v_uv;
flat in int v_materialIndex;

struct PaletteEntry {
  vec4 color;    // xyz = RGB, w = opacity
  vec4 emissive; // xyz = RGB, w = emissiveIntensity
};

uniform vec3 u_lightDirection;
uniform vec3 u_lightColor;
uniform float u_lightIntensity;
uniform vec3 u_ambientColor;
uniform float u_ambientIntensity;

uniform vec3 u_baseColor;
uniform float u_opacity;
uniform bool u_hasPalette;
uniform PaletteEntry u_palette[32];

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragEmissive;

void main() {
  vec3 normal = normalize(v_normal);
  vec3 baseColor = u_baseColor;
  float alpha = u_opacity;
  vec3 emissive = vec3(0.0);

  if (u_hasPalette) {
    int idx = clamp(v_materialIndex, 0, 31);
    baseColor = u_palette[idx].color.rgb;
    alpha = u_palette[idx].color.a;
    emissive = u_palette[idx].emissive.rgb * u_palette[idx].emissive.a;
  }

  vec3 ambient = u_ambientColor * u_ambientIntensity;
  float NdotL = max(dot(normal, u_lightDirection), 0.0);
  vec3 diffuse = u_lightColor * u_lightIntensity * NdotL;

  vec3 litColor = baseColor * (ambient + diffuse);
  vec3 finalColor = litColor + emissive;

  fragColor = vec4(finalColor, alpha);
  fragEmissive = vec4(emissive, 1.0);
}
`

export const LAMBERT_SKINNED_VERT = `#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in float a_materialIndex;
layout(location = 4) in vec4 a_joints;
layout(location = 5) in vec4 a_weights;

uniform mat4 u_viewProjection;
uniform mat4 u_worldMatrix;
uniform mat4 u_normalMatrix;
uniform mat4 u_boneMatrices[32];

out vec3 v_worldPos;
out vec3 v_normal;
out vec2 v_uv;
flat out int v_materialIndex;

void main() {
  mat4 skinMatrix =
    a_weights.x * u_boneMatrices[int(a_joints.x)] +
    a_weights.y * u_boneMatrices[int(a_joints.y)] +
    a_weights.z * u_boneMatrices[int(a_joints.z)] +
    a_weights.w * u_boneMatrices[int(a_joints.w)];

  vec4 skinnedPos = skinMatrix * vec4(a_position, 1.0);
  v_worldPos = skinnedPos.xyz;

  vec4 skinnedNorm = skinMatrix * vec4(a_normal, 0.0);
  v_normal = normalize(skinnedNorm.xyz);
  v_uv = a_uv;
  v_materialIndex = int(a_materialIndex);
  gl_Position = u_viewProjection * skinnedPos;
}
`

export const BASIC_SKINNED_VERT = `#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in float a_materialIndex;
layout(location = 4) in vec4 a_joints;
layout(location = 5) in vec4 a_weights;

uniform mat4 u_viewProjection;
uniform mat4 u_worldMatrix;
uniform mat4 u_boneMatrices[32];

out vec2 v_uv;

void main() {
  mat4 skinMatrix =
    a_weights.x * u_boneMatrices[int(a_joints.x)] +
    a_weights.y * u_boneMatrices[int(a_joints.y)] +
    a_weights.z * u_boneMatrices[int(a_joints.z)] +
    a_weights.w * u_boneMatrices[int(a_joints.w)];

  vec4 skinnedPos = skinMatrix * vec4(a_position, 1.0);
  gl_Position = u_viewProjection * skinnedPos;
  v_uv = a_uv;
}
`

export const BASIC_VERT = `#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;

uniform mat4 u_viewProjection;
uniform mat4 u_worldMatrix;

out vec2 v_uv;

void main() {
  gl_Position = u_viewProjection * u_worldMatrix * vec4(a_position, 1.0);
  v_uv = a_uv;
}
`

export const BASIC_FRAG = `#version 300 es
precision highp float;

in vec2 v_uv;

uniform vec3 u_baseColor;
uniform float u_opacity;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragEmissive;

void main() {
  fragColor = vec4(u_baseColor, u_opacity);
  fragEmissive = vec4(0.0, 0.0, 0.0, 1.0);
}
`

// Fullscreen triangle vertex shader (no vertex buffer needed)
export const FULLSCREEN_VERT = `#version 300 es
precision highp float;

out vec2 v_uv;

void main() {
  // Generate fullscreen triangle from vertex ID
  float x = float((gl_VertexID & 1) << 2);
  float y = float((gl_VertexID & 2) << 1);
  v_uv = vec2(x * 0.5, y * 0.5);
  gl_Position = vec4(x - 1.0, y - 1.0, 0.0, 1.0);
}
`

export const BLOOM_DOWNSAMPLE_FRAG = `#version 300 es
precision highp float;

in vec2 v_uv;
uniform sampler2D u_srcTexture;
uniform vec2 u_texelSize;
uniform bool u_useKarisAverage;

out vec4 fragColor;

float karisWeight(vec3 c) {
  float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
  return 1.0 / (1.0 + luma);
}

void main() {
  // 13-tap Jimenez downsample
  vec3 a = texture(u_srcTexture, v_uv + vec2(-2.0, -2.0) * u_texelSize).rgb;
  vec3 b = texture(u_srcTexture, v_uv + vec2( 0.0, -2.0) * u_texelSize).rgb;
  vec3 c = texture(u_srcTexture, v_uv + vec2( 2.0, -2.0) * u_texelSize).rgb;
  vec3 d = texture(u_srcTexture, v_uv + vec2(-2.0,  0.0) * u_texelSize).rgb;
  vec3 e = texture(u_srcTexture, v_uv).rgb;
  vec3 f = texture(u_srcTexture, v_uv + vec2( 2.0,  0.0) * u_texelSize).rgb;
  vec3 g = texture(u_srcTexture, v_uv + vec2(-2.0,  2.0) * u_texelSize).rgb;
  vec3 h = texture(u_srcTexture, v_uv + vec2( 0.0,  2.0) * u_texelSize).rgb;
  vec3 i = texture(u_srcTexture, v_uv + vec2( 2.0,  2.0) * u_texelSize).rgb;
  vec3 j = texture(u_srcTexture, v_uv + vec2(-1.0, -1.0) * u_texelSize).rgb;
  vec3 k = texture(u_srcTexture, v_uv + vec2( 1.0, -1.0) * u_texelSize).rgb;
  vec3 l = texture(u_srcTexture, v_uv + vec2(-1.0,  1.0) * u_texelSize).rgb;
  vec3 m = texture(u_srcTexture, v_uv + vec2( 1.0,  1.0) * u_texelSize).rgb;

  vec3 result;
  if (u_useKarisAverage) {
    // Karis average for first downsample to prevent fireflies
    vec3 g0 = (a + b + d + e) * 0.25;
    vec3 g1 = (b + c + e + f) * 0.25;
    vec3 g2 = (d + e + g + h) * 0.25;
    vec3 g3 = (e + f + h + i) * 0.25;
    vec3 g4 = (j + k + l + m) * 0.25;
    float w0 = karisWeight(g0);
    float w1 = karisWeight(g1);
    float w2 = karisWeight(g2);
    float w3 = karisWeight(g3);
    float w4 = karisWeight(g4);
    float wSum = w0 + w1 + w2 + w3 + w4;
    result = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / wSum;
  } else {
    result = e * 0.125;
    result += (a + c + g + i) * 0.03125;
    result += (b + d + f + h) * 0.0625;
    result += (j + k + l + m) * 0.125;
  }

  fragColor = vec4(result, 1.0);
}
`

export const BLOOM_UPSAMPLE_FRAG = `#version 300 es
precision highp float;

in vec2 v_uv;
uniform sampler2D u_srcTexture;
uniform vec2 u_texelSize;

out vec4 fragColor;

void main() {
  // 9-tap tent filter
  vec3 result = texture(u_srcTexture, v_uv + vec2(-1.0, -1.0) * u_texelSize).rgb * 1.0
              + texture(u_srcTexture, v_uv + vec2( 0.0, -1.0) * u_texelSize).rgb * 2.0
              + texture(u_srcTexture, v_uv + vec2( 1.0, -1.0) * u_texelSize).rgb * 1.0
              + texture(u_srcTexture, v_uv + vec2(-1.0,  0.0) * u_texelSize).rgb * 2.0
              + texture(u_srcTexture, v_uv).rgb * 4.0
              + texture(u_srcTexture, v_uv + vec2( 1.0,  0.0) * u_texelSize).rgb * 2.0
              + texture(u_srcTexture, v_uv + vec2(-1.0,  1.0) * u_texelSize).rgb * 1.0
              + texture(u_srcTexture, v_uv + vec2( 0.0,  1.0) * u_texelSize).rgb * 2.0
              + texture(u_srcTexture, v_uv + vec2( 1.0,  1.0) * u_texelSize).rgb * 1.0;
  fragColor = vec4(result / 16.0, 1.0);
}
`

export const BLIT_FRAG = `#version 300 es
precision highp float;

in vec2 v_uv;
uniform sampler2D u_sceneTexture;
uniform sampler2D u_bloomTexture;
uniform float u_bloomIntensity;

out vec4 fragColor;

void main() {
  vec3 scene = texture(u_sceneTexture, v_uv).rgb;
  vec3 bloom = texture(u_bloomTexture, v_uv).rgb;
  vec3 color = scene + bloom * u_bloomIntensity;
  // Gamma correction
  color = pow(color, vec3(1.0 / 2.2));
  fragColor = vec4(color, 1.0);
}
`

export const RESOLVE_FRAG = `#version 300 es
precision highp float;

in vec2 v_uv;
uniform sampler2D u_texture;

out vec4 fragColor;

void main() {
  fragColor = texture(u_texture, v_uv);
}
`
