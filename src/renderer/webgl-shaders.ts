// GLSL Shaders – GPU programs for the WebGL2 renderer.
//
// Shaders are small programs that run on the GPU for every vertex and every pixel.
//   Vertex shader   – Transforms 3D positions into screen coordinates.
//   Fragment shader – Computes the final color of each pixel.
//
// This file contains five shader variants plus shadow depth shaders:
//   Lambert          – Diffuse lighting (ambient + directional light × surface normal)
//                      with shadow map sampling (PCF 3×3).
//   Lambert Textured – Same as Lambert but also samples color map and AO map textures.
//   Lambert Skinned  – Same lighting + shadows, but vertices are deformed by bone matrices.
//   Basic            – Flat unlit color (no lighting calculations).
//   Basic Skinned    – Flat unlit color with skeletal deformation.
//
// Plus two shadow depth shaders (depth-only, used to render the shadow map):
//   Shadow Depth         – Writes depth from light's perspective (static meshes).
//   Shadow Depth Skinned – Same but with skeletal deformation.
//
// Plus outline shaders (inverted hull technique):
//   Outline              – Inflates vertices along normals, outputs solid color.
//   Outline Skinned      – Same but with skeletal deformation applied before inflation.
//
// Plus three post-processing shaders (fullscreen triangle, no vertex buffer needed):
//   Bloom Downsample – 13-tap filter that progressively shrinks the emissive image.
//                      First pass uses Karis average to prevent "firefly" artifacts.
//   Bloom Upsample   – 9-tap tent filter that blurs back up, additively blending.
//   Blit             – Combines the scene color + bloom and applies gamma correction.
//
// Data flow: CPU uploads uniform buffers (UBOs) with matrices and light info. The vertex
// shader reads per-object transforms from ObjectBlock/SkinnedObjectBlock (binding 1) and
// per-frame data (view-projection, lights) from FrameBlock (binding 0).

// Inline GLSL shaders for WebGL2

// ─── Shared UBO block declarations ───────────────────────────────────
//
// FrameBlock (binding 0, 208 bytes / 52 floats, std140):
//   mat4  u_viewProjection       float 0-15   (64 bytes)
//   vec3  u_lightDirection       float 16-18  (12 bytes + 4 pad)
//   float _lightPad              float 19
//   vec3  u_lightColor           float 20-22  (12 bytes)
//   float u_lightIntensity       float 23
//   vec3  u_ambientColor         float 24-26  (12 bytes)
//   float u_ambientIntensity     float 27
//   float u_shadowEnabled        float 28
//   float _sp1, _sp2, _sp3       float 29-31  (pad for mat4 alignment)
//   mat4  u_shadowVP             float 32-47
//   float u_constantBias         float 48
//   float u_slopeBias            float 49
//   float u_invMapSize           float 50
//   float _biasPad               float 51
//
// ObjectBlock (binding 1, 128 bytes std140):
//   mat4 u_worldMatrix    offset 0
//   mat4 u_normalMatrix   offset 64
//
// SkinnedObjectBlock (binding 1, 2176 bytes std140):
//   mat4 u_worldMatrix       offset 0
//   mat4 u_normalMatrix      offset 64
//   mat4 u_boneMatrices[32]  offset 128
//
// ShadowBlock (binding 2, 64 bytes std140):
//   mat4 u_shadowVP          offset 0

const FRAME_BLOCK = `
layout(std140) uniform FrameBlock {
  mat4 u_viewProjection;
  vec3 u_lightDirection;
  float _lightPad;
  vec3 u_lightColor;
  float u_lightIntensity;
  vec3 u_ambientColor;
  float u_ambientIntensity;
  float u_shadowEnabled;
  float _sp1; float _sp2; float _sp3;
  mat4 u_shadowVP;
  float u_constantBias;
  float u_slopeBias;
  float u_invMapSize;
  float _biasPad;
};`

const OBJECT_BLOCK = `
layout(std140) uniform ObjectBlock {
  mat4 u_worldMatrix;
  mat4 u_normalMatrix;
};`

const SKINNED_OBJECT_BLOCK = `
layout(std140) uniform SkinnedObjectBlock {
  mat4 u_worldMatrix;
  mat4 u_normalMatrix;
  mat4 u_boneMatrices[32];
};`

const SHADOW_BLOCK = `
layout(std140) uniform ShadowBlock {
  mat4 u_shadowVP;
};`

const SHADOW_FUNCTIONS = `
float pcf9(vec2 uv, float refDepth, float ts) {
  float s = 0.0;
  s += texture(u_shadowMap, vec3(uv + vec2(-ts, -ts), refDepth));
  s += texture(u_shadowMap, vec3(uv + vec2(0.0, -ts), refDepth));
  s += texture(u_shadowMap, vec3(uv + vec2( ts, -ts), refDepth));
  s += texture(u_shadowMap, vec3(uv + vec2(-ts, 0.0), refDepth));
  s += texture(u_shadowMap, vec3(uv,                   refDepth));
  s += texture(u_shadowMap, vec3(uv + vec2( ts, 0.0), refDepth));
  s += texture(u_shadowMap, vec3(uv + vec2(-ts,  ts), refDepth));
  s += texture(u_shadowMap, vec3(uv + vec2(0.0,  ts), refDepth));
  s += texture(u_shadowMap, vec3(uv + vec2( ts,  ts), refDepth));
  return s / 9.0;
}

float sampleShadow(vec3 worldPos, float NdotL) {
  if (u_shadowEnabled < 0.5) return 1.0;

  vec4 lightClip = u_shadowVP * vec4(worldPos, 1.0);
  vec2 uv = lightClip.xy * 0.5 + 0.5;
  float depth = lightClip.z * 0.5 + 0.5;

  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || depth < 0.0 || depth > 1.0) {
    return 1.0;
  }

  float bias = u_constantBias + u_slopeBias * (1.0 - NdotL);
  return pcf9(uv, depth - bias, u_invMapSize);
}
`

// ─── Shadow depth shaders (depth-only pass) ─────────────────────────

export const SHADOW_DEPTH_VERT = `#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;

${SHADOW_BLOCK}
${OBJECT_BLOCK}

void main() {
  gl_Position = u_shadowVP * u_worldMatrix * vec4(a_position, 1.0);
}
`

export const SHADOW_DEPTH_SKINNED_VERT = `#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 4) in vec4 a_joints;
layout(location = 5) in vec4 a_weights;

${SHADOW_BLOCK}
${SKINNED_OBJECT_BLOCK}

void main() {
  mat4 skinMatrix =
    a_weights.x * u_boneMatrices[int(a_joints.x)] +
    a_weights.y * u_boneMatrices[int(a_joints.y)] +
    a_weights.z * u_boneMatrices[int(a_joints.z)] +
    a_weights.w * u_boneMatrices[int(a_joints.w)];
  vec4 skinnedPos = skinMatrix * vec4(a_position, 1.0);
  gl_Position = u_shadowVP * skinnedPos;
}
`

export const SHADOW_DEPTH_FRAG = `#version 300 es
precision mediump float; // mediump sufficient — shader body is empty
void main() {}
`

// ─── Scene shaders ──────────────────────────────────────────────────

export const LAMBERT_VERT = `#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in float a_materialIndex;

${FRAME_BLOCK}
${OBJECT_BLOCK}

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

${FRAME_BLOCK}

uniform vec3 u_baseColor;
uniform float u_opacity;
uniform bool u_hasPalette;
uniform PaletteEntry u_palette[32];
uniform highp sampler2DShadow u_shadowMap;
uniform bool u_receiveShadow;
uniform float u_emissiveBrightness;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragEmissive;

${SHADOW_FUNCTIONS}

void main() {
  vec3 normal = normalize(v_normal);
  vec3 baseColor = u_baseColor;
  vec3 emissive = vec3(0.0);

  if (u_hasPalette) {
    int idx = clamp(v_materialIndex, 0, 31);
    baseColor = u_palette[idx].color.rgb;
    emissive = u_palette[idx].emissive.rgb * u_palette[idx].emissive.a;
  }

  vec3 ambient = u_ambientColor * u_ambientIntensity;
  float NdotL = max(dot(normal, u_lightDirection), 0.0);
  float shadow = u_receiveShadow ? sampleShadow(v_worldPos, NdotL) : 1.0;
  vec3 diffuse = u_lightColor * u_lightIntensity * NdotL * shadow;

  vec3 litColor = baseColor * (ambient + diffuse);
  float brightness = dot(emissive, vec3(0.299, 0.587, 0.114));
  vec3 screenEmissive = mix(emissive, vec3(brightness), clamp(brightness, 0.0, 1.0) * u_emissiveBrightness);
  vec3 finalColor = litColor + screenEmissive;

  fragColor = vec4(finalColor, u_opacity);
  fragEmissive = vec4(emissive * u_opacity, u_opacity);
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

${FRAME_BLOCK}
${SKINNED_OBJECT_BLOCK}

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

export const LAMBERT_TEXTURED_FRAG = `#version 300 es
precision highp float;

in vec3 v_worldPos;
in vec3 v_normal;
in vec2 v_uv;
flat in int v_materialIndex;

struct PaletteEntry {
  vec4 color;    // xyz = RGB, w = opacity
  vec4 emissive; // xyz = RGB, w = emissiveIntensity
};

${FRAME_BLOCK}

uniform vec3 u_baseColor;
uniform float u_opacity;
uniform bool u_hasPalette;
uniform PaletteEntry u_palette[32];
uniform highp sampler2DShadow u_shadowMap;
uniform bool u_receiveShadow;
uniform float u_emissiveBrightness;
uniform sampler2D u_colorMap;
uniform sampler2D u_aoMap;
uniform float u_aoIntensity;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragEmissive;

${SHADOW_FUNCTIONS}

void main() {
  vec3 normal = normalize(v_normal);
  vec3 baseColor = u_baseColor;
  vec3 emissive = vec3(0.0);

  if (u_hasPalette) {
    int idx = clamp(v_materialIndex, 0, 31);
    baseColor = u_palette[idx].color.rgb;
    emissive = u_palette[idx].emissive.rgb * u_palette[idx].emissive.a;
  }

  // Sample texture maps (dummy white textures → identity when no map assigned)
  vec3 texColor = texture(u_colorMap, v_uv).rgb;
  baseColor *= texColor;

  float ao = mix(1.0, texture(u_aoMap, v_uv).r, u_aoIntensity);
  vec3 ambient = u_ambientColor * u_ambientIntensity * ao;

  float NdotL = max(dot(normal, u_lightDirection), 0.0);
  float shadow = u_receiveShadow ? sampleShadow(v_worldPos, NdotL) : 1.0;
  vec3 diffuse = u_lightColor * u_lightIntensity * NdotL * shadow;

  vec3 litColor = baseColor * (ambient + diffuse);
  float brightness = dot(emissive, vec3(0.299, 0.587, 0.114));
  vec3 screenEmissive = mix(emissive, vec3(brightness), clamp(brightness, 0.0, 1.0) * u_emissiveBrightness);
  vec3 finalColor = litColor + screenEmissive;

  fragColor = vec4(finalColor, u_opacity);
  fragEmissive = vec4(emissive * u_opacity, u_opacity);
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

${FRAME_BLOCK}
${SKINNED_OBJECT_BLOCK}

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

${FRAME_BLOCK}
${OBJECT_BLOCK}

out vec2 v_uv;

void main() {
  gl_Position = u_viewProjection * u_worldMatrix * vec4(a_position, 1.0);
  v_uv = a_uv;
}
`

export const BASIC_FRAG = `#version 300 es
precision mediump float; // No precision-sensitive ops — mediump saves ALU registers on mobile GPUs

in vec2 v_uv;

uniform vec3 u_baseColor;
uniform float u_opacity;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragEmissive;

void main() {
  fragColor = vec4(u_baseColor, u_opacity);
  fragEmissive = vec4(0.0, 0.0, 0.0, u_opacity);
}
`

// ─── Outline shaders (inverted hull technique) ───────────────────────

export const OUTLINE_VERT = `#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;

${FRAME_BLOCK}
${OBJECT_BLOCK}

uniform float u_outlineThickness;

void main() {
  vec3 inflated = a_position + normalize(a_normal) * u_outlineThickness;
  gl_Position = u_viewProjection * u_worldMatrix * vec4(inflated, 1.0);
}
`

export const OUTLINE_SKINNED_VERT = `#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in float a_materialIndex;
layout(location = 4) in vec4 a_joints;
layout(location = 5) in vec4 a_weights;

${FRAME_BLOCK}
${SKINNED_OBJECT_BLOCK}

uniform float u_outlineThickness;

void main() {
  mat4 skinMatrix =
    a_weights.x * u_boneMatrices[int(a_joints.x)] +
    a_weights.y * u_boneMatrices[int(a_joints.y)] +
    a_weights.z * u_boneMatrices[int(a_joints.z)] +
    a_weights.w * u_boneMatrices[int(a_joints.w)];

  vec4 skinnedPos = skinMatrix * vec4(a_position, 1.0);
  vec3 skinnedNorm = normalize((skinMatrix * vec4(a_normal, 0.0)).xyz);
  vec3 inflated = skinnedPos.xyz + skinnedNorm * u_outlineThickness;
  gl_Position = u_viewProjection * vec4(inflated, 1.0);
}
`

export const OUTLINE_FRAG = `#version 300 es
precision mediump float;

uniform vec3 u_outlineColor;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragEmissive;

void main() {
  fragColor = vec4(u_outlineColor, 1.0);
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
  fragColor = vec4(color, 1.0);
}
`
