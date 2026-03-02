// GLSL Shaders – GPU programs for the WebGL2 renderer.
//
// Shaders are small programs that run on the GPU for every vertex and every pixel.
//   Vertex shader   – Transforms 3D positions into screen coordinates.
//   Fragment shader – Computes the final color of each pixel.
//
// This file contains shader variants plus shadow depth shaders:
//   Lambert          – Diffuse lighting (ambient + directional light × surface normal)
//                      with shadow map sampling (PCF 3×3).
//   Lambert VC       – Same as Lambert but reads per-vertex color and emissive attributes
//                      (baked from palette via bakePalette()). Also samples the AO map,
//                      per-material tiled AO via a 2D array texture (world-space XY tiling),
//                      per-material tiled normal maps via a second 2D array texture
//                      with cotangent-frame normal mapping (screen-space derivatives),
//                      and per-material noise color blending via a procedural dot noise
//                      function that blends between primary and secondary vertex colors.
//   Lambert Textured – Same as Lambert but also samples color map and AO map textures.
//   Lambert Skinned  – Same lighting + shadows, but vertices are deformed by bone matrices.
//   Lambert Skinned VC – Skinned variant with vertex colors and emissive.
//   Basic            – Flat unlit color (no lighting calculations).
//   Basic Skinned    – Flat unlit color with skeletal deformation.
//
// Plus two shadow depth shaders (depth-only, used to render the shadow map):
//   Shadow Depth         – Writes depth from light's perspective (static meshes).
//   Shadow Depth Skinned – Same but with skeletal deformation.
//
// Outlined Lambert meshes use combined geometry with doubled vertices (original + outline with
// smooth normals). The Lambert shader reads normal.w to distinguish main vs outline fragments.
// Outline vertices are inflated by thickness from the object UBO, and the fragment shader uses
// gl_FrontFacing to discard front-facing outline triangles (keeping only back-facing silhouette).
//
// Plus three post-processing shaders (fullscreen triangle, no vertex buffer needed):
//   Bloom Downsample – 13-tap filter that progressively shrinks the emissive image.
//                      First pass uses Karis average to prevent "firefly" artifacts.
//   Bloom Upsample   – 9-tap tent filter that blurs back up, additively blending.
//   Blit             – Combines the scene color + bloom and applies gamma correction.
//
// Custom shader builders (buildLambertVert, buildBasicVert, buildLambertCustomFrag, etc.)
// generate shader source with user-provided GLSL code injected at hook points. The basic
// custom vert shaders use vec4 a_normal to match the snorm8x4 attribute layout, and expose
// v_worldPos/v_normal so custom code can access them even though the standard basic shader
// does not use lighting. Custom uniforms are declared as a std140 UBO (CustomBlock at
// binding point 2), accessible as `uniforms.xxx` in both vertex and fragment.
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
// ObjectBlock (binding 1, 144 bytes std140):
//   mat4 u_worldMatrix              offset 0
//   mat4 u_normalMatrix             offset 64
//   vec4 u_outlineColorAndThickness  offset 128
//
// SkinnedObjectBlock (binding 1, 2192 bytes std140):
//   mat4 u_worldMatrix              offset 0
//   mat4 u_normalMatrix             offset 64
//   vec4 u_outlineColorAndThickness  offset 128
//   mat4 u_boneMatrices[32]          offset 144
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
  vec3 u_cameraPos;
  float u_elapsed;
};`

const OBJECT_BLOCK = `
layout(std140) uniform ObjectBlock {
  mat4 u_worldMatrix;
  mat4 u_normalMatrix;
  vec4 u_outlineColorAndThickness;
};`

const SKINNED_OBJECT_BLOCK = `
layout(std140) uniform SkinnedObjectBlock {
  mat4 u_worldMatrix;
  mat4 u_normalMatrix;
  vec4 u_outlineColorAndThickness;
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
layout(location = 1) in vec4 a_normal;
layout(location = 2) in vec2 a_uv;

${FRAME_BLOCK}
${OBJECT_BLOCK}

out vec3 v_worldPos;
out vec3 v_normal;
out vec2 v_uv;
flat out float v_outlineFlag;

void main() {
  float flag = a_normal.w;
  v_outlineFlag = flag;
  vec3 pos = a_position;
  if (flag > 0.5) {
    pos = pos + normalize(a_normal.xyz) * u_outlineColorAndThickness.w;
  }
  vec4 worldPos = u_worldMatrix * vec4(pos, 1.0);
  v_worldPos = worldPos.xyz;
  v_normal = normalize((u_normalMatrix * vec4(a_normal.xyz, 0.0)).xyz);
  v_uv = a_uv;
  gl_Position = u_viewProjection * worldPos;
  gl_Position.z += flag * 0.0001 * gl_Position.w; // Depth bias to prevent outline z-fighting
}
`

const _makeLambertFrag = (objectBlock: string) => `#version 300 es
precision highp float;

in vec3 v_worldPos;
in vec3 v_normal;
in vec2 v_uv;
flat in float v_outlineFlag;

${FRAME_BLOCK}
${objectBlock}

uniform vec3 u_baseColor;
uniform float u_opacity;
uniform highp sampler2DShadow u_shadowMap;
uniform bool u_receiveShadow;
uniform float u_emissiveBrightness;
uniform float u_wrapLighting;
uniform float u_darkness;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragEmissive;

${SHADOW_FUNCTIONS}

void main() {
  if (v_outlineFlag > 0.5) {
    if (gl_FrontFacing || u_outlineColorAndThickness.w <= 0.0) discard;
    fragColor = vec4(u_outlineColorAndThickness.xyz, 1.0);
    fragEmissive = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  vec3 normal = normalize(v_normal);
  vec3 baseColor = u_baseColor;

  vec3 ambient = u_ambientColor * u_ambientIntensity;
  float rawNdotL = dot(normal, u_lightDirection);
  float shadow = u_receiveShadow ? sampleShadow(v_worldPos, max(rawNdotL, 0.0)) : 1.0;
  float wrap = u_wrapLighting;
  float realNdotL = max(rawNdotL, 0.0);
  float wrapNdotL = max(rawNdotL + wrap, 0.0) / (1.0 + wrap);
  float NdotL = wrapNdotL - realNdotL * (1.0 - shadow);
  vec3 diffuse = u_lightColor * u_lightIntensity * NdotL;

  vec3 finalColor = baseColor * (ambient * (1.0 - max(-rawNdotL, 0.0) * u_darkness) + diffuse);

  fragColor = vec4(finalColor, u_opacity);
  fragEmissive = vec4(0.0, 0.0, 0.0, u_opacity);
}
`

export const LAMBERT_FRAG = _makeLambertFrag(OBJECT_BLOCK)
export const LAMBERT_SKINNED_FRAG = _makeLambertFrag(SKINNED_OBJECT_BLOCK)

export const LAMBERT_SKINNED_VERT = `#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec4 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in vec4 a_joints;
layout(location = 4) in vec4 a_weights;

${FRAME_BLOCK}
${SKINNED_OBJECT_BLOCK}

out vec3 v_worldPos;
out vec3 v_normal;
out vec2 v_uv;
flat out float v_outlineFlag;

void main() {
  float flag = a_normal.w;
  v_outlineFlag = flag;

  mat4 skinMatrix =
    a_weights.x * u_boneMatrices[int(a_joints.x)] +
    a_weights.y * u_boneMatrices[int(a_joints.y)] +
    a_weights.z * u_boneMatrices[int(a_joints.z)] +
    a_weights.w * u_boneMatrices[int(a_joints.w)];

  vec4 skinnedPos = skinMatrix * vec4(a_position, 1.0);
  vec3 skinnedNorm = normalize((skinMatrix * vec4(a_normal.xyz, 0.0)).xyz);

  vec3 worldPos = skinnedPos.xyz;
  if (flag > 0.5) {
    worldPos = worldPos + skinnedNorm * u_outlineColorAndThickness.w;
  }
  v_worldPos = worldPos;
  v_normal = skinnedNorm;
  v_uv = a_uv;
  gl_Position = u_viewProjection * vec4(worldPos, 1.0);
  gl_Position.z += flag * 0.0001 * gl_Position.w; // Depth bias to prevent outline z-fighting
}
`

// ─── Vertex-color Lambert shaders ────────────────────────────────────

export const LAMBERT_VC_VERT = `#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec4 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in vec4 a_color;
layout(location = 6) in vec4 a_emissive;
layout(location = 7) in vec4 a_tiledNormal;
layout(location = 8) in vec4 a_noiseColor;

${FRAME_BLOCK}
${OBJECT_BLOCK}

out vec3 v_worldPos;
out vec3 v_normal;
out vec2 v_uv;
flat out float v_outlineFlag;
out vec4 v_color;
out vec4 v_emissive;
out vec4 v_tiledNormal;
out vec4 v_noiseColor;

void main() {
  float flag = a_normal.w;
  v_outlineFlag = flag;
  vec3 pos = a_position;
  if (flag > 0.5) {
    pos = pos + normalize(a_normal.xyz) * u_outlineColorAndThickness.w;
  }
  vec4 worldPos = u_worldMatrix * vec4(pos, 1.0);
  v_worldPos = worldPos.xyz;
  v_normal = normalize((u_normalMatrix * vec4(a_normal.xyz, 0.0)).xyz);
  v_uv = a_uv;
  v_color = a_color;
  v_emissive = a_emissive;
  v_tiledNormal = a_tiledNormal;
  v_noiseColor = a_noiseColor;
  gl_Position = u_viewProjection * worldPos;
  gl_Position.z += flag * 0.0001 * gl_Position.w; // Depth bias to prevent outline z-fighting
}
`

const _makeLambertVCFrag = (objectBlock: string) => `#version 300 es
precision highp float;

in vec3 v_worldPos;
in vec3 v_normal;
in vec2 v_uv;
flat in float v_outlineFlag;
in vec4 v_color;
in vec4 v_emissive;
in vec4 v_tiledNormal;
in vec4 v_noiseColor;

${FRAME_BLOCK}
${objectBlock}

uniform vec3 u_baseColor;
uniform float u_opacity;
uniform highp sampler2DShadow u_shadowMap;
uniform bool u_receiveShadow;
uniform float u_emissiveBrightness;
uniform float u_wrapLighting;
uniform float u_darkness;
uniform sampler2D u_aoMap;
uniform float u_aoIntensity;
uniform highp sampler2DArray u_tiledAoArray;
uniform float u_tiledAoScales[16];
uniform highp sampler2DArray u_tiledNormalArray;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragEmissive;

${SHADOW_FUNCTIONS}

// Dot Noise by Xor — procedural value noise via dot products with a gold matrix
float dot_noise(vec3 p) {
  mat3 gold = mat3(
    1.0, 0.6180339887, 0.3819660113,
    0.6180339887, 0.3819660113, 1.0,
    0.3819660113, 1.0, 0.6180339887
  );
  vec3 fp = fract(p);
  vec3 ip = p - fp;
  vec3 sp = fp * fp * (3.0 - 2.0 * fp);
  float c000 = fract(dot(ip + vec3(0.0, 0.0, 0.0), gold[0]) * 13.37) - 0.5;
  float c001 = fract(dot(ip + vec3(0.0, 0.0, 1.0), gold[0]) * 13.37) - 0.5;
  float c010 = fract(dot(ip + vec3(0.0, 1.0, 0.0), gold[0]) * 13.37) - 0.5;
  float c011 = fract(dot(ip + vec3(0.0, 1.0, 1.0), gold[0]) * 13.37) - 0.5;
  float c100 = fract(dot(ip + vec3(1.0, 0.0, 0.0), gold[0]) * 13.37) - 0.5;
  float c101 = fract(dot(ip + vec3(1.0, 0.0, 1.0), gold[0]) * 13.37) - 0.5;
  float c110 = fract(dot(ip + vec3(1.0, 1.0, 0.0), gold[0]) * 13.37) - 0.5;
  float c111 = fract(dot(ip + vec3(1.0, 1.0, 1.0), gold[0]) * 13.37) - 0.5;
  float n0 = mix(mix(c000, c100, sp.x), mix(c010, c110, sp.x), sp.y);
  float n1 = mix(mix(c001, c101, sp.x), mix(c011, c111, sp.x), sp.y);
  float oct1 = mix(n0, n1, sp.z);
  vec3 p2 = p * 2.0;
  vec3 fp2 = fract(p2);
  vec3 ip2 = p2 - fp2;
  vec3 sp2 = fp2 * fp2 * (3.0 - 2.0 * fp2);
  float d000 = fract(dot(ip2 + vec3(0.0, 0.0, 0.0), gold[1]) * 17.31) - 0.5;
  float d001 = fract(dot(ip2 + vec3(0.0, 0.0, 1.0), gold[1]) * 17.31) - 0.5;
  float d010 = fract(dot(ip2 + vec3(0.0, 1.0, 0.0), gold[1]) * 17.31) - 0.5;
  float d011 = fract(dot(ip2 + vec3(0.0, 1.0, 1.0), gold[1]) * 17.31) - 0.5;
  float d100 = fract(dot(ip2 + vec3(1.0, 0.0, 0.0), gold[1]) * 17.31) - 0.5;
  float d101 = fract(dot(ip2 + vec3(1.0, 0.0, 1.0), gold[1]) * 17.31) - 0.5;
  float d110 = fract(dot(ip2 + vec3(1.0, 1.0, 0.0), gold[1]) * 17.31) - 0.5;
  float d111 = fract(dot(ip2 + vec3(1.0, 1.0, 1.0), gold[1]) * 17.31) - 0.5;
  float m0 = mix(mix(d000, d100, sp2.x), mix(d010, d110, sp2.x), sp2.y);
  float m1 = mix(mix(d001, d101, sp2.x), mix(d011, d111, sp2.x), sp2.y);
  float oct2 = mix(m0, m1, sp2.z);
  vec3 p3 = p * 4.0;
  vec3 fp3 = fract(p3);
  vec3 ip3 = p3 - fp3;
  vec3 sp3 = fp3 * fp3 * (3.0 - 2.0 * fp3);
  float e000 = fract(dot(ip3 + vec3(0.0, 0.0, 0.0), gold[2]) * 23.71) - 0.5;
  float e001 = fract(dot(ip3 + vec3(0.0, 0.0, 1.0), gold[2]) * 23.71) - 0.5;
  float e010 = fract(dot(ip3 + vec3(0.0, 1.0, 0.0), gold[2]) * 23.71) - 0.5;
  float e011 = fract(dot(ip3 + vec3(0.0, 1.0, 1.0), gold[2]) * 23.71) - 0.5;
  float e100 = fract(dot(ip3 + vec3(1.0, 0.0, 0.0), gold[2]) * 23.71) - 0.5;
  float e101 = fract(dot(ip3 + vec3(1.0, 0.0, 1.0), gold[2]) * 23.71) - 0.5;
  float e110 = fract(dot(ip3 + vec3(1.0, 1.0, 0.0), gold[2]) * 23.71) - 0.5;
  float e111 = fract(dot(ip3 + vec3(1.0, 1.0, 1.0), gold[2]) * 23.71) - 0.5;
  float q0 = mix(mix(e000, e100, sp3.x), mix(e010, e110, sp3.x), sp3.y);
  float q1 = mix(mix(e001, e101, sp3.x), mix(e011, e111, sp3.x), sp3.y);
  float oct3 = mix(q0, q1, sp3.z);
  return oct1 + oct2 + oct3;
}

void main() {
  if (v_outlineFlag > 0.5) {
    if (gl_FrontFacing || u_outlineColorAndThickness.w <= 0.0) discard;
    fragColor = vec4(u_outlineColorAndThickness.xyz, 1.0);
    fragEmissive = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  vec3 normal = normalize(v_normal);
  vec3 baseColor = u_baseColor * v_color.rgb;
  // Noise color blending: blend between baseColor and noiseColor using dot noise
  float noiseScale = v_noiseColor.a;
  if (noiseScale > 0.0) {
    float t = dot_noise(v_worldPos * noiseScale) / 6.0 + 0.5;
    baseColor = mix(baseColor, u_baseColor * v_noiseColor.rgb, t);
  }
  vec3 emissive = v_emissive.rgb;

  // Tiled AO: decode layer index from color.a, sample array texture using mesh UVs
  uint tiledAoLayerRaw = uint(round(v_color.a * 255.0));
  float tiledAoIntensity = v_emissive.a;
  float tiledAo = 1.0;
  if (tiledAoLayerRaw > 0u) {
    uint layer = tiledAoLayerRaw - 1u;
    float scale = u_tiledAoScales[layer];
    float tiledSample = texture(u_tiledAoArray, vec3(v_uv * scale, float(layer))).r;
    tiledAo = mix(1.0, tiledSample, tiledAoIntensity);
  }

  // Tiled normals: cotangent frame normal mapping
  uint tnLayerRaw = uint(round(v_tiledNormal.x * 255.0));
  if (tnLayerRaw > 0u) {
    float tnIntensity = v_tiledNormal.y;
    float tnScale = v_tiledNormal.z;
    uint tnLayer = tnLayerRaw - 1u;
    vec2 tnUV = v_uv * tnScale;
    vec3 tnSample = texture(u_tiledNormalArray, vec3(tnUV, float(tnLayer))).rgb;
    vec3 tnNormal = tnSample * 2.0 - 1.0;
    // Build cotangent frame from screen-space derivatives
    vec3 dp1 = dFdx(v_worldPos);
    vec3 dp2 = dFdy(v_worldPos);
    vec2 duv1 = dFdx(tnUV);
    vec2 duv2 = dFdy(tnUV);
    vec3 dp2perp = cross(dp2, normal);
    vec3 dp1perp = cross(normal, dp1);
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
    float invmax = inversesqrt(max(dot(T, T), dot(B, B)));
    vec3 tbn_T = T * invmax;
    vec3 tbn_B = B * invmax;
    vec3 perturbedNormal = normalize(tbn_T * tnNormal.x + tbn_B * tnNormal.y + normal * tnNormal.z);
    normal = normalize(mix(normal, perturbedNormal, tnIntensity));
  }

  float ao = mix(1.0, texture(u_aoMap, v_uv).r, u_aoIntensity) * tiledAo;
  vec3 ambient = u_ambientColor * u_ambientIntensity * ao;
  float rawNdotL = dot(normal, u_lightDirection);
  float shadow = u_receiveShadow ? sampleShadow(v_worldPos, max(rawNdotL, 0.0)) : 1.0;
  float wrap = u_wrapLighting;
  float realNdotL = max(rawNdotL, 0.0);
  float wrapNdotL = max(rawNdotL + wrap, 0.0) / (1.0 + wrap);
  float NdotL = wrapNdotL - realNdotL * (1.0 - shadow);
  vec3 diffuse = u_lightColor * u_lightIntensity * NdotL;

  vec3 litColor = baseColor * (ambient * (1.0 - max(-rawNdotL, 0.0) * u_darkness) + diffuse);
  float brightness = dot(emissive, vec3(0.299, 0.587, 0.114));
  vec3 screenEmissive = mix(emissive, vec3(brightness), clamp(brightness, 0.0, 1.0) * u_emissiveBrightness);
  vec3 finalColor = litColor + screenEmissive;

  fragColor = vec4(finalColor, u_opacity);
  fragEmissive = vec4(emissive * u_opacity, u_opacity);
}
`

export const LAMBERT_VC_FRAG = _makeLambertVCFrag(OBJECT_BLOCK)
export const LAMBERT_SKINNED_VC_FRAG = _makeLambertVCFrag(SKINNED_OBJECT_BLOCK)

export const LAMBERT_SKINNED_VC_VERT = `#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec4 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in vec4 a_color;
layout(location = 4) in vec4 a_joints;
layout(location = 5) in vec4 a_weights;
layout(location = 6) in vec4 a_emissive;
layout(location = 7) in vec4 a_tiledNormal;
layout(location = 8) in vec4 a_noiseColor;

${FRAME_BLOCK}
${SKINNED_OBJECT_BLOCK}

out vec3 v_worldPos;
out vec3 v_normal;
out vec2 v_uv;
flat out float v_outlineFlag;
out vec4 v_color;
out vec4 v_emissive;
out vec4 v_tiledNormal;
out vec4 v_noiseColor;

void main() {
  float flag = a_normal.w;
  v_outlineFlag = flag;

  mat4 skinMatrix =
    a_weights.x * u_boneMatrices[int(a_joints.x)] +
    a_weights.y * u_boneMatrices[int(a_joints.y)] +
    a_weights.z * u_boneMatrices[int(a_joints.z)] +
    a_weights.w * u_boneMatrices[int(a_joints.w)];

  vec4 skinnedPos = skinMatrix * vec4(a_position, 1.0);
  vec3 skinnedNorm = normalize((skinMatrix * vec4(a_normal.xyz, 0.0)).xyz);

  vec3 worldPos = skinnedPos.xyz;
  if (flag > 0.5) {
    worldPos = worldPos + skinnedNorm * u_outlineColorAndThickness.w;
  }
  v_worldPos = worldPos;
  v_normal = skinnedNorm;
  v_uv = a_uv;
  v_color = a_color;
  v_emissive = a_emissive;
  v_tiledNormal = a_tiledNormal;
  v_noiseColor = a_noiseColor;
  gl_Position = u_viewProjection * vec4(worldPos, 1.0);
  gl_Position.z += flag * 0.0001 * gl_Position.w; // Depth bias to prevent outline z-fighting
}
`

// ─── Textured Lambert shader ─────────────────────────────────────────

export const LAMBERT_TEXTURED_FRAG = `#version 300 es
precision highp float;

in vec3 v_worldPos;
in vec3 v_normal;
in vec2 v_uv;
flat in float v_outlineFlag;

${FRAME_BLOCK}
${OBJECT_BLOCK}

uniform vec3 u_baseColor;
uniform float u_opacity;
uniform highp sampler2DShadow u_shadowMap;
uniform bool u_receiveShadow;
uniform float u_emissiveBrightness;
uniform float u_wrapLighting;
uniform float u_darkness;
uniform sampler2D u_colorMap;
uniform sampler2D u_aoMap;
uniform float u_aoIntensity;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragEmissive;

${SHADOW_FUNCTIONS}

void main() {
  if (v_outlineFlag > 0.5) {
    if (gl_FrontFacing || u_outlineColorAndThickness.w <= 0.0) discard;
    fragColor = vec4(u_outlineColorAndThickness.xyz, 1.0);
    fragEmissive = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  vec3 normal = normalize(v_normal);

  // Sample texture maps (dummy white textures → identity when no map assigned)
  vec3 texColor = texture(u_colorMap, v_uv).rgb;
  vec3 baseColor = u_baseColor * texColor;

  float ao = mix(1.0, texture(u_aoMap, v_uv).r, u_aoIntensity);
  vec3 ambient = u_ambientColor * u_ambientIntensity * ao;

  float rawNdotL = dot(normal, u_lightDirection);
  float shadow = u_receiveShadow ? sampleShadow(v_worldPos, max(rawNdotL, 0.0)) : 1.0;
  float wrap = u_wrapLighting;
  float realNdotL = max(rawNdotL, 0.0);
  float wrapNdotL = max(rawNdotL + wrap, 0.0) / (1.0 + wrap);
  float NdotL = wrapNdotL - realNdotL * (1.0 - shadow);
  vec3 diffuse = u_lightColor * u_lightIntensity * NdotL;

  vec3 finalColor = baseColor * (ambient * (1.0 - max(-rawNdotL, 0.0) * u_darkness) + diffuse);

  fragColor = vec4(finalColor, u_opacity);
  fragEmissive = vec4(0.0, 0.0, 0.0, u_opacity);
}
`

export const BASIC_SKINNED_VERT = `#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in vec4 a_joints;
layout(location = 4) in vec4 a_weights;

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

// ─── Custom shader builders ──────────────────────────────────────────
//
// These functions inject user-provided GLSL code snippets into the standard material shaders.
// Vertex hook: runs after v_worldPos/v_normal/v_uv computation, before gl_Position.
//   Available mutable variables: v_worldPos (vec3), v_normal (vec3), v_uv (vec2).
// Fragment hook: runs after finalColor/alpha computation, before output.
//   Available mutable variables: finalColor (vec3), alpha (float).

export const buildLambertVert = (customVertex?: string, customUniformsDecl?: string) => `#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec4 a_normal;
layout(location = 2) in vec2 a_uv;

${FRAME_BLOCK}
${OBJECT_BLOCK}

out vec3 v_worldPos;
out vec3 v_normal;
out vec2 v_uv;
flat out float v_outlineFlag;
${customUniformsDecl ?? ''}

void main() {
  float flag = a_normal.w;
  v_outlineFlag = flag;
  vec3 pos = a_position;
  if (flag > 0.5) {
    pos = pos + normalize(a_normal.xyz) * u_outlineColorAndThickness.w;
  }
  vec4 worldPos = u_worldMatrix * vec4(pos, 1.0);
  v_worldPos = worldPos.xyz;
  v_normal = normalize((u_normalMatrix * vec4(a_normal.xyz, 0.0)).xyz);
  v_uv = a_uv;
  ${customVertex ?? ''}
  gl_Position = u_viewProjection * vec4(v_worldPos, 1.0);
  gl_Position.z += flag * 0.0001 * gl_Position.w; // Depth bias to prevent outline z-fighting
}
`

export const buildLambertSkinnedVert = (customVertex?: string, customUniformsDecl?: string) => `#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec4 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in vec4 a_joints;
layout(location = 4) in vec4 a_weights;

${FRAME_BLOCK}
${SKINNED_OBJECT_BLOCK}

out vec3 v_worldPos;
out vec3 v_normal;
out vec2 v_uv;
flat out float v_outlineFlag;
${customUniformsDecl ?? ''}

void main() {
  float flag = a_normal.w;
  v_outlineFlag = flag;

  mat4 skinMatrix =
    a_weights.x * u_boneMatrices[int(a_joints.x)] +
    a_weights.y * u_boneMatrices[int(a_joints.y)] +
    a_weights.z * u_boneMatrices[int(a_joints.z)] +
    a_weights.w * u_boneMatrices[int(a_joints.w)];

  vec4 skinnedPos = skinMatrix * vec4(a_position, 1.0);
  vec3 skinnedNorm = normalize((skinMatrix * vec4(a_normal.xyz, 0.0)).xyz);

  vec3 worldPos = skinnedPos.xyz;
  if (flag > 0.5) {
    worldPos = worldPos + skinnedNorm * u_outlineColorAndThickness.w;
  }
  v_worldPos = worldPos;
  v_normal = skinnedNorm;
  v_uv = a_uv;
  ${customVertex ?? ''}
  gl_Position = u_viewProjection * vec4(v_worldPos, 1.0);
  gl_Position.z += flag * 0.0001 * gl_Position.w; // Depth bias to prevent outline z-fighting
}
`

const _buildLambertCustomFrag = (
  objectBlock: string,
  customFragment?: string,
  customUniformsDecl?: string,
) => `#version 300 es
precision highp float;

in vec3 v_worldPos;
in vec3 v_normal;
in vec2 v_uv;
flat in float v_outlineFlag;

${FRAME_BLOCK}
${objectBlock}

uniform vec3 u_baseColor;
uniform float u_opacity;
uniform highp sampler2DShadow u_shadowMap;
uniform bool u_receiveShadow;
uniform float u_emissiveBrightness;
uniform float u_wrapLighting;
uniform float u_darkness;
${customUniformsDecl ?? ''}

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragEmissive;

${SHADOW_FUNCTIONS}

void main() {
  if (v_outlineFlag > 0.5) {
    if (gl_FrontFacing || u_outlineColorAndThickness.w <= 0.0) discard;
    fragColor = vec4(u_outlineColorAndThickness.xyz, 1.0);
    fragEmissive = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  vec3 normal = normalize(v_normal);
  vec3 baseColor = u_baseColor;

  vec3 ambient = u_ambientColor * u_ambientIntensity;
  float rawNdotL = dot(normal, u_lightDirection);
  float shadow = u_receiveShadow ? sampleShadow(v_worldPos, max(rawNdotL, 0.0)) : 1.0;
  float wrap = u_wrapLighting;
  float realNdotL = max(rawNdotL, 0.0);
  float wrapNdotL = max(rawNdotL + wrap, 0.0) / (1.0 + wrap);
  float NdotL = wrapNdotL - realNdotL * (1.0 - shadow);
  vec3 diffuse = u_lightColor * u_lightIntensity * NdotL;

  vec3 finalColor = baseColor * (ambient * (1.0 - max(-rawNdotL, 0.0) * u_darkness) + diffuse);
  float alpha = u_opacity;
  ${customFragment ?? ''}

  fragColor = vec4(finalColor, alpha);
  fragEmissive = vec4(0.0, 0.0, 0.0, alpha);
}
`

export const buildLambertCustomFrag = (customFragment?: string, customUniformsDecl?: string) =>
  _buildLambertCustomFrag(OBJECT_BLOCK, customFragment, customUniformsDecl)

export const buildLambertSkinnedCustomFrag = (customFragment?: string, customUniformsDecl?: string) =>
  _buildLambertCustomFrag(SKINNED_OBJECT_BLOCK, customFragment, customUniformsDecl)

export const buildBasicVert = (customVertex?: string, customUniformsDecl?: string) => `#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec4 a_normal;
layout(location = 2) in vec2 a_uv;

${FRAME_BLOCK}
${OBJECT_BLOCK}

out vec3 v_worldPos;
out vec3 v_normal;
out vec2 v_uv;
${customUniformsDecl ?? ''}

void main() {
  vec4 worldPos = u_worldMatrix * vec4(a_position, 1.0);
  v_worldPos = worldPos.xyz;
  v_normal = normalize((u_normalMatrix * vec4(a_normal.xyz, 0.0)).xyz);
  v_uv = a_uv;
  ${customVertex ?? ''}
  gl_Position = u_viewProjection * vec4(v_worldPos, 1.0);
}
`

export const buildBasicSkinnedVert = (customVertex?: string, customUniformsDecl?: string) => `#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec4 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in vec4 a_joints;
layout(location = 4) in vec4 a_weights;

${FRAME_BLOCK}
${SKINNED_OBJECT_BLOCK}

out vec3 v_worldPos;
out vec3 v_normal;
out vec2 v_uv;
${customUniformsDecl ?? ''}

void main() {
  mat4 skinMatrix =
    a_weights.x * u_boneMatrices[int(a_joints.x)] +
    a_weights.y * u_boneMatrices[int(a_joints.y)] +
    a_weights.z * u_boneMatrices[int(a_joints.z)] +
    a_weights.w * u_boneMatrices[int(a_joints.w)];

  vec4 skinnedPos = skinMatrix * vec4(a_position, 1.0);
  v_worldPos = skinnedPos.xyz;
  v_normal = normalize((skinMatrix * vec4(a_normal.xyz, 0.0)).xyz);
  v_uv = a_uv;
  ${customVertex ?? ''}
  gl_Position = u_viewProjection * vec4(v_worldPos, 1.0);
}
`

export const buildBasicCustomFrag = (customFragment?: string, customUniformsDecl?: string) => `#version 300 es
precision highp float;

in vec3 v_worldPos;
in vec3 v_normal;
in vec2 v_uv;

uniform vec3 u_baseColor;
uniform float u_opacity;
${customUniformsDecl ?? ''}

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragEmissive;

void main() {
  vec3 finalColor = u_baseColor;
  float alpha = u_opacity;
  ${customFragment ?? ''}
  fragColor = vec4(finalColor, alpha);
  fragEmissive = vec4(0.0, 0.0, 0.0, alpha);
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
