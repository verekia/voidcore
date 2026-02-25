// Shader Snippets – Reusable WGSL and GLSL code blocks for custom shaders.
//
// These snippets can be used in custom shaders (via the `customShader` material option)
// to access engine-provided utility functions. Each snippet is a string constant that
// can be concatenated into your shader code.
//
//   DOT_NOISE_WGSL – 3-octave procedural value noise function for WGSL.
//                    Signature: fn dot_noise(p: vec3<f32>) -> f32
//                    Returns a value in [-3, 3]. Remap to [0, 1] via: n / 6.0 + 0.5
//
//   DOT_NOISE_GLSL – Same function for GLSL (WebGL2).
//                    Signature: float dot_noise(vec3 p)

/** 3-octave procedural value noise (WGSL). Input: world-space position. Output: [-3, 3]. */
export const DOT_NOISE_WGSL = /* wgsl */ `
fn dot_noise(p: vec3<f32>) -> f32 {
  let gold = mat3x3<f32>(
    vec3<f32>(1.0, 0.6180339887, 0.3819660113),
    vec3<f32>(0.6180339887, 0.3819660113, 1.0),
    vec3<f32>(0.3819660113, 1.0, 0.6180339887)
  );
  let fp = fract(p);
  let ip = p - fp;
  let sp = fp * fp * (3.0 - 2.0 * fp);
  let c000 = fract(dot(ip + vec3<f32>(0.0, 0.0, 0.0), gold[0]) * 13.37) - 0.5;
  let c001 = fract(dot(ip + vec3<f32>(0.0, 0.0, 1.0), gold[0]) * 13.37) - 0.5;
  let c010 = fract(dot(ip + vec3<f32>(0.0, 1.0, 0.0), gold[0]) * 13.37) - 0.5;
  let c011 = fract(dot(ip + vec3<f32>(0.0, 1.0, 1.0), gold[0]) * 13.37) - 0.5;
  let c100 = fract(dot(ip + vec3<f32>(1.0, 0.0, 0.0), gold[0]) * 13.37) - 0.5;
  let c101 = fract(dot(ip + vec3<f32>(1.0, 0.0, 1.0), gold[0]) * 13.37) - 0.5;
  let c110 = fract(dot(ip + vec3<f32>(1.0, 1.0, 0.0), gold[0]) * 13.37) - 0.5;
  let c111 = fract(dot(ip + vec3<f32>(1.0, 1.0, 1.0), gold[0]) * 13.37) - 0.5;
  let n0 = mix(mix(c000, c100, sp.x), mix(c010, c110, sp.x), sp.y);
  let n1 = mix(mix(c001, c101, sp.x), mix(c011, c111, sp.x), sp.y);
  let oct1 = mix(n0, n1, sp.z);
  let p2 = p * 2.0;
  let fp2 = fract(p2);
  let ip2 = p2 - fp2;
  let sp2 = fp2 * fp2 * (3.0 - 2.0 * fp2);
  let d000 = fract(dot(ip2 + vec3<f32>(0.0, 0.0, 0.0), gold[1]) * 17.31) - 0.5;
  let d001 = fract(dot(ip2 + vec3<f32>(0.0, 0.0, 1.0), gold[1]) * 17.31) - 0.5;
  let d010 = fract(dot(ip2 + vec3<f32>(0.0, 1.0, 0.0), gold[1]) * 17.31) - 0.5;
  let d011 = fract(dot(ip2 + vec3<f32>(0.0, 1.0, 1.0), gold[1]) * 17.31) - 0.5;
  let d100 = fract(dot(ip2 + vec3<f32>(1.0, 0.0, 0.0), gold[1]) * 17.31) - 0.5;
  let d101 = fract(dot(ip2 + vec3<f32>(1.0, 0.0, 1.0), gold[1]) * 17.31) - 0.5;
  let d110 = fract(dot(ip2 + vec3<f32>(1.0, 1.0, 0.0), gold[1]) * 17.31) - 0.5;
  let d111 = fract(dot(ip2 + vec3<f32>(1.0, 1.0, 1.0), gold[1]) * 17.31) - 0.5;
  let m0 = mix(mix(d000, d100, sp2.x), mix(d010, d110, sp2.x), sp2.y);
  let m1 = mix(mix(d001, d101, sp2.x), mix(d011, d111, sp2.x), sp2.y);
  let oct2 = mix(m0, m1, sp2.z);
  let p3 = p * 4.0;
  let fp3 = fract(p3);
  let ip3 = p3 - fp3;
  let sp3 = fp3 * fp3 * (3.0 - 2.0 * fp3);
  let e000 = fract(dot(ip3 + vec3<f32>(0.0, 0.0, 0.0), gold[2]) * 23.71) - 0.5;
  let e001 = fract(dot(ip3 + vec3<f32>(0.0, 0.0, 1.0), gold[2]) * 23.71) - 0.5;
  let e010 = fract(dot(ip3 + vec3<f32>(0.0, 1.0, 0.0), gold[2]) * 23.71) - 0.5;
  let e011 = fract(dot(ip3 + vec3<f32>(0.0, 1.0, 1.0), gold[2]) * 23.71) - 0.5;
  let e100 = fract(dot(ip3 + vec3<f32>(1.0, 0.0, 0.0), gold[2]) * 23.71) - 0.5;
  let e101 = fract(dot(ip3 + vec3<f32>(1.0, 0.0, 1.0), gold[2]) * 23.71) - 0.5;
  let e110 = fract(dot(ip3 + vec3<f32>(1.0, 1.0, 0.0), gold[2]) * 23.71) - 0.5;
  let e111 = fract(dot(ip3 + vec3<f32>(1.0, 1.0, 1.0), gold[2]) * 23.71) - 0.5;
  let q0 = mix(mix(e000, e100, sp3.x), mix(e010, e110, sp3.x), sp3.y);
  let q1 = mix(mix(e001, e101, sp3.x), mix(e011, e111, sp3.x), sp3.y);
  let oct3 = mix(q0, q1, sp3.z);
  return oct1 + oct2 + oct3;
}
`

/** 3-octave procedural value noise (GLSL). Input: world-space position. Output: [-3, 3]. */
export const DOT_NOISE_GLSL = /* glsl */ `
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
`
