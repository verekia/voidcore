// Voidcore — Public API

// Engine
export { createEngine, Engine, type EngineConfig } from './engine.ts'
export { type Renderer, type RendererConfig, type FrameStats } from './renderer/renderer.ts'

// Scene
export { createScene, Scene } from './scene/scene.ts'
export { Node } from './scene/node.ts'
export { createMesh, Mesh } from './scene/mesh.ts'
export { createGroup, Group } from './scene/group.ts'
export { createPerspectiveCamera, PerspectiveCamera } from './scene/camera.ts'
export { createDirectionalLight, DirectionalLight } from './scene/light.ts'

// Geometry
export { createGeometry, Geometry } from './geometry/geometry.ts'
export {
  createPlaneGeometry,
  createBoxGeometry,
  createSphereGeometry,
  createConeGeometry,
  createCylinderGeometry,
  createCapsuleGeometry,
  createCircleGeometry,
} from './geometry/primitives.ts'

// Materials
export { createBasicMaterial, createLambertMaterial, Material } from './materials/material.ts'

// Controls
export { createOrbitControls, OrbitControls } from './controls/orbit.ts'

// Loaders
export { loadGLTF } from './loaders/gltf.ts'

// Math
export {
  type Vec2,
  type Vec3,
  type Vec4,
  type Quat,
  type Mat4,
  type AABB,
  vec3Create,
  vec3Set,
  vec3Copy,
  vec3Add,
  vec3Sub,
  vec3Scale,
  vec3Normalize,
  vec3Cross,
  vec3Dot,
  vec3Length,
  vec3Lerp,
  vec3TransformMat4,
  vec3TransformQuat,
  vec4Create,
  vec4Set,
  mat4Create,
  mat4Identity,
  mat4Multiply,
  mat4Invert,
  mat4Perspective,
  mat4LookAt,
  mat4Compose,
  quatCreate,
  quatFromAxisAngle,
  quatSlerp,
  quatNormalize,
  VEC3_ZERO,
  VEC3_ONE,
  VEC3_UP,
  VEC3_FORWARD,
  VEC3_RIGHT,
  QUAT_IDENTITY,
  MAT4_IDENTITY,
} from './math/index.ts'
