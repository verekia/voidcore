// Voidcore — Public API

// Engine
export { Engine, type EngineConfig } from './engine.ts'
export { type Renderer, type RendererConfig, type FrameStats, type ShadowConfig } from './renderer/renderer.ts'

// Scheduler
export { Scheduler, type SchedulerState, type SchedulerCallback, type SchedulerCallbackOptions } from './scheduler.ts'

// Scene
export { Scene } from './scene/scene.ts'
export { Node } from './scene/node.ts'
export { Mesh } from './scene/mesh.ts'
export { Group } from './scene/group.ts'
export { PerspectiveCamera, type CameraOptions } from './scene/camera.ts'
export { DirectionalLight, type DirectionalLightOptions } from './scene/light.ts'

// Geometry
export { Geometry } from './geometry/geometry.ts'
export {
  PlaneGeometry,
  BoxGeometry,
  SphereGeometry,
  ConeGeometry,
  CylinderGeometry,
  CapsuleGeometry,
  CircleGeometry,
} from './geometry/primitives.ts'

// Materials
export { BasicMaterial, LambertMaterial, Material } from './materials/material.ts'

// Controls
export { OrbitControls } from './controls/orbit.ts'

// Animation
export { Skeleton, type AnimationClip, type KeyframeTrack, AnimationMixer, AnimationAction } from './animation/index.ts'

// Raycasting
export { Raycaster, buildMeshBVH, type RaycastHit } from './raycasting/index.ts'

// Loaders
export { loadGLTF } from './loaders/gltf.ts'

// Overlay
export { createOverlayManager, type OverlayHandle, type OverlayOptions } from './overlay.ts'

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
