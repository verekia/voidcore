// Voidcore — Public API

// Engine
export { Engine, type EngineConfig } from './engine'
export { type Renderer, type RendererConfig, type FrameStats, type ShadowConfig } from './renderer/renderer'

// Scheduler
export { Scheduler, type SchedulerState, type SchedulerCallback, type SchedulerCallbackOptions } from './scheduler'

// Scene
export { Scene } from './scene/scene'
export { Node } from './scene/node'
export { Mesh } from './scene/mesh'
export { Group } from './scene/group'
export { PerspectiveCamera, type CameraOptions } from './scene/camera'
export { AmbientLight, type AmbientLightOptions, DirectionalLight, type DirectionalLightOptions } from './scene/light'
export { cloneScene, type CloneOptions, type CloneResult } from './scene/clone'

// Geometry
export { Geometry, mergeGeometries } from './geometry/geometry'
export {
  PlaneGeometry,
  BoxGeometry,
  SphereGeometry,
  ConeGeometry,
  CylinderGeometry,
  CapsuleGeometry,
  CircleGeometry,
} from './geometry/primitives'

// Materials
export { BasicMaterial, LambertMaterial, Material } from './materials/material'
export { Texture } from './materials/texture'

// Controls
export { OrbitControls } from './controls/orbit'

// Animation
export { Skeleton, type AnimationClip, type KeyframeTrack, AnimationMixer, AnimationAction } from './animation/index'

// Raycasting
export { Raycaster, buildMeshBVH, prebuildBVH, type RaycastHit } from './raycasting/index'

// Loaders
export { loadGLTF } from './loaders/gltf'
export { loadKTX2 } from './loaders/ktx2'

// Overlay
export { createOverlayManager, type OverlayHandle, type OverlayOptions } from './overlay'

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
} from './math/index'
