// Voidcore — Public API (engine + React bindings)

// Engine
export { Engine, type EngineConfig } from './engine'
export { type Renderer, type RendererConfig, type FrameStats, type ShadowConfig } from './renderer/renderer'

// Scheduler
export { Scheduler, type SchedulerState, type SchedulerCallback, type SchedulerCallbackOptions } from './scheduler'

// Scene
export { Scene } from './scene/scene'
export { Node } from './scene/node'
export { Mesh, type MeshOutline } from './scene/mesh'
export { Group } from './scene/group'
export { PerspectiveCamera, type CameraOptions } from './scene/camera'
export { AmbientLight, type AmbientLightOptions, DirectionalLight, type DirectionalLightOptions } from './scene/light'
export { cloneScene, type CloneOptions, type CloneResult } from './scene/clone'

// Geometry
export {
  Geometry,
  bakePalette,
  clearColoredGeometryCache,
  mergeGeometries,
  mergeStaticIntoSkinned,
  computeSmoothNormals,
} from './geometry/geometry'
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
export { BasicMaterial, LambertMaterial, Material, type MaterialSide, type PaletteEntry } from './materials/material'
export { Texture, type CompressedTextureFormat, type TextureFormat } from './materials/texture'

// Controls
export { OrbitControls } from './controls/orbit'

// Animation
export { Skeleton, type AnimationClip, type KeyframeTrack, AnimationMixer, AnimationAction } from './animation/index'

// Raycasting
export { Raycaster, buildMeshBVH, prebuildBVH, type RaycastHit } from './raycasting/index'

// Loaders
export { loadGLTF, type GLTFResult } from './loaders/gltf'
export { loadKTX2 } from './loaders/ktx2'

// Helpers
export { DirectionalLightHelper } from './helpers/directional-light-helper'

// Overlay
export { createOverlayManager, type OverlayHandle, type OverlayOptions } from './overlay'

// React bindings
export { BakeShadows } from './react/BakeShadows'
export { Canvas, type CanvasProps } from './react/Canvas'
export { Html, type HtmlProps } from './react/Html'
export {
  useEngine,
  useFrame,
  useLoader,
  useGLTF,
  useKTX2,
  useColoredGeometry,
  useAnimations,
  type UseGLTFOptions,
  type ClonedMesh,
} from './react/hooks'
export { VoidContext, type VoidStore, type FrameCallback } from './react/context'
export type {
  NodeProps,
  MeshProps,
  GroupProps,
  AmbientLightProps,
  DirectionalLightProps,
  CameraProps,
  GeometryProps,
  MaterialProps,
  PrimitiveProps,
} from './react/types'

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
