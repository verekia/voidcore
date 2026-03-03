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
export { Sprite } from './scene/sprite'
export { Group } from './scene/group'
export { PerspectiveCamera, type CameraOptions } from './scene/camera'
export { AmbientLight, type AmbientLightOptions, DirectionalLight, type DirectionalLightOptions } from './scene/light'
export { GIProbeGrid, type GIProbeGridOptions, type ProbeData } from './scene/gi-probes'
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
export { generateGrass, createGrassMaterial, type GrassOptions } from './geometry/grass'

// Materials
export {
  BasicMaterial,
  LambertMaterial,
  Material,
  type CustomShader,
  type MaterialSide,
  type PaletteEntry,
} from './materials/material'
export { SpriteMaterial, type SpriteMaterialOptions } from './materials/sprite-material'
export { Texture, type CompressedTextureFormat, type TextureFormat } from './materials/texture'

// Controls
export { OrbitControls } from './controls/orbit'

// Animation
export { Skeleton, type AnimationClip, type KeyframeTrack, AnimationMixer, AnimationAction } from './animation/index'

// Raycasting
export { Raycaster, buildMeshBVH, prebuildBVH, createRaycastHit, type RaycastHit } from './raycasting/index'

// Loaders
export { loadGLTF, type GLTFResult } from './loaders/gltf'
export { loadKTX2 } from './loaders/ktx2'

// Shader snippets
export { DOT_NOISE_WGSL, DOT_NOISE_GLSL } from './renderer/shaders'

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
  useColoredStaticGeometry,
  useAnimations,
  useGrass,
  useGIProbes,
  type UseGLTFOptions,
  type UseGIProbesOptions,
  type ClonedMesh,
} from './react/hooks'
export { VoidContext, type VoidStore, type FrameCallback } from './react/context'
export type {
  NodeProps,
  MeshProps,
  SpriteProps,
  GroupProps,
  AmbientLightProps,
  DirectionalLightProps,
  CameraProps,
  GeometryProps,
  MaterialProps,
  SpriteMaterialProps,
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
  vec2Create,
  vec2Set,
  vec3Create,
  vec3Set,
  vec3Copy,
  vec3Add,
  vec3Sub,
  vec3Scale,
  vec3Negate,
  vec3Normalize,
  vec3Cross,
  vec3Dot,
  vec3Length,
  vec3LengthSq,
  vec3Distance,
  vec3DistanceSq,
  vec3Min,
  vec3Max,
  vec3Lerp,
  vec3TransformMat4,
  vec3TransformQuat,
  vec4Create,
  vec4Set,
  vec4TransformMat4,
  mat4Create,
  mat4Identity,
  mat4Copy,
  mat4Multiply,
  mat4Transpose,
  mat4Invert,
  mat4Perspective,
  mat4LookAt,
  mat4Compose,
  mat4GetTranslation,
  quatCreate,
  quatIdentity,
  quatCopy,
  quatMultiply,
  quatConjugate,
  quatInvert,
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
