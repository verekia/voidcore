// Voidcore React — Declarative React bindings for VoidCore's 3D engine.
//
// Provides a <Canvas> root component that initializes the engine, plus JSX elements
// for scene graph nodes (<mesh>, <group>, <directionalLight>, <ambientLight>), geometries, and materials.
// Hooks (useFrame, useEngine, useGLTF, useAnimations) connect React components to the
// engine's render loop and asset pipeline.

export { BakeShadows } from './BakeShadows'
export { Canvas, type CanvasProps } from './Canvas'
export { DirectionalLightHelper, type DirectionalLightHelperProps } from './DirectionalLightHelper'
export { Html, type HtmlProps } from './Html'
export { useEngine, useFrame, useLoader, useGLTF, useKTX2, useAnimations } from './hooks'
export { VoidContext, type VoidStore, type FrameCallback } from './context'
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
} from './types'
