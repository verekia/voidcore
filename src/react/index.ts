// Voidcore React — Declarative React bindings for VoidCore's 3D engine.
//
// Provides a <Canvas> root component that initializes the engine, plus JSX elements
// for scene graph nodes (<mesh>, <group>, <directionalLight>), geometries, and materials.
// Hooks (useFrame, useEngine, useGLTF, useAnimations) connect React components to the
// engine's render loop and asset pipeline.

export { Canvas, type CanvasProps } from './Canvas.tsx'
export { Html, type HtmlProps } from './Html.tsx'
export { useEngine, useFrame, useLoader, useGLTF, useAnimations } from './hooks.ts'
export { VoidContext, type VoidStore, type FrameCallback } from './context.ts'
export type {
  NodeProps,
  MeshProps,
  GroupProps,
  DirectionalLightProps,
  CameraProps,
  GeometryProps,
  MaterialProps,
  PrimitiveProps,
} from './types.ts'
