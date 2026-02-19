// Types – JSX element catalogue and prop types for VoidCore's React bindings.
//
// Maps lowercase JSX tag names to engine constructors (e.g. <mesh> → Mesh, <boxGeometry> → BoxGeometry).
// Also defines TypeScript prop interfaces and extends JSX.IntrinsicElements for autocompletion.

import type { ReactNode, Ref } from 'react'

import {
  PlaneGeometry,
  BoxGeometry,
  SphereGeometry,
  ConeGeometry,
  CylinderGeometry,
  CapsuleGeometry,
  CircleGeometry,
} from '../geometry/primitives.ts'
import { BasicMaterial, LambertMaterial } from '../materials/material.ts'
import { PerspectiveCamera } from '../scene/camera.ts'
import { Group } from '../scene/group.ts'
import { DirectionalLight } from '../scene/light.ts'
import { Mesh } from '../scene/mesh.ts'
import { Scene } from '../scene/scene.ts'

import type { Skeleton } from '../animation/skeleton.ts'
import type { Geometry } from '../geometry/geometry.ts'
import type {
  PlaneGeometryOptions,
  BoxGeometryOptions,
  SphereGeometryOptions,
  ConeGeometryOptions,
  CylinderGeometryOptions,
  CapsuleGeometryOptions,
  CircleGeometryOptions,
} from '../geometry/primitives.ts'
import type { Material, MaterialOptions } from '../materials/material.ts'
import type { CameraOptions } from '../scene/camera.ts'
import type { DirectionalLightOptions } from '../scene/light.ts'

// ─── Catalogue ────────────────────────────────────────────────────────────────

export const catalogue: Record<string, new (...args: any[]) => any> = {
  mesh: Mesh,
  group: Group,
  scene: Scene,
  perspectiveCamera: PerspectiveCamera,
  directionalLight: DirectionalLight,
  planeGeometry: PlaneGeometry,
  boxGeometry: BoxGeometry,
  sphereGeometry: SphereGeometry,
  coneGeometry: ConeGeometry,
  cylinderGeometry: CylinderGeometry,
  capsuleGeometry: CapsuleGeometry,
  circleGeometry: CircleGeometry,
  basicMaterial: BasicMaterial,
  lambertMaterial: LambertMaterial,
}

// ─── Category sets ────────────────────────────────────────────────────────────

export const GEOMETRY_TYPES = new Set([
  'planeGeometry',
  'boxGeometry',
  'sphereGeometry',
  'coneGeometry',
  'cylinderGeometry',
  'capsuleGeometry',
  'circleGeometry',
])

export const MATERIAL_TYPES = new Set(['basicMaterial', 'lambertMaterial'])

// ─── Prop interfaces ─────────────────────────────────────────────────────────

export interface BaseProps {
  key?: string | number
  ref?: Ref<any>
  children?: ReactNode
  attach?: string
}

export interface NodeProps extends BaseProps {
  position?: [number, number, number]
  rotation?: [number, number, number, number]
  scale?: [number, number, number] | number
  visible?: boolean
  castShadow?: boolean
  receiveShadow?: boolean
  name?: string
  frustumCulled?: boolean
  onClick?: (event: any) => void
  onPointerOver?: (event: any) => void
  onPointerOut?: (event: any) => void
  onPointerDown?: (event: any) => void
  onPointerUp?: (event: any) => void
  onPointerMove?: (event: any) => void
}

export interface MeshProps extends NodeProps {
  geometry?: Geometry
  material?: Material
  skeleton?: Skeleton
}

export interface GroupProps extends NodeProps {}

export interface DirectionalLightProps extends NodeProps {
  args?: [DirectionalLightOptions?]
  color?: [number, number, number]
  intensity?: number
}

export interface CameraProps extends NodeProps {
  args?: [CameraOptions?]
  fov?: number
  near?: number
  far?: number
}

export interface GeometryProps extends BaseProps {
  args?: [any?]
}

export interface MaterialProps extends BaseProps {
  args?: [MaterialOptions?]
  color?: [number, number, number]
  opacity?: number
  transparent?: boolean
  vertexColors?: boolean
}

export interface PrimitiveProps extends NodeProps {
  object: any
}

// ─── JSX IntrinsicElements ────────────────────────────────────────────────────

type VoidElements = {
  mesh: MeshProps
  group: GroupProps
  scene: NodeProps
  perspectiveCamera: CameraProps
  directionalLight: DirectionalLightProps

  planeGeometry: GeometryProps & { args?: [PlaneGeometryOptions?] }
  boxGeometry: GeometryProps & { args?: [BoxGeometryOptions?] }
  sphereGeometry: GeometryProps & { args?: [SphereGeometryOptions?] }
  coneGeometry: GeometryProps & { args?: [ConeGeometryOptions?] }
  cylinderGeometry: GeometryProps & { args?: [CylinderGeometryOptions?] }
  capsuleGeometry: GeometryProps & { args?: [CapsuleGeometryOptions?] }
  circleGeometry: GeometryProps & { args?: [CircleGeometryOptions?] }

  basicMaterial: MaterialProps
  lambertMaterial: MaterialProps

  primitive: PrimitiveProps
}

declare module 'react' {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace JSX {
    interface IntrinsicElements extends VoidElements {}
  }
}
