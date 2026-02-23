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
} from '../geometry/primitives'
import { BasicMaterial, LambertMaterial } from '../materials/material'
import { PerspectiveCamera } from '../scene/camera'
import { Group } from '../scene/group'
import { AmbientLight, DirectionalLight } from '../scene/light'
import { Mesh } from '../scene/mesh'
import { Scene } from '../scene/scene'

import type { Skeleton } from '../animation/skeleton'
import type { Geometry } from '../geometry/geometry'
import type {
  PlaneGeometryOptions,
  BoxGeometryOptions,
  SphereGeometryOptions,
  ConeGeometryOptions,
  CylinderGeometryOptions,
  CapsuleGeometryOptions,
  CircleGeometryOptions,
} from '../geometry/primitives'
import type { CustomShader, Material, MaterialOptions, MaterialSide, PaletteEntry } from '../materials/material'
import type { CameraOptions } from '../scene/camera'
import type { AmbientLightOptions, DirectionalLightOptions } from '../scene/light'
import type { MeshOutline } from '../scene/mesh'

// ─── Catalogue ────────────────────────────────────────────────────────────────

export const catalogue: Record<string, new (...args: any[]) => any> = {
  mesh: Mesh,
  group: Group,
  scene: Scene,
  perspectiveCamera: PerspectiveCamera,
  directionalLight: DirectionalLight,
  ambientLight: AmbientLight,
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
  outline?: MeshOutline | number
  maxDistance?: number
}

export interface GroupProps extends NodeProps {}

export interface DirectionalLightProps extends NodeProps {
  args?: [DirectionalLightOptions?]
  color?: [number, number, number]
  intensity?: number
  shadowMapSize?: number
  shadowNear?: number
  shadowFar?: number
  shadowBias?: number
  shadowSlopeBias?: number
}

export interface AmbientLightProps extends NodeProps {
  args?: [AmbientLightOptions?]
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
  palette?: PaletteEntry[]
  vertexColors?: boolean
  emissiveBrightness?: number
  opacity?: number
  transparent?: boolean
  side?: MaterialSide
  customShader?: CustomShader
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
  ambientLight: AmbientLightProps

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
