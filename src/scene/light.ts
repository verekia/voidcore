// Light – Light sources that illuminate the scene.
//
// A DirectionalLight simulates a distant light source (like the sun) where all rays are
// parallel. Its direction is determined by its position in the scene (the position vector
// is treated as a direction). The light has a color and intensity that are multiplied with
// the surface's diffuse color in the shader. Shadow configuration (ortho box size, near/far,
// bias) lives on the light itself — the renderer reads these properties each frame.
//
// An AmbientLight provides constant low-level illumination that affects all objects equally,
// regardless of their position or orientation. It has no direction — just a color and intensity.
//
// new DirectionalLight() – Constructor with default white light at full intensity.
// new AmbientLight()     – Constructor with default white ambient light at full intensity.

import { Node } from './node'

export interface DirectionalLightOptions {
  color?: [number, number, number]
  intensity?: number
  castShadow?: boolean
  shadowMapSize?: number
  shadowNear?: number
  shadowFar?: number
  shadowBias?: number
  shadowSlopeBias?: number
}

export class DirectionalLight extends Node {
  color: [number, number, number]
  intensity: number
  shadowMapSize: number
  shadowNear: number
  shadowFar: number
  shadowBias: number
  shadowSlopeBias: number

  constructor(opts: DirectionalLightOptions = {}) {
    super()
    this.type = 'directionalLight'
    this.color = opts.color ?? [1, 1, 1]
    this.intensity = opts.intensity ?? 1.0
    this.castShadow = opts.castShadow ?? false
    this.shadowMapSize = opts.shadowMapSize ?? 200
    this.shadowNear = opts.shadowNear ?? 1
    this.shadowFar = opts.shadowFar ?? 300
    this.shadowBias = opts.shadowBias ?? 0.001
    this.shadowSlopeBias = opts.shadowSlopeBias ?? 0.005
    this.frustumCulled = false
    this.receiveShadow = false
  }
}

export interface AmbientLightOptions {
  color?: [number, number, number]
  intensity?: number
}

export class AmbientLight extends Node {
  color: [number, number, number]
  intensity: number

  constructor(opts: AmbientLightOptions = {}) {
    super()
    this.type = 'ambientLight'
    this.color = opts.color ?? [1, 1, 1]
    this.intensity = opts.intensity ?? 1.0
    this.castShadow = false
    this.frustumCulled = false
    this.receiveShadow = false
  }
}
