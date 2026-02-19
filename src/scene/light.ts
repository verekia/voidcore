// Light – Light sources that illuminate the scene.
//
// A DirectionalLight simulates a distant light source (like the sun) where all rays are
// parallel. Its direction is determined by its position in the scene (the position vector
// is treated as a direction). The light has a color and intensity that are multiplied with
// the surface's diffuse color in the shader.
//
// new DirectionalLight() – Constructor with default white light at full intensity.

import { Node } from './node.ts'

export interface DirectionalLightOptions {
  color?: [number, number, number]
  intensity?: number
  castShadow?: boolean
}

export class DirectionalLight extends Node {
  color: [number, number, number]
  intensity: number

  constructor(opts: DirectionalLightOptions = {}) {
    super()
    this.type = 'directionalLight'
    this.color = opts.color ?? [1, 1, 1]
    this.intensity = opts.intensity ?? 1.0
    this.castShadow = opts.castShadow ?? true
    this.frustumCulled = false
    this.receiveShadow = false
  }
}
