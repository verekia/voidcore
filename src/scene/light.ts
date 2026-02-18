import { Node } from './node.ts'

export class DirectionalLight extends Node {
  color: [number, number, number]
  intensity: number

  constructor(color: [number, number, number] = [1, 1, 1], intensity = 1.0, castShadow = true) {
    super()
    this.type = 'directionalLight'
    this.color = color
    this.intensity = intensity
    this.castShadow = castShadow
    this.frustumCulled = false
    this.receiveShadow = false
  }
}

export interface DirectionalLightOptions {
  color?: [number, number, number]
  intensity?: number
  castShadow?: boolean
}

export const createDirectionalLight = (opts: DirectionalLightOptions = {}): DirectionalLight =>
  new DirectionalLight(opts.color ?? [1, 1, 1], opts.intensity ?? 1.0, opts.castShadow ?? true)
