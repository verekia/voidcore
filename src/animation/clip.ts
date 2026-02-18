export interface KeyframeTrack {
  boneIndex: number
  path: 'translation' | 'rotation' | 'scale'
  times: Float32Array
  values: Float32Array
  interpolation: 'LINEAR' | 'STEP'
}

export interface AnimationClip {
  name: string
  duration: number
  tracks: KeyframeTrack[]
}
