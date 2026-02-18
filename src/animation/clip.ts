// Animation Clip – Data structures for keyframe animation.
//
// An AnimationClip is a named collection of KeyframeTracks with a total duration.
// Each KeyframeTrack stores an array of timestamps and corresponding values (positions,
// rotations, or scales) for a single bone. During playback, the mixer samples between
// keyframes to compute smooth in-between poses.
//
// KeyframeTrack – One channel of animation (e.g. "bone #3, translation") with time/value arrays.
// AnimationClip – A full animation (e.g. "walk cycle") containing multiple tracks.

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
