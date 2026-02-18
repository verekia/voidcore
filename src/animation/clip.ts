import { vec3Lerp, quatSlerp } from '../math/index.ts'

import type { Skeleton } from './skeleton.ts'

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

// Scratch arrays for interpolation
const _v3a = new Float32Array(3)
const _v3b = new Float32Array(3)
const _v3out = new Float32Array(3)
const _q4a = new Float32Array(4)
const _q4b = new Float32Array(4)
const _q4out = new Float32Array(4)

const binarySearch = (times: Float32Array, t: number): number => {
  let lo = 0
  let hi = times.length - 1
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1
    if (times[mid]! < t) lo = mid + 1
    else hi = mid - 1
  }
  return lo
}

export const sampleClip = (clip: AnimationClip, skeleton: Skeleton, time: number) => {
  const t = time % clip.duration

  for (let i = 0; i < clip.tracks.length; i++) {
    const track = clip.tracks[i]!
    const bone = skeleton.bones[track.boneIndex]
    if (!bone) continue

    const times = track.times
    const values = track.values
    const numKeyframes = times.length

    if (numKeyframes === 0) continue

    // Clamp to range
    if (t <= times[0]!) {
      applyKeyframe(bone, track, 0)
      continue
    }
    if (t >= times[numKeyframes - 1]!) {
      applyKeyframe(bone, track, numKeyframes - 1)
      continue
    }

    // Binary search for the right keyframe pair
    const nextIdx = binarySearch(times, t)
    const prevIdx = nextIdx - 1

    if (prevIdx < 0) {
      applyKeyframe(bone, track, 0)
      continue
    }

    if (track.interpolation === 'STEP') {
      applyKeyframe(bone, track, prevIdx)
      continue
    }

    // LINEAR interpolation
    const t0 = times[prevIdx]!
    const t1 = times[nextIdx]!
    const alpha = t1 > t0 ? (t - t0) / (t1 - t0) : 0

    const stride = track.path === 'rotation' ? 4 : 3
    const offA = prevIdx * stride
    const offB = nextIdx * stride

    if (track.path === 'rotation') {
      _q4a[0] = values[offA]!
      _q4a[1] = values[offA + 1]!
      _q4a[2] = values[offA + 2]!
      _q4a[3] = values[offA + 3]!
      _q4b[0] = values[offB]!
      _q4b[1] = values[offB + 1]!
      _q4b[2] = values[offB + 2]!
      _q4b[3] = values[offB + 3]!
      quatSlerp(_q4out, _q4a, _q4b, alpha)
      bone.rotation[0] = _q4out[0]!
      bone.rotation[1] = _q4out[1]!
      bone.rotation[2] = _q4out[2]!
      bone.rotation[3] = _q4out[3]!
    } else {
      _v3a[0] = values[offA]!
      _v3a[1] = values[offA + 1]!
      _v3a[2] = values[offA + 2]!
      _v3b[0] = values[offB]!
      _v3b[1] = values[offB + 1]!
      _v3b[2] = values[offB + 2]!
      vec3Lerp(_v3out, _v3a, _v3b, alpha)

      const target = track.path === 'translation' ? bone.position : bone.scale
      target[0] = _v3out[0]!
      target[1] = _v3out[1]!
      target[2] = _v3out[2]!
    }

    bone._dirtyLocal = true
  }
}

const applyKeyframe = (
  bone: { position: Float32Array; rotation: Float32Array; scale: Float32Array; _dirtyLocal: boolean },
  track: KeyframeTrack,
  idx: number,
) => {
  const values = track.values
  const stride = track.path === 'rotation' ? 4 : 3
  const off = idx * stride

  if (track.path === 'rotation') {
    bone.rotation[0] = values[off]!
    bone.rotation[1] = values[off + 1]!
    bone.rotation[2] = values[off + 2]!
    bone.rotation[3] = values[off + 3]!
  } else {
    const target = track.path === 'translation' ? bone.position : bone.scale
    target[0] = values[off]!
    target[1] = values[off + 1]!
    target[2] = values[off + 2]!
  }

  bone._dirtyLocal = true
}
