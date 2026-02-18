// Animation Mixer – Plays and blends skeletal animations.
//
// The AnimationMixer drives one or more AnimationActions on a Skeleton. Each frame it:
//   1. Advances time on each playing action (handling looping, ping-pong, etc.)
//   2. Updates fade-in/fade-out weights for smooth transitions between animations
//   3. Samples each action's clip at the current time to get bone poses
//   4. Blends all active actions together using weighted interpolation
//   5. Writes the final blended pose back to the skeleton's bones
//
// AnimationAction   – Controls playback of a single clip (play, stop, fadeIn, fadeOut, crossFadeTo).
// AnimationMixer    – Manages multiple actions, blends them, and applies the result to bones.
// createAnimationMixer() – Factory that creates a mixer for a given skeleton.

import { quatSlerp } from '../math/index.ts'

import type { AnimationClip, KeyframeTrack } from './clip.ts'
import type { Skeleton } from './skeleton.ts'

const easeInOutQuad = (t: number): number => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2)

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

// ─── Action ──────────────────────────────────────────────────────────

export class AnimationAction {
  clip: AnimationClip
  loop: 'repeat' | 'once' | 'pingpong' = 'repeat'
  timeScale = 1.0
  weight = 1.0
  time = 0
  paused = false
  playing = false

  _fadeFrom = 0
  _fadeTo = 0
  _fadeDuration = 0
  _fadeElapsed = 0
  _fading = false
  _cachedIndices: number[] | null = null

  constructor(clip: AnimationClip) {
    this.clip = clip
  }

  play(): this {
    this.playing = true
    this.time = 0
    this._cachedIndices = null
    return this
  }

  stop(): this {
    this.playing = false
    this.weight = 0
    this.time = 0
    return this
  }

  fadeIn(duration: number): this {
    this._fadeFrom = 0
    this._fadeTo = 1
    this._fadeDuration = duration
    this._fadeElapsed = 0
    this._fading = true
    this.weight = 0
    return this
  }

  fadeOut(duration: number): this {
    this._fadeFrom = this.weight
    this._fadeTo = 0
    this._fadeDuration = duration
    this._fadeElapsed = 0
    this._fading = true
    return this
  }

  crossFadeTo(target: AnimationAction, duration: number): this {
    target.weight = 0
    target.play()
    target.fadeIn(duration)
    this.fadeOut(duration)
    return this
  }
}

// ─── Mixer ───────────────────────────────────────────────────────────

export class AnimationMixer {
  skeleton: Skeleton
  private _actions: AnimationAction[] = []

  // Rest pose (captured at creation)
  private _restPos: Float32Array
  private _restRot: Float32Array
  private _restScale: Float32Array

  // Blended result
  private _blendedPos: Float32Array
  private _blendedRot: Float32Array
  private _blendedScale: Float32Array

  // Per-action sample buffer
  private _samplePos: Float32Array
  private _sampleRot: Float32Array
  private _sampleScale: Float32Array

  // Scratch quaternions for slerp
  private _q4a = new Float32Array(4)
  private _q4b = new Float32Array(4)

  private _active: AnimationAction[] = []

  constructor(skeleton: Skeleton) {
    this.skeleton = skeleton
    const n = skeleton.bones.length

    this._restPos = new Float32Array(n * 3)
    this._restRot = new Float32Array(n * 4)
    this._restScale = new Float32Array(n * 3)
    this._blendedPos = new Float32Array(n * 3)
    this._blendedRot = new Float32Array(n * 4)
    this._blendedScale = new Float32Array(n * 3)
    this._samplePos = new Float32Array(n * 3)
    this._sampleRot = new Float32Array(n * 4)
    this._sampleScale = new Float32Array(n * 3)

    // Capture rest pose from current bone transforms
    for (let i = 0; i < n; i++) {
      const bone = skeleton.bones[i]!
      const p = i * 3
      const r = i * 4
      this._restPos[p] = bone.position[0]!
      this._restPos[p + 1] = bone.position[1]!
      this._restPos[p + 2] = bone.position[2]!
      this._restRot[r] = bone.rotation[0]!
      this._restRot[r + 1] = bone.rotation[1]!
      this._restRot[r + 2] = bone.rotation[2]!
      this._restRot[r + 3] = bone.rotation[3]!
      this._restScale[p] = bone.scale[0]!
      this._restScale[p + 1] = bone.scale[1]!
      this._restScale[p + 2] = bone.scale[2]!
    }
  }

  clipAction(clip: AnimationClip): AnimationAction {
    const action = new AnimationAction(clip)
    this._actions.push(action)
    return action
  }

  update(dt: number) {
    // 1. Advance times, update fades, collect active actions
    const active = this._active
    active.length = 0
    let totalWeight = 0

    for (let i = 0; i < this._actions.length; i++) {
      const action = this._actions[i]!
      if (!action.playing) continue

      // Advance time
      if (!action.paused) {
        action.time += dt * action.timeScale

        if (action.clip.duration > 0) {
          if (action.loop === 'repeat') {
            action.time = action.time % action.clip.duration
            if (action.time < 0) action.time += action.clip.duration
          } else if (action.loop === 'once') {
            if (action.time >= action.clip.duration) {
              action.time = action.clip.duration
              if (!action._fading) action.playing = false
            }
          } else if (action.loop === 'pingpong') {
            const d = action.clip.duration
            const cycle = Math.floor(action.time / d)
            action.time = action.time % d
            if (action.time < 0) action.time += d
            if (cycle % 2 === 1) action.time = d - action.time
          }
        }
      }

      // Update fade
      if (action._fading) {
        action._fadeElapsed += dt
        let t = action._fadeDuration > 0 ? action._fadeElapsed / action._fadeDuration : 1
        if (t >= 1) {
          t = 1
          action._fading = false
        }
        action.weight = action._fadeFrom + (action._fadeTo - action._fadeFrom) * easeInOutQuad(t)

        if (!action._fading && action._fadeTo <= 0) {
          action.playing = false
          action.weight = 0
          continue
        }
      }

      if (action.weight > 0) {
        active.push(action)
        totalWeight += action.weight
      }
    }

    if (active.length === 0 || totalWeight <= 0) return

    // 2. Sample and blend incrementally
    const n = this.skeleton.bones.length
    let accWeight = 0

    for (let ai = 0; ai < active.length; ai++) {
      const action = active[ai]!
      const nw = action.weight / totalWeight
      accWeight += nw

      // Initialize sample buffer to rest pose, then override with tracks
      this._samplePos.set(this._restPos)
      this._sampleRot.set(this._restRot)
      this._sampleScale.set(this._restScale)
      this._sampleClipInto(action.clip, action.time, action)

      if (ai === 0) {
        // First action: copy directly
        this._blendedPos.set(this._samplePos)
        this._blendedRot.set(this._sampleRot)
        this._blendedScale.set(this._sampleScale)
      } else {
        // Incremental blend: t = nw / accWeight
        const t = accWeight > 0 ? nw / accWeight : 0

        for (let bi = 0; bi < n; bi++) {
          const p = bi * 3
          const r = bi * 4

          // Lerp position
          this._blendedPos[p] = this._blendedPos[p]! + (this._samplePos[p]! - this._blendedPos[p]!) * t
          this._blendedPos[p + 1] = this._blendedPos[p + 1]! + (this._samplePos[p + 1]! - this._blendedPos[p + 1]!) * t
          this._blendedPos[p + 2] = this._blendedPos[p + 2]! + (this._samplePos[p + 2]! - this._blendedPos[p + 2]!) * t

          // Slerp rotation
          this._q4a[0] = this._blendedRot[r]!
          this._q4a[1] = this._blendedRot[r + 1]!
          this._q4a[2] = this._blendedRot[r + 2]!
          this._q4a[3] = this._blendedRot[r + 3]!
          this._q4b[0] = this._sampleRot[r]!
          this._q4b[1] = this._sampleRot[r + 1]!
          this._q4b[2] = this._sampleRot[r + 2]!
          this._q4b[3] = this._sampleRot[r + 3]!
          quatSlerp(this._q4a, this._q4a, this._q4b, t)
          this._blendedRot[r] = this._q4a[0]!
          this._blendedRot[r + 1] = this._q4a[1]!
          this._blendedRot[r + 2] = this._q4a[2]!
          this._blendedRot[r + 3] = this._q4a[3]!

          // Lerp scale
          this._blendedScale[p] = this._blendedScale[p]! + (this._sampleScale[p]! - this._blendedScale[p]!) * t
          this._blendedScale[p + 1] =
            this._blendedScale[p + 1]! + (this._sampleScale[p + 1]! - this._blendedScale[p + 1]!) * t
          this._blendedScale[p + 2] =
            this._blendedScale[p + 2]! + (this._sampleScale[p + 2]! - this._blendedScale[p + 2]!) * t
        }
      }
    }

    // 3. Apply blended pose to bones
    for (let bi = 0; bi < n; bi++) {
      const bone = this.skeleton.bones[bi]!
      const p = bi * 3
      const r = bi * 4

      bone.position[0] = this._blendedPos[p]!
      bone.position[1] = this._blendedPos[p + 1]!
      bone.position[2] = this._blendedPos[p + 2]!
      bone.rotation[0] = this._blendedRot[r]!
      bone.rotation[1] = this._blendedRot[r + 1]!
      bone.rotation[2] = this._blendedRot[r + 2]!
      bone.rotation[3] = this._blendedRot[r + 3]!
      bone.scale[0] = this._blendedScale[p]!
      bone.scale[1] = this._blendedScale[p + 1]!
      bone.scale[2] = this._blendedScale[p + 2]!
      bone._dirtyLocal = true
    }
    this.skeleton._dirty = true
  }

  // ─── Internal clip sampling ─────────────────────────────────────

  private _sampleClipInto(clip: AnimationClip, time: number, action: AnimationAction) {
    const t = clip.duration > 0 ? time % clip.duration : 0

    // Lazily allocate cached indices array
    let cached = action._cachedIndices
    if (!cached) {
      cached = Array.from({ length: clip.tracks.length }, () => 0)
      action._cachedIndices = cached
    }

    for (let ti = 0; ti < clip.tracks.length; ti++) {
      const track = clip.tracks[ti]!
      const times = track.times
      const numKf = times.length

      if (numKf === 0) continue

      let idx0: number
      let idx1: number
      let alpha: number

      if (t <= times[0]!) {
        idx0 = 0
        idx1 = 0
        alpha = 0
      } else if (t >= times[numKf - 1]!) {
        idx0 = numKf - 1
        idx1 = numKf - 1
        alpha = 0
      } else {
        // Try cached index first
        const ci = cached[ti]!
        if (ci < numKf && ci > 0 && times[ci - 1]! <= t && t < times[ci]!) {
          idx1 = ci
        } else if (ci + 1 < numKf && ci >= 0 && times[ci]! <= t && t < times[ci + 1]!) {
          idx1 = ci + 1
        } else {
          idx1 = binarySearch(times, t)
        }
        cached[ti] = idx1
        idx0 = idx1 - 1
        if (idx0 < 0) {
          idx0 = 0
          alpha = 0
        } else {
          const t0 = times[idx0]!
          const t1 = times[idx1]!
          alpha = t1 > t0 ? (t - t0) / (t1 - t0) : 0
        }
      }

      if (track.interpolation === 'STEP') alpha = 0

      this._writeTrackSample(track, idx0, idx1, alpha)
    }
  }

  private _writeTrackSample(track: KeyframeTrack, idx0: number, idx1: number, alpha: number) {
    const values = track.values
    const bi = track.boneIndex
    const stride = track.path === 'rotation' ? 4 : 3
    const off0 = idx0 * stride
    const off1 = idx1 * stride

    if (track.path === 'translation') {
      const p = bi * 3
      if (alpha === 0) {
        this._samplePos[p] = values[off0]!
        this._samplePos[p + 1] = values[off0 + 1]!
        this._samplePos[p + 2] = values[off0 + 2]!
      } else {
        this._samplePos[p] = values[off0]! + (values[off1]! - values[off0]!) * alpha
        this._samplePos[p + 1] = values[off0 + 1]! + (values[off1 + 1]! - values[off0 + 1]!) * alpha
        this._samplePos[p + 2] = values[off0 + 2]! + (values[off1 + 2]! - values[off0 + 2]!) * alpha
      }
    } else if (track.path === 'rotation') {
      const r = bi * 4
      if (alpha === 0) {
        this._sampleRot[r] = values[off0]!
        this._sampleRot[r + 1] = values[off0 + 1]!
        this._sampleRot[r + 2] = values[off0 + 2]!
        this._sampleRot[r + 3] = values[off0 + 3]!
      } else {
        this._q4a[0] = values[off0]!
        this._q4a[1] = values[off0 + 1]!
        this._q4a[2] = values[off0 + 2]!
        this._q4a[3] = values[off0 + 3]!
        this._q4b[0] = values[off1]!
        this._q4b[1] = values[off1 + 1]!
        this._q4b[2] = values[off1 + 2]!
        this._q4b[3] = values[off1 + 3]!
        quatSlerp(this._q4a, this._q4a, this._q4b, alpha)
        this._sampleRot[r] = this._q4a[0]!
        this._sampleRot[r + 1] = this._q4a[1]!
        this._sampleRot[r + 2] = this._q4a[2]!
        this._sampleRot[r + 3] = this._q4a[3]!
      }
    } else {
      const s = bi * 3
      if (alpha === 0) {
        this._sampleScale[s] = values[off0]!
        this._sampleScale[s + 1] = values[off0 + 1]!
        this._sampleScale[s + 2] = values[off0 + 2]!
      } else {
        this._sampleScale[s] = values[off0]! + (values[off1]! - values[off0]!) * alpha
        this._sampleScale[s + 1] = values[off0 + 1]! + (values[off1 + 1]! - values[off0 + 1]!) * alpha
        this._sampleScale[s + 2] = values[off0 + 2]! + (values[off1 + 2]! - values[off0 + 2]!) * alpha
      }
    }
  }
}

export const createAnimationMixer = (skeleton: Skeleton): AnimationMixer => new AnimationMixer(skeleton)
