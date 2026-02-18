# Voidcore — Animation

## Skeleton Structure

Flat bone array with parent indices and inverse bind matrices:

```typescript
interface Skeleton {
  bones: Node[] // Bone nodes (regular scene graph nodes)
  boneInverseBindMatrices: Mat4[] // One per bone, loaded from glTF
  boneMatrices: Float32Array // Computed skinning matrices (contiguous, GPU-uploadable)
  boneCount: number

  getBone(name: string): Node | null // Find bone by name
}
```

- **Bones are regular scene nodes** in the scene graph hierarchy. They participate in dirty flag propagation and world matrix updates like any other node.
- **Inverse bind matrices** transform vertices from model space to bone space. Loaded once from glTF, stored permanently.
- **Bone matrices array** is a contiguous `Float32Array(boneCount * 16)` updated each frame by the animation mixer and uploaded directly to the GPU.

## GPU Skinning

Vertex shader skinning with 4 bones per vertex (the glTF standard maximum for basic skinning):

```glsl
// Vertex shader skinning
mat4 boneMatrix = a_weights.x * u_boneMatrices[a_joints.x]
               + a_weights.y * u_boneMatrices[a_joints.y]
               + a_weights.z * u_boneMatrices[a_joints.z]
               + a_weights.w * u_boneMatrices[a_joints.w];

vec4 skinnedPosition = boneMatrix * vec4(a_position, 1.0);
vec3 skinnedNormal = mat3(boneMatrix) * a_normal;  // No need for inverse transpose if uniform scale
```

### Bone Matrix Storage

Hard limit of **32 bones per skeleton**:

```
32 bones × 64 bytes (mat4) = 2KB — well within UBO size limits on all platforms
```

Bone matrices are packed into the per-object uniform buffer (bind group 2), after the world matrix. The shader indexes into the array by joint index. No texture fallback path needed — the UBO is always sufficient.

### Upload

The contiguous `Float32Array` of bone matrices uploads in a single buffer write — no per-bone copies needed:

```typescript
device.writeBuffer(objectBuffer, boneOffset, skeleton.boneMatrices)
```

## Animation Clips

Named clips with per-bone keyframe tracks, matching the glTF animation structure:

```typescript
interface AnimationClip {
  name: string // e.g., 'idle', 'walk', 'attack'
  duration: number // Clip length in seconds
  tracks: KeyframeTrack[] // One track per animated bone per property
}

interface KeyframeTrack {
  boneIndex: number // Which bone this track controls
  property: 'position' | 'rotation' | 'scale'
  times: Float32Array // Keyframe timestamps (sorted ascending)
  values: Float32Array // Keyframe values (3 floats for pos/scale, 4 for rotation)
  interpolation: 'linear' | 'step' | 'cubicspline'
}
```

A single animation clip can have multiple tracks — one per bone per animated property. A walk animation might have 60 bones × 1 rotation track each = 60 tracks.

## Keyframe Interpolation

### Keyframe Lookup

Binary search with cached last keyframe index for O(1) sequential playback:

```typescript
const findKeyframe = (times: Float32Array, time: number, lastIndex: number): number => {
  // O(1): check if still in the same segment (sequential playback, common case)
  if (lastIndex < times.length - 1 && time >= times[lastIndex] && time < times[lastIndex + 1]) {
    return lastIndex
  }

  // O(1): check next segment (advancing one frame)
  if (lastIndex + 1 < times.length - 1 && time >= times[lastIndex + 1] && time < times[lastIndex + 2]) {
    return lastIndex + 1
  }

  // O(log n): binary search fallback (seeking, looping)
  return binarySearch(times, time)
}
```

### Interpolation Modes

- **Linear**: `lerp` for position/scale, `slerp` for rotation quaternions
- **Step**: snap to the keyframe at or before the current time (no interpolation)
- **Cubic spline**: glTF cubic spline with in/out tangents (Hermite interpolation)

```typescript
// Linear interpolation between keyframes
const t = (time - times[i]) / (times[i + 1] - times[i])

// Position/scale: component-wise lerp
vec3Lerp(out, values[i], values[i + 1], t)

// Rotation: spherical lerp (shortest path)
quatSlerp(out, values[i], values[i + 1], t)
```

## Animation Mixer

Action-based mixer API for managing playback and blending:

```typescript
const mixer = createAnimationMixer(skeleton)

// Create actions from clips
const idle = mixer.clipAction(idleClip)
const walk = mixer.clipAction(walkClip)
const attack = mixer.clipAction(attackClip)

// Play
idle.play()

// Crossfade to walk over 300ms
idle.crossFadeTo(walk, 0.3)

// Action properties
walk.loop = 'repeat' // 'repeat' | 'once' | 'pingpong'
walk.timeScale = 1.5 // Playback speed (negative = reverse)
walk.weight = 1.0 // Blend weight
walk.time = 0 // Current time (can be set for seeking)
walk.paused = false
```

### Action Interface

```typescript
interface AnimationAction {
  clip: AnimationClip
  loop: 'repeat' | 'once' | 'pingpong'
  timeScale: number // Playback speed multiplier
  weight: number // Blend weight (0-1)
  time: number // Current playback time
  paused: boolean
  playing: boolean

  play(): AnimationAction
  stop(): AnimationAction
  fadeIn(duration: number): AnimationAction
  fadeOut(duration: number): AnimationAction
  crossFadeTo(target: AnimationAction, duration: number): AnimationAction
}
```

### Blending

Multiple actions can play simultaneously with normalized weights:

```typescript
// If idle.weight = 0.3 and walk.weight = 0.7:
// normalizedIdle = 0.3 / (0.3 + 0.7) = 0.3
// normalizedWalk = 0.7 / (0.3 + 0.7) = 0.7
// Final bone transform = slerp(idlePose, walkPose, normalizedWalk)
```

For rotation blending, `slerp` is used. For position and scale, linear interpolation.

Weight normalization ensures poses don't overshoot regardless of how many actions are active. If only one action plays with weight 1.0, it's used directly (no normalization overhead).

### Crossfade

```typescript
idle.crossFadeTo(walk, 0.3)
```

Internally:

1. `walk.play()` with `walk.weight = 0`
2. Over 0.3 seconds: `idle.weight` fades from current → 0, `walk.weight` fades from 0 → 1
3. When complete: `idle.stop()` (removed from active list)

The blend uses a smooth easing curve (quadratic ease-in-out) for natural transitions.

## Mixer Update

Called once per frame:

```typescript
mixer.update(deltaTime)
```

Update sequence:

1. **Advance action times**: `action.time += deltaTime * action.timeScale`
2. **Handle looping**: wrap time for `repeat`, clamp for `once`, reverse for `pingpong`
3. **Update crossfade weights**: interpolate weights for active crossfades
4. **Remove finished actions**: actions with `loop: 'once'` that have completed, faded-out actions
5. **Sample keyframes**: for all active actions, evaluate every track at current time
6. **Blend bone transforms**: combine sampled poses using normalized weights
7. **Compute bone matrices**: `boneMatrix[i] = bone[i].worldMatrix × inverseBindMatrix[i]`
8. **Write to contiguous array**: pack all bone matrices into `skeleton.boneMatrices`

### Zero Allocations

The mixer pre-allocates all scratch quaternions, vectors, and intermediate pose arrays at creation time. `mixer.update()` performs zero heap allocations.

```typescript
// Pre-allocated at mixer creation:
const _posePositions = new Float32Array(boneCount * 3)
const _poseRotations = new Float32Array(boneCount * 4)
const _poseScales = new Float32Array(boneCount * 3)
const _tempQuat = quatCreate()
const _tempVec3 = vec3Create()
```

## Bone Attachment

Bones are regular scene nodes. Attaching objects is standard parenting:

```typescript
const handBone = skeleton.getBone('hand_R')
handBone.add(swordMesh)
```

The attached mesh inherits the bone's animated world transform through normal scene graph propagation. When the animation system updates bone world matrices, the dirty flag propagation marks the sword mesh dirty, and its world matrix is recomputed from the bone's transform.

No special attachment API, no extra frame-delay — it uses the same update path as every other parent-child relationship.

## Performance Budget

```
Per animated character (60-bone skeleton):
  Keyframe sampling:     ~0.02ms
  Blending:              ~0.01ms
  Bone matrix compute:   ~0.01ms
  Total:                 ~0.04ms per character

10 animated characters: ~0.4ms total
```

This leaves ample room in the 16.6ms frame budget. The main costs are the keyframe sampling (binary search / cache hit) and the bone matrix multiplication chain.
