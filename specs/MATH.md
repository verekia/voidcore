# Voidcore — Math

## Coordinate System

**Z-up, right-handed:**

- **X** = right
- **Y** = forward
- **Z** = up

Matches Blender convention. All math functions, constants, and `lookAt` are implemented with Z-up baked in.

## Data Backing

All math types are `Float32Array` type aliases:

```typescript
type Vec2 = Float32Array // length 2
type Vec3 = Float32Array // length 3
type Vec4 = Float32Array // length 4
type Quat = Float32Array // length 4
type Mat4 = Float32Array // length 16
type AABB = Float32Array // length 6: [minX, minY, minZ, maxX, maxY, maxZ]
```

Type aliases over `Float32Array` rather than wrapper classes. This enables:

- **Direct GPU upload**: math values can be written straight to uniform buffers
- **Pooling and reuse**: no object overhead, just array slots
- **Sub-array views**: `Float32Array.subarray()` into contiguous buffers (zero-copy)
- **Interop**: no conversion needed between math types and GPU buffer data

```typescript
// A node's position is a subarray view into a larger buffer:
const positions = new Float32Array(1000 * 3)
node.position = positions.subarray(nodeIndex * 3, nodeIndex * 3 + 3) // Zero-copy view
node.position.set(1, 2, 3) // Writes directly into the contiguous buffer
```

## Matrix Convention

**Column-major 4×4 matrices** — matches WebGL/WebGPU native format:

```
Memory layout (indices):
[m00, m10, m20, m30,    // Column 0
 m01, m11, m21, m31,    // Column 1
 m02, m12, m22, m32,    // Column 2
 m03, m13, m23, m33]    // Column 3

Where m[row][col]
```

Column-major is the native format for both:

- WebGL: `gl.uniformMatrix4fv(loc, false, matrix)` (the `false` = no transpose)
- WebGPU/WGSL: `mat4x4<f32>` stored column-major

No transpose needed at upload time.

## Quaternion Convention

**[x, y, z, w] order** — matches glTF convention:

```typescript
const QUAT_IDENTITY: Quat = new Float32Array([0, 0, 0, 1])
```

## API Style

**Pure functional API with output parameter first:**

```typescript
// Output parameter first, inputs follow
const vec3Add = (out: Vec3, a: Vec3, b: Vec3): Vec3 => {
  out[0] = a[0] + b[0]
  out[1] = a[1] + b[1]
  out[2] = a[2] + b[2]
  return out
}

const mat4Multiply = (out: Mat4, a: Mat4, b: Mat4): Mat4 => {
  // 64 multiplications + 48 additions, inlined for performance
  // ... (no loop, fully unrolled)
  return out
}

const quatSlerp = (out: Quat, a: Quat, b: Quat, t: number): Quat => {
  // Spherical linear interpolation with shortest-path
  // ... handles antipodal quaternions (dot < 0 → negate one)
  return out
}
```

All functions:

- Return the output parameter for chaining: `vec3Normalize(v, vec3Add(v, a, b))`
- **Never allocate** — the caller provides the output buffer
- Allow `out` to alias an input (`vec3Add(a, a, b)` works correctly)

## Scratch Variables

### Module-Level Scratch

Pre-allocated at module scope, reused by functions within that module:

```typescript
// Private to the module — not exported
const _tempVec3 = vec3Create()
const _tempVec3b = vec3Create()
const _tempMat4 = mat4Create()
const _tempQuat = quatCreate()
```

Used by engine internals (frustum culling, BVH traversal, animation blending) to avoid allocation in hot paths.

### Frame-Scoped Scratch Pool

For user code in `useFrame` / `onFrame` callbacks:

```typescript
const scratchPool = {
  vec3s: new Float32Array(64 * 3), // 64 Vec3 slots
  mat4s: new Float32Array(16 * 16), // 16 Mat4 slots
  quats: new Float32Array(32 * 4), // 32 Quat slots
  vec3Index: 0,
  mat4Index: 0,
  quatIndex: 0,

  nextVec3(): Vec3 {
    const offset = this.vec3Index * 3
    this.vec3Index++
    return this.vec3s.subarray(offset, offset + 3)
  },

  nextMat4(): Mat4 {
    const offset = this.mat4Index * 16
    this.mat4Index++
    return this.mat4s.subarray(offset, offset + 16)
  },

  reset() {
    this.vec3Index = 0
    this.mat4Index = 0
    this.quatIndex = 0
  },
}
```

Reset at the end of each frame (step 18 in the frame lifecycle). Values are valid for the duration of one frame.

## Core Operations

### Vec3

```
vec3Create(): Vec3                                        // [0, 0, 0]
vec3Set(out, x, y, z): Vec3
vec3Copy(out, a): Vec3
vec3Add(out, a, b): Vec3
vec3Sub(out, a, b): Vec3
vec3Scale(out, a, scalar): Vec3
vec3Negate(out, a): Vec3
vec3Dot(a, b): number
vec3Cross(out, a, b): Vec3
vec3Length(a): number
vec3LengthSq(a): number
vec3Normalize(out, a): Vec3
vec3Lerp(out, a, b, t): Vec3
vec3Min(out, a, b): Vec3                                  // Component-wise min
vec3Max(out, a, b): Vec3                                  // Component-wise max
vec3TransformMat4(out, a, m): Vec3                        // Apply mat4 as point
vec3TransformQuat(out, a, q): Vec3                        // Rotate by quaternion
vec3Distance(a, b): number
vec3DistanceSq(a, b): number
```

### Vec4

```
vec4Create(): Vec4
vec4Set(out, x, y, z, w): Vec4
vec4TransformMat4(out, a, m): Vec4                        // Full 4D transform
```

### Mat4

```
mat4Create(): Mat4                                         // Identity
mat4Identity(out): Mat4
mat4Copy(out, a): Mat4
mat4Multiply(out, a, b): Mat4                             // Fully unrolled
mat4Invert(out, a): Mat4 | null                           // Returns null if singular
mat4Transpose(out, a): Mat4
mat4FromTranslation(out, v): Mat4
mat4FromRotation(out, q): Mat4
mat4FromScale(out, v): Mat4
mat4Compose(out, position, rotation, scale): Mat4         // TRS composition
mat4Decompose(m, outPos, outRot, outScale): void          // Extract TRS
mat4Perspective(out, fov, aspect, near, far, depth): Mat4 // depth: 'zero-to-one' | 'neg-one-to-one'
mat4Ortho(out, left, right, bottom, top, near, far): Mat4
mat4LookAt(out, eye, target, up): Mat4                    // Z-up aware
mat4GetTranslation(out, m): Vec3
mat4GetRotation(out, m): Quat
mat4GetScale(out, m): Vec3
```

`mat4Perspective` takes a `depth` parameter to handle WebGPU (`[0,1]`) vs WebGL2 (`[-1,1]`) depth range.

`mat4LookAt` is Z-up aware — the default up vector is `[0, 0, 1]`.

### Quat

```
quatCreate(): Quat                                         // [0, 0, 0, 1] identity
quatIdentity(out): Quat
quatCopy(out, a): Quat
quatMultiply(out, a, b): Quat
quatInvert(out, a): Quat
quatConjugate(out, a): Quat
quatNormalize(out, a): Quat
quatFromAxisAngle(out, axis, angle): Quat
quatSlerp(out, a, b, t): Quat                            // Shortest-path
quatLookRotation(out, forward, up): Quat                  // Z-up aware
```

### AABB

```
aabbCreate(): AABB                                         // [Inf, Inf, Inf, -Inf, -Inf, -Inf]
aabbFromPoints(out, positions, count): AABB               // Scan vertex positions
aabbExpandByPoint(out, point): AABB
aabbUnion(out, a, b): AABB                                // Enclosing AABB
aabbIntersects(a, b): boolean                              // Overlap test
aabbTransform(out, a, matrix): AABB                       // Transform corners, re-envelope
aabbCenter(out, a): Vec3
aabbSize(out, a): Vec3
aabbSurfaceArea(a): number                                 // For SAH BVH cost
```

### Frustum

```
frustumFromViewProjection(out, vpMatrix): Float32Array    // Gribb-Hartmann plane extraction
frustumContainsAABB(planes, aabb): boolean                 // P-vertex/N-vertex test
frustumContainsPoint(planes, point): boolean
```

6 planes, each stored as `[nx, ny, nz, d]` (16 bytes per plane, 96 bytes total). The P-vertex test checks the AABB vertex most aligned with the plane normal — if it's outside, the entire AABB is outside.

### Ray

```
rayCreate(): Ray                                           // origin + direction
raySet(ray, origin, direction): Ray
rayAt(out, ray, t): Vec3                                  // Point at distance t
rayIntersectsAABB(ray, min, max, maxDist): boolean        // Slab method
rayIntersectsTriangle(ray, v0, v1, v2): HitResult | null  // Möller-Trumbore
rayFromCamera(ray, ndcX, ndcY, camera): Ray               // Unproject screen point
```

`ray.invDirection` is pre-computed and cached for efficient slab-method AABB tests.

## Depth Range

The perspective projection matrix adjusts based on the active backend:

```typescript
// WebGPU: Z maps to [0, 1]
mat4Perspective(out, fov, aspect, near, far, 'zero-to-one')

// WebGL2: Z maps to [-1, 1]
mat4Perspective(out, fov, aspect, near, far, 'neg-one-to-one')
```

The backend provides its depth range convention, and the camera system passes it through. User code doesn't need to think about this.

## Constants

Frozen to prevent accidental mutation:

```typescript
const VEC3_ZERO: Vec3 = Object.freeze(new Float32Array([0, 0, 0])) as Vec3
const VEC3_ONE: Vec3 = Object.freeze(new Float32Array([1, 1, 1])) as Vec3
const VEC3_UP: Vec3 = Object.freeze(new Float32Array([0, 0, 1])) as Vec3 // Z-up
const VEC3_FORWARD: Vec3 = Object.freeze(new Float32Array([0, 1, 0])) as Vec3 // Y-forward
const VEC3_RIGHT: Vec3 = Object.freeze(new Float32Array([1, 0, 0])) as Vec3 // X-right
const QUAT_IDENTITY: Quat = Object.freeze(new Float32Array([0, 0, 0, 1])) as Quat
const MAT4_IDENTITY: Mat4 = Object.freeze(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])) as Mat4
```

These are safe to pass as default parameters — `Object.freeze` prevents any function from accidentally writing to them.

## Z-Up Awareness

`mat4LookAt`, `quatLookRotation`, and spherical coordinate conversions all use Z-up:

```typescript
// lookAt with Z-up: camera looks at target with Z as the "up" direction
const mat4LookAt = (out: Mat4, eye: Vec3, target: Vec3, up: Vec3 = VEC3_UP): Mat4 => {
  // Forward = normalize(target - eye)
  // Right = normalize(cross(forward, up))
  // CameraUp = cross(right, forward)
  // ... build view matrix
}
```

No coordinate system conversion happens in the math library. The Z-up convention is baked into all directional assumptions.

## No External Dependencies

Custom math library with zero dependencies. No gl-matrix, no wgpu-matrix, nothing external.

- Size: ~5-8KB minified
- Purpose-built for Z-up
- Avoids overhead of general-purpose libraries (axis-agnostic `lookAt`, configurable conventions)
- Fully tree-shakeable — unused functions are eliminated

## Testing

Unit tests for all math functions with known-good values. Special attention to:

- **Quaternion-matrix round-trip**: `compose(decompose(m)) ≈ m`
- **Z-up lookAt**: camera facing +Y with Z-up produces correct view matrix
- **Perspective depth mapping**: near plane maps to 0 (WebGPU) or -1 (WebGL2), far plane to 1
- **AABB transform**: transformed AABB fully encloses all transformed corners
- **Frustum culling**: objects at edges correctly classified as inside/outside
- **Slerp edge cases**: antipodal quaternions, identity interpolation, t=0/1
- **Matrix inversion**: `M × M⁻¹ ≈ I` within Float32 precision
