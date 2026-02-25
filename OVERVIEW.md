# VoidCore Engine Overview

This document explains how VoidCore works — from the math foundations to the rendering pipeline to the React bindings. It's written for developers with intermediate graphics knowledge who want to understand both the general concepts behind a real-time 3D engine and the specific decisions that shape VoidCore.

Reading time: ~15 minutes.

---

## Table of Contents

1. [What Is VoidCore?](#what-is-voidcore)
2. [Architecture at a Glance](#architecture-at-a-glance)
3. [Math Foundations](#math-foundations)
4. [The Scene Graph](#the-scene-graph)
5. [Geometry and Vertex Data](#geometry-and-vertex-data)
6. [Materials and Shading](#materials-and-shading)
7. [Textures and Compression](#textures-and-compression)
8. [The Rendering Pipeline](#the-rendering-pipeline)
9. [Shaders](#shaders)
10. [Shadows](#shadows)
11. [Transparency and Draw Order](#transparency-and-draw-order)
12. [Post-Processing: Bloom](#post-processing-bloom)
13. [Skeletal Animation](#skeletal-animation)
14. [Asset Loading](#asset-loading)
15. [Raycasting and BVH](#raycasting-and-bvh)
16. [Camera and Controls](#camera-and-controls)
17. [The Scheduler](#the-scheduler)
18. [Sprites and Billboards](#sprites-and-billboards)
19. [HTML Overlays](#html-overlays)
20. [React Bindings](#react-bindings)
21. [Performance Philosophy](#performance-philosophy)

---

## What Is VoidCore?

VoidCore is a real-time 3D graphics engine written in TypeScript. It supports two GPU backends — WebGPU (modern, higher performance) and WebGL2 (wider browser compatibility) — and includes an optional React declarative layer for building 3D scenes with JSX.

The engine covers the core responsibilities of any game engine's rendering layer: managing a tree of 3D objects, computing their transforms, uploading geometry and material data to the GPU, issuing draw calls, and handling features like shadows, transparency, skeletal animation, and post-processing. It also provides asset loaders for standard formats (glTF/GLB, KTX2), a raycasting system for mouse picking, camera controls, and a scheduling system for the main loop.

---

## Architecture at a Glance

A VoidCore application has this structure:

```
Engine
├── Scheduler          – Single rAF loop, drives the entire application
├── Renderer           – WebGPU or WebGL2 backend (auto-detected)
│   ├── Shadow pass    – Depth-only render from the light's perspective
│   ├── Main pass      – MSAA MRT render (color + emissive targets)
│   ├── Bloom pass     – Downsample → upsample post-processing chain
│   └── Blit pass      – Final composite with gamma correction
├── Scene              – Root of the scene graph (tree of Nodes)
│   ├── Meshes         – Geometry + Material = visible objects
│   ├── Sprites        – Camera-facing billboard planes
│   ├── Lights         – Directional and ambient light sources
│   ├── Groups         – Empty containers for hierarchy organization
│   └── Camera         – Defines the viewpoint and projection
├── OrbitControls      – Mouse/touch camera interaction
├── Raycaster          – Ray-mesh intersection for picking
└── OverlayManager     – HTML elements tracked to 3D positions
```

The `Engine` is the main entry point. It owns the scheduler (the rAF loop) and the renderer (the GPU backend). Each frame, the scheduler ticks all registered callbacks in priority order, one of which typically calls `engine.render(scene, camera)` to draw the frame.

`Engine.create()` is an async factory because GPU initialization is asynchronous — requesting a WebGPU adapter and device, or creating a WebGL2 context, are operations that may need to query hardware capabilities.

---

## Math Foundations

Every 3D engine needs linear algebra: vectors, matrices, and quaternions. VoidCore's math library (`math/index.ts`) provides these primitives.

### Types

All math types are backed by `Float32Array`. This matters for two reasons: GPUs expect float32 data, so there's zero conversion cost when uploading to the GPU; and typed arrays have predictable memory layout, which helps with performance.

- **Vec3** (Float32Array, length 3) — A 3D vector. Used for positions, directions, scales, colors.
- **Vec4** (Float32Array, length 4) — A 4D vector. Used for homogeneous coordinates and shader uniforms.
- **Quat** (Float32Array, length 4, `[x, y, z, w]`) — A quaternion representing a rotation. Quaternions avoid gimbal lock (a problem with Euler angles where you lose a degree of freedom) and interpolate smoothly via spherical linear interpolation (slerp).
- **Mat4** (Float32Array, length 16, column-major) — A 4×4 transformation matrix. This is the workhorse of 3D graphics: a single Mat4 can encode position, rotation, and scale. The GPU multiplies every vertex by this matrix to place it in the world.
- **AABB** (Float32Array, length 6, `[minX, minY, minZ, maxX, maxY, maxZ]`) — An axis-aligned bounding box. Used for fast spatial queries like frustum culling and raycasting.
- **Frustum** — Six clip planes extracted from the view-projection matrix. Any object whose AABB lies entirely outside one of these planes is off-screen and can be skipped.

### The "Write Into Output" Pattern

Instead of `const result = vec3Add(a, b)` which allocates a new array every call, VoidCore uses `vec3Add(out, a, b)` where `out` is a pre-allocated buffer. This is critical in a render loop that runs 60+ times per second — allocating thousands of short-lived arrays per frame would create garbage collection pressure that causes frame hitches.

### Coordinate System

VoidCore uses a **Z-up, right-handed** coordinate system:

- **+X** is right
- **+Y** is forward
- **+Z** is up

This matches many CAD/engineering conventions. Some engines use Y-up (notably Three.js and Unity); the choice is arbitrary but must be consistent throughout the engine — from primitive generation to camera controls to shader math.

### Column-Major Matrices

Matrices are stored in column-major order, meaning the first four elements are the first _column_, not the first row. This matches what WebGPU and WebGL expect, so matrices can be uploaded to the GPU without transposition. Key matrix operations include:

- **mat4Compose** — Builds a matrix from position, rotation (quaternion), and scale.
- **mat4Perspective** — Builds a perspective projection matrix from field-of-view, aspect ratio, and near/far planes.
- **mat4LookAt** — Builds a view matrix that positions the "camera" at an eye point, looking at a target.
- **mat4Multiply** — Chains transforms: `parent × child` applies the parent's transform to the child.
- **mat4Invert** — Computes the inverse (used to go from world space back to local space).

---

## The Scene Graph

The scene graph is a tree structure where every 3D object is a **Node** (`scene/node.ts`). Nodes form parent-child relationships: moving a parent automatically moves all its children.

### Node Types

- **Group** — An empty container. Groups don't render anything; they exist to organize children so you can transform them as a unit (e.g., a "car" group containing body, wheels, and window meshes).
- **Mesh** — A visible 3D object, composed of a Geometry (shape) and a Material (appearance).
- **Sprite** — A billboard plane that always faces the camera.
- **PerspectiveCamera** — Defines the viewpoint and projection.
- **DirectionalLight** — A distant light source with parallel rays (like the sun).
- **AmbientLight** — Constant low-level illumination affecting all objects equally.

### Transforms

Each node stores a local transform as three components:

- **position** (Vec3) — Where the node is, relative to its parent.
- **rotation** (Quat) — How the node is oriented, relative to its parent.
- **scale** (Vec3) — How the node is scaled, relative to its parent.

From these, two matrices are computed:

- **Local matrix** = `mat4Compose(position, rotation, scale)` — The transform relative to the parent.
- **World matrix** = `parent.worldMatrix × localMatrix` — The transform relative to the world origin.

The world matrix is what the GPU actually uses. When you set a node's position, VoidCore doesn't immediately recompute matrices. Instead, it sets a **dirty flag** (`_dirtyLocal`) and propagates a **subtree dirty flag** (`_subtreeDirty`) up to all ancestors. Later, when the scene is about to be rendered, `scene.updateGraph()` walks the tree top-down and only recomputes matrices for dirty nodes and their descendants.

This is a common optimization: in a scene with 500 objects but only 10 animated characters, only the animated characters and their bone hierarchies need matrix updates each frame. The `_subtreeDirty` flag lets the traversal skip entire static subtrees without even visiting them.

The traversal uses pre-allocated stacks (kept on the Scene object) for an iterative depth-first walk, avoiding recursion overhead and per-frame allocations.

### Scene

The **Scene** (`scene/scene.ts`) is the root Node. It additionally maintains a name registry (`Map<string, Node>`) for O(1) lookup of any node by name — useful for finding specific bones or meshes after loading a model.

---

## Geometry and Vertex Data

A **Geometry** (`geometry/geometry.ts`) holds the raw arrays that describe the shape of a 3D object. The GPU needs this data uploaded as buffers to draw triangles.

### Vertex Attributes

Each vertex has:

- **Position** (Float32, 3 components) — Where the vertex is in 3D space.
- **Normal** (Float32, 3 components) — A unit-length vector perpendicular to the surface. Used for lighting calculations (how much light hits this point depends on the angle between the normal and the light direction).
- **UV** (Float32, 2 components, optional) — Texture coordinates that map a 2D image onto the 3D surface. (0,0) is one corner of the texture, (1,1) is the opposite corner.
- **Index buffer** (Uint16 or Uint32) — Defines which vertices form each triangle. Three consecutive indices = one triangle. Index buffers save memory by letting multiple triangles share vertices.

Optional attributes for advanced features:

- **Colors / Emissive Colors** — Per-vertex colors baked from a palette (see Materials).
- **Joints / Weights** — Bone indices and blend weights for skeletal animation.
- **Material Indices** — Per-vertex palette entry index for the `bakePalette()` system.

### Vertex Packing

When uploading to the GPU, not every attribute needs full 32-bit float precision. VoidCore packs attributes into smaller formats to reduce memory bandwidth (`renderer/pack.ts`):

| Attribute     | CPU Format           | GPU Format          | Savings                                                                              |
| ------------- | -------------------- | ------------------- | ------------------------------------------------------------------------------------ |
| Normals       | Float32×3 (12 bytes) | Snorm8×4 (4 bytes)  | 3× smaller. Unit-length normals only need [-1,1] range; 8-bit gives 1/127 precision. |
| UVs           | Float32×2 (8 bytes)  | Float16×2 (4 bytes) | 2× smaller. Texture coordinates rarely need more than ~3 decimal digits.             |
| Bone weights  | Float32×4 (16 bytes) | Unorm8×4 (4 bytes)  | 4× smaller. Weights are [0,1] and sum to 1.0; 1/255 precision is enough.             |
| Vertex colors | Float32×4 (16 bytes) | Unorm8×4 (4 bytes)  | 4× smaller. Palette-baked colors are [0,1] range.                                    |
| Emissive      | Float32×4 (16 bytes) | Float16×4 (8 bytes) | 2× smaller. Emissive can exceed 1.0 (HDR), so half-float is needed.                  |

The bone weight packing includes a normalization step that ensures the four quantized values sum to exactly 255 — without this, rounding errors cause the skin matrix to drift, and vertices shift noticeably at large distances from the origin.

### Procedural Primitives

VoidCore generates common shapes procedurally (`geometry/primitives.ts`): Plane, Box, Sphere, Cone, Cylinder, Capsule, and Circle. Each primitive follows the same pattern: loop over a parameter grid (angles, segments), compute each vertex's position and normal using trigonometry, generate UV coordinates, then connect vertices into triangles via indices. All primitives are Z-up and centered at the origin.

### Bounding Boxes

Every Geometry computes an axis-aligned bounding box (AABB) from its vertex positions. This AABB is used for:

- **Frustum culling** — If the AABB (transformed to world space) is entirely outside the camera's view frustum, skip drawing the mesh.
- **Raycasting** — First test the ray against the AABB (fast); only test individual triangles if the ray hits the AABB.

### The Palette System (bakePalette)

For stylized games where models use flat colors rather than texture maps, VoidCore provides a palette system. Instead of assigning one color per material, you define a palette: an array of color entries (with optional emissive glow) indexed per-vertex via `materialIndices`.

`bakePalette(geometry, palette)` resolves these indices into per-vertex `colors` and `emissiveColors` arrays baked directly into the geometry. Results are cached by geometry+palette reference (using a WeakMap) so repeated calls with the same inputs return the same Geometry.

Palette entries can also include per-material **tiled AO textures** and **tiled normal maps** — textures that tile across surfaces in world-space coordinates to add fine detail like grout lines or surface roughness. The baking process packs texture layer indices and intensities into vertex attributes, and the renderer uploads unique textures as 2D array textures for efficient GPU sampling.

---

## Materials and Shading

A **Material** (`materials/material.ts`) defines how a mesh surface looks when rendered. VoidCore supports two shading models:

- **BasicMaterial** — Unlit flat color. The surface ignores lights entirely. Useful for debug visualization, UI elements, or artistic flat-shaded looks.
- **LambertMaterial** — Diffuse shading that reacts to directional and ambient light. The amount of light hitting a surface point is computed as `dot(surfaceNormal, lightDirection)` — this is Lambert's cosine law, the simplest physically-motivated lighting model.

### Material Properties

- **color** — Base RGB color (`[r, g, b]`, 0–1 range).
- **opacity** / **transparent** — Alpha transparency. When `transparent: true`, the mesh is drawn back-to-front with blending enabled and depth writes off.
- **side** — Face culling control: `'front'` (default, back faces culled), `'back'` (front faces culled), or `'double'` (no culling, both faces visible).
- **colorMap** — A texture that multiplies the base color per-pixel (diffuse/albedo map).
- **aoMap** / **aoIntensity** — An ambient occlusion texture that darkens crevices where ambient light can't reach.
- **emissive** / **emissiveIntensity** — Self-illumination color and strength. High-intensity emissive colors desaturate toward white, simulating the overexposed look of real glowing surfaces.

### Custom Shaders

For effects that go beyond the built-in shading models, materials support custom shader injection. You provide WGSL (for WebGPU) and/or GLSL (for WebGL2) code snippets that are inserted at well-defined hook points in the vertex and fragment stages:

- **Vertex hook** — Runs after world position, normal, and UVs are computed, but before clip-space projection. You can modify position (for vertex displacement effects), normal, or UVs.
- **Fragment hook** — Runs after the final color and alpha are computed, but before output. You can modify color (for tinting, desaturation, etc.) or alpha.

Custom uniforms (`uniforms: Record<string, number>`) let you pass arbitrary float values from JavaScript to the shader each frame — useful for time-based animations, material parameters, etc. These are backed by a GPU uniform buffer with per-material caching.

---

## Textures and Compression

A **Texture** (`materials/texture.ts`) stores pixel data for GPU texture maps. Textures can be uncompressed (RGBA8, 32 bits per pixel) or GPU-compressed:

- **ASTC 4×4** (8 bpp) — Best for modern mobile GPUs.
- **BC7** (8 bpp) — Best for modern desktop GPUs.
- **BC3** (8 bpp) — Wider desktop support than BC7.
- **ETC2 RGBA** (8 bpp) — Native on all mobile WebGL2/GLES3 devices.

GPU-compressed textures are uploaded directly to the GPU in their compressed form — the GPU decompresses them on-the-fly during sampling. This halves texture memory and bandwidth compared to RGBA8. Both renderers detect supported compressed formats at initialization and expose them via `engine.compressedTextureFormats`.

---

## The Rendering Pipeline

Each frame, the renderer executes a multi-pass pipeline. Both the WebGPU (`renderer/webgpu.ts`) and WebGL2 (`renderer/webgl.ts`) backends follow the same logical structure:

### 1. Setup

- Resize the canvas to match the display resolution, accounting for device pixel ratio (DPR). The `maxDpr` setting caps the resolution scaling to save GPU on high-density displays (default 1.25 on mobile, 1.5 on desktop).
- Update the camera's projection matrix if the viewport changed.

### 2. Scene Graph Update

- `scene.updateGraph()` recomputes world matrices for any nodes whose transforms have changed since last frame.

### 3. Mesh Collection and Culling

A single pass over the scene graph (`shared.ts: collectMeshes`) performs:

- **Distance culling** — Meshes with a `maxDistance` are hidden (including their shadows) when the camera is farther than that distance. Uses squared-distance comparison (no square root) for zero allocation.
- **Frustum culling** — Each mesh's world-space AABB is tested against the camera's view frustum (six clip planes). Meshes entirely outside the frustum are skipped.
- **Shadow caster collection** — Meshes outside the camera frustum but inside the light's shadow frustum are still collected for shadow rendering (so they can cast shadows into the visible area).

### 4. Sorting

Meshes are sorted to minimize GPU state changes and ensure correct draw order (`renderer/sort.ts`).

The sort uses a **32-bit composite key** per mesh:

- **Bit 30: Layer** — 0 = opaque, 1 = transparent. This ensures all opaque meshes draw before any transparent mesh.
- **Opaque keys** prioritize state change reduction: pipeline ID (bits 29–22) → material ID (bits 21–10) → depth (bits 9–0, nearest first for early-Z).
- **Transparent keys** prioritize correct depth order: depth (bits 29–20, farthest first) → pipeline → material.

The sort algorithm is a **4-pass LSD radix sort** with an 8-bit radix. Radix sort is O(n) and stable, making it ideal for the thousands of meshes a scene might contain. Sort state (buffers, histograms) is pre-allocated to avoid per-frame allocations, and passes ping-pong between two index arrays to eliminate copy-back overhead.

### 5. Uniform Upload

Per-frame data (view-projection matrix, light direction, light color, shadow matrix, ambient color) is uploaded to a GPU uniform buffer once.

Per-object data (world matrix, base color, outline parameters) is batched into dynamic uniform buffers — typically 1–2 bulk uploads instead of N individual uploads.

### 6. Shadow Pass

A depth-only render from the light's perspective into a shadow map texture. Only shadow-casting meshes are drawn. The result is a depth texture that encodes "how far is the nearest surface from the light?" at each texel.

### 7. Main Render Pass (MSAA + MRT)

The main pass draws all visible meshes into two render targets simultaneously (Multiple Render Targets):

- **Color target** — The regular scene color.
- **Emissive target** — Only emissive/glowing contributions, used by the bloom pass.

Multi-sample anti-aliasing (MSAA, 4× samples) smooths jagged triangle edges. After the pass, MSAA samples are resolved (averaged) to produce clean single-sample images.

### 8. Bloom Post-Processing

The emissive target feeds a bloom chain:

1. **Downsample** — A 13-tap filter progressively shrinks the emissive image through several mip levels. The first level uses Karis averaging to suppress firefly artifacts (tiny extremely bright pixels).
2. **Upsample** — A 9-tap tent filter blurs back up, additively blending each level.

The result is a soft glow around bright areas.

### 9. Final Blit

A fullscreen pass composites the scene color with the bloom contribution and applies gamma correction (linear → sRGB) for correct display on consumer monitors.

---

## Shaders

Shaders are small programs that run on the GPU. VoidCore maintains parallel shader sets for both backends:

- **WGSL** (`renderer/webgpu-shaders.ts`) — WebGPU Shading Language.
- **GLSL** (`renderer/webgl-shaders.ts`) — OpenGL Shading Language (version 300 es).

Both sets are functionally identical. The shaders are authored as template strings in TypeScript, making them easy to compose with custom shader injection.

### Shader Variants

The engine uses several shader variants rather than a single uber-shader:

- **Lambert** — Diffuse lighting with ambient + directional light, shadow sampling (PCF 3×3).
- **Lambert VC** (Vertex Color) — Same as Lambert but reads per-vertex color and emissive attributes. Also samples AO maps and tiled AO/normal textures.
- **Lambert Textured** — Lambert with color map and AO map texture sampling.
- **Lambert Skinned** — Lambert with bone matrix vertex deformation.
- **Lambert Skinned VC** — Skinned + vertex colors.
- **Basic** — Flat unlit color.
- **Basic Skinned** — Unlit with bone deformation.
- **Shadow Depth** — Depth-only for the shadow pass (static meshes).
- **Shadow Depth Skinned** — Depth-only with bone deformation.
- **Bloom Downsample / Upsample** — Post-processing.
- **Blit** — Final composite with gamma correction.

### Data Flow

CPU → GPU data flows through **uniform buffers** (UBOs in WebGL2, bind groups in WebGPU):

- **Frame uniforms (binding 0)** — View-projection matrix, light direction and color, shadow matrix, ambient color. Uploaded once per frame.
- **Object uniforms (binding 1)** — World matrix, normal matrix, base color, outline parameters. Batched per object.
- **Custom uniforms (binding 2)** — User-provided float values for custom shader effects.

Vertex data flows through **vertex buffers** — the packed geometry attributes (positions, normals, UVs, etc.) uploaded once and reused until the geometry changes.

### Outlines

Mesh outlines use the **inverted hull technique**: the geometry is doubled (original vertices + copies with smooth normals inflated outward by the outline thickness). Both halves are drawn in a single draw call with face culling disabled. The fragment shader uses the `front_facing` built-in to discard front-facing fragments from the outline copy — only back-facing fragments (the visible silhouette edge) survive. This halves draw call count compared to a two-pass approach.

Smooth normals are computed by averaging normals across all faces sharing the same vertex position, ensuring the outline inflates uniformly even at hard edges.

---

## Shadows

VoidCore uses **shadow mapping** — a technique where the scene is rendered from the light's perspective to build a depth map, then during the main render, each fragment checks the depth map to determine if it's in shadow.

### Shadow Map Generation

A **DirectionalLight** simulates an infinitely distant light source (like the sun) with parallel rays. Shadow configuration lives on the light itself:

- **shadowMapSize** — The orthographic box size (how large an area the shadow covers).
- **shadowNear / shadowFar** — Near and far planes for the shadow volume.
- **shadowBias / shadowSlopeBias** — Depth offsets to prevent shadow acne (a moiré artifact caused by floating-point precision when a surface tests its own depth).

The shadow volume is a fixed orthographic box centered at the world origin, oriented along the light direction. The renderer computes a light-space view-projection matrix from these parameters and renders all shadow-casting meshes into a depth texture.

### PCF Filtering

Raw shadow maps produce hard, aliased shadow edges because each fragment either is or isn't in shadow. VoidCore uses **PCF (Percentage Closer Filtering)** with a 3×3 kernel: instead of a single depth comparison, nine samples are taken around the fragment's shadow-map position and averaged. This produces soft shadow edges.

On WebGPU, this uses hardware comparison sampling (`textureSampleCompareLevel`). On WebGL2, the equivalent is a shadow sampler with `COMPARE_REF_TO_TEXTURE`.

### Shadow Baking

For static scenes where neither the light nor shadow-casting objects move, the shadow map can be **baked** (frozen). Setting `engine.shadowsBaked = true` (or mounting `<BakeShadows />` in React) tells the renderer to stop re-rendering the shadow map each frame, saving the cost of the entire shadow pass.

---

## Transparency and Draw Order

Transparent objects require special handling because alpha blending is order-dependent — you must draw back-to-front for correct results.

VoidCore's sort key places all transparent meshes (bit 30 = 1) after all opaque meshes (bit 30 = 0). Within the transparent batch, meshes are sorted farthest-first. After sorting, a binary search (`findTransparentStart`) finds the boundary index between opaque and transparent draws.

Opaque meshes are drawn with depth testing and depth writing enabled. Transparent meshes are drawn with depth testing enabled but **depth writing disabled** — otherwise a transparent surface would occlude objects behind it.

WebGPU uses **premultiplied alpha blending** (blend factors: src=one, dst=one-minus-src-alpha) rather than the more common straight alpha (src=src-alpha). This avoids a specific Vulkan driver bug on some Android devices where the `src-alpha` blend factor combined with comparison texture sampling triggers `VK_ERROR_UNKNOWN`.

---

## Post-Processing: Bloom

Bloom simulates the way real cameras and human eyes perceive very bright light sources — bright areas bleed outward into a soft glow.

VoidCore renders emissive contributions into a separate render target (the "emissive" MRT attachment). This target feeds the bloom chain:

1. **Downsample** — The emissive image is progressively downsampled through several mip levels using a 13-tap filter (box filter with bilinear taps). The first downsample level applies **Karis averaging** — a weighted average that suppresses firefly artifacts (single extremely bright pixels that would otherwise dominate the bloom).

2. **Upsample** — Starting from the smallest mip, a 9-tap tent filter (3×3 bilinear taps) blurs each level and additively blends it into the next larger level. This builds up a multi-scale glow.

3. **Composite** — The final blit pass adds the bloom result to the scene color. Bloom intensity is configurable.

---

## Skeletal Animation

Skeletal animation deforms a mesh's vertices according to a hierarchy of "bones" that move over time — like an invisible puppet skeleton inside the mesh.

### Components

- **Skeleton** (`animation/skeleton.ts`) — A set of bone nodes (which are regular scene graph Nodes) plus **inverse bind matrices**. The inverse bind matrix for each bone defines how the mesh was originally posed in the T-pose: `boneMatrix = bone.worldMatrix × inverseBindMatrix`. These matrices are written directly into a shared `Float32Array` buffer using `mat4MultiplyInto` (avoiding a temporary matrix allocation per bone), then uploaded to the GPU so the vertex shader can deform vertices.

- **AnimationClip** (`animation/clip.ts`) — A named collection of **KeyframeTracks** with a total duration. Each track stores arrays of timestamps and corresponding values (positions, rotations, or scales) for a single bone. Think of it as a spreadsheet: rows are time points, columns are bone properties.

- **AnimationMixer** (`animation/mixer.ts`) — Plays and blends multiple animations simultaneously. Each frame it: (1) advances time on each playing action (handling looping, ping-pong), (2) updates fade-in/fade-out weights for smooth transitions, (3) samples each clip at the current time to interpolate between keyframes, (4) blends all active actions using weighted interpolation, (5) writes the final pose to the skeleton's bones. Keyframe sampling uses binary search for efficient time-to-keyframe lookup, and rotation tracks use **slerp** (spherical linear interpolation) for smooth rotation blending.

- **AnimationAction** — Controls playback of a single clip: play, stop, pause, fade in/out, cross-fade to another action. Cross-fading smoothly transitions between animations (e.g., walk → run) by fading one out while fading the other in.

### Vertex Skinning

In the vertex shader, each vertex has up to 4 bone influences (stored as `joints` and `weights` attributes). The shader computes a weighted blend of the bone matrices:

```
skinMatrix = weight[0] * boneMatrix[joint[0]]
           + weight[1] * boneMatrix[joint[1]]
           + weight[2] * boneMatrix[joint[2]]
           + weight[3] * boneMatrix[joint[3]]
```

Then: `skinnedPosition = skinMatrix × localPosition`. The same transform is applied to normals (for correct lighting on deformed surfaces).

VoidCore supports up to 32 bones per skeleton, which is sufficient for most character models.

### Cloning

When instancing multiple copies of the same animated character, each instance needs its own bone hierarchy and skeleton to animate independently. `cloneScene()` (`scene/clone.ts`) deep-copies the node tree — meshes share geometry and material references (saving memory), but bones and skeletons are fully cloned with proper remapping.

---

## Asset Loading

### glTF / GLB

glTF (GL Transmission Format) is the standard format for 3D models on the web — like JPEG for images. VoidCore loads the binary variant (.glb), which packages everything into a single file.

The loader (`loaders/gltf.ts`) works in several passes:

1. **Parse the GLB binary** — Extract the JSON chunk (scene description) and BIN chunk (raw vertex/animation data).
2. **Create nodes** — For each glTF node, create a Mesh, Group, or bone with its transform. For each mesh primitive, read vertex attributes from binary accessors, optionally decoding Draco-compressed geometry.
3. **Build hierarchy** — Connect parent-child relationships.
4. **Resolve skins** — Create Skeletons from joint lists and inverse bind matrices.
5. **Parse animations** — Read keyframe tracks (time → value arrays) for bone transforms.
6. **Assemble** — Package everything under a root Group with the scene graph, meshes, skeletons, and animation clips.

glTF materials use PBR (Physically Based Rendering) definitions, which VoidCore simplifies to its Lambert/Basic shading models.

### KTX2 / Basis Universal

KTX2 is a GPU texture container format that stores Basis Universal compressed data (`loaders/ktx2.ts`). The key insight is that GPU-compressed formats are device-specific (ASTC for mobile, BC7 for desktop), but you don't want to ship separate files for every device.

Basis Universal solves this by using an intermediate compressed format (ETC1S or UASTC) that can be **transcoded** — converted at load time — into any GPU-native format. The transcoder is a WebAssembly module loaded once and cached.

The format priority is: ASTC 4×4 > BC7 > ETC2 RGBA > BC3 > RGBA8 (uncompressed fallback). The loader picks the best format supported by the device, transcodes, and returns a `Texture` ready for GPU upload.

---

## Raycasting and BVH

**Raycasting** (`raycasting/index.ts`) shoots an invisible ray into the 3D scene and finds which triangles it intersects. Common uses include collision detection, line-of-sight checks, ground placement (projecting objects onto terrain), pseudo-physics (e.g. gravity via downward rays), and mouse/pointer picking.

### The Problem

Testing a ray against every triangle of every mesh is O(n) per mesh — too slow when meshes have thousands of triangles.

### Bounding Volume Hierarchy (BVH)

A BVH is a tree of nested bounding boxes. The root box contains the entire mesh. It's split into two child boxes, each containing roughly half the triangles, and so on recursively. To test a ray, you first test the root box (fast). If it misses, you're done. If it hits, test both children. Continue until you reach leaf nodes containing just a few triangles, then test those triangles directly. This turns O(n) into roughly O(log n).

VoidCore's BVH uses **binned SAH** (Surface Area Heuristic) for construction: at each split, it evaluates multiple candidate split positions along each axis, picks the split that minimizes the expected ray-test cost (based on the surface areas of the resulting child boxes), and recurses. The tree is stored in a **flat array** for cache-friendly traversal.

### Algorithms

- **Ray-AABB intersection** — The slab method: compute the ray's entry and exit distances along each axis. If the intervals overlap, the ray hits the box.
- **Ray-Triangle intersection** — Möller-Trumbore algorithm: a fast algebraic method using cross products to simultaneously test intersection and compute barycentric coordinates (used for UV/normal interpolation at the hit point).

### Zero-Allocation Raycasting

For per-frame raycasts (e.g., collision tests, ground snapping, continuous pointer tracking), you can pre-allocate hit objects with `createRaycastHit()` and pass them as a target array. The raycaster writes into these pre-allocated objects and returns the hit count instead of allocating new arrays — eliminating GC pressure in the hot path.

---

## Camera and Controls

### PerspectiveCamera

A **PerspectiveCamera** (`scene/camera.ts`) uses perspective projection: objects farther away appear smaller, matching human vision. It's defined by:

- **fov** — Field of view in degrees (how "wide" the lens is, default 60°).
- **near / far** — Clipping planes (objects outside this range are invisible).
- **aspect** — Viewport width/height ratio (set automatically by the renderer).

The camera produces two matrices:

- **Projection matrix** — Converts 3D eye-space coordinates to clip space (applies the perspective effect).
- **View matrix** — Converts world-space coordinates to eye-space (where is the camera looking from and at what?).

The GPU multiplies these together into a **view-projection matrix** that transforms every vertex from world space directly to clip space.

### OrbitControls

**OrbitControls** (`controls/orbit.ts`) provide mouse/touch camera interaction. The camera's position is defined in **spherical coordinates** (azimuth angle, elevation angle, distance) relative to a target point.

Input mapping:

- **Left drag / 1-finger** → Orbit (rotate azimuth and elevation).
- **Right drag / 2-finger** → Pan (shift the target point).
- **Scroll / pinch** → Zoom (change distance).

All input uses Pointer Events, which unify mouse and touch into a single code path. Multi-touch (pinch zoom + two-finger pan) tracks active pointers in a Map. Velocity is added on input, then **damping** smoothly decays the velocity each frame for inertia — releasing a drag doesn't stop the camera instantly but lets it coast to a stop.

---

## The Scheduler

The **Scheduler** (`scheduler.ts`) manages a single `requestAnimationFrame` loop that drives the entire application.

### How It Works

Callbacks are registered with:

- **Priority** (numeric, lower runs first, can be negative) — Control execution order. For example, animation updates might run at priority -1, physics at 0, rendering at 1.
- **FPS throttle** (optional, per-callback) — Limit a callback to at most N executions per second.

A global FPS cap can limit the overall tick rate (e.g., 30 FPS for battery savings).

Both throttles use **accumulator-based remainder tracking** so the average rate converges accurately to the target — a simple "skip every other frame" approach would give 30 FPS on a 60 Hz display but only 20 FPS on a 90 Hz display.

### Design Points

- **Zero allocation** in the hot path: a single `SchedulerState` object (`{ dt, elapsed, frame }`) is reused every frame.
- **Lazy compaction**: removed callbacks are nulled out, then compacted on the next sort pass (avoids array splicing during iteration).
- **Mid-frame safety**: new registrations don't execute in the same frame they're added; removals are skipped via a null guard.

---

## Sprites and Billboards

A **Sprite** (`scene/sprite.ts`) is a flat rectangular plane that automatically orients to face the camera every frame. Sprites are useful for particles, labels, health bars, or any flat element that should always be visible regardless of viewing angle.

All sprites share a single 1×1 `PlaneGeometry` (avoiding per-sprite allocation). The renderer computes a **billboard world matrix** on the CPU by extracting the camera's right and up vectors from the view matrix, then building a new matrix that uses these as the sprite's local X/Y axes while preserving its world position and scale.

**SpriteMaterial** properties:

- **rotation** — 2D rotation (radians) around the view axis.
- **sizeAttenuation** — When true (default), sprites shrink with distance like normal geometry. When false, sprites maintain constant screen size regardless of distance (useful for labels or HUD markers).

Sprites default to `castShadow: false` and use `SpriteMaterial` (transparent, unlit).

---

## HTML Overlays

The **overlay system** (`overlay.ts`) positions HTML DOM elements to track 3D world positions on screen. This is useful for name labels, tooltips, health bars, or any UI that should be anchored to a 3D object but rendered with full HTML/CSS capabilities.

### Projection Pipeline

Each frame, the overlay manager projects tracked positions through:

```
World position → Clip space (via view-projection matrix) → Perspective divide → NDC → Screen pixels
```

The resulting screen position is applied via a CSS `translate3d` transform (GPU-accelerated, no layout thrash). A 1.1 frustum margin prevents labels from popping out abruptly at screen edges.

### Optimizations

- **Dirty checking** — DOM writes are skipped when the position change is sub-pixel (< 0.5px).
- **Centering** via `CSS translate(-50%, -50%)` — No JavaScript element measurement needed.
- **Depth-based z-index** — Closer overlays render on top.
- **Distance scaling** — Optional perspective-consistent sizing.
- **Pointer events** — The overlay container is `pointer-events: none` by default; individual elements opt in.

---

## React Bindings

VoidCore includes an optional declarative React layer that maps JSX elements to engine objects, based on [React Three Fiber](https://github.com/pmndrs/react-three-fiber)'s API design. This lets you describe 3D scenes the same way you describe DOM UIs.

### Custom Reconciler

The core of the React integration is a **custom react-reconciler** (`react/reconciler.ts`). React's reconciler is the engine behind the virtual DOM diff — it determines what changed and what DOM operations to perform. By providing a custom "host config," VoidCore teaches React to create and manage engine objects (Mesh, Group, Light, etc.) instead of DOM elements.

The reconciler maps JSX tag names to engine constructors via a catalogue:

```jsx
<mesh>            →  new Mesh()
<boxGeometry>     →  new BoxGeometry()
<lambertMaterial> →  new LambertMaterial()
<directionalLight> → new DirectionalLight()
<ambientLight>    →  new AmbientLight()
<group>           →  new Group()
<sprite>          →  new Sprite()
```

Props are applied to engine objects through `applyProps`, which handles position arrays, color arrays, scalar properties, and event handlers. Geometry and material children auto-attach to their parent mesh via the `attach` convention.

### Canvas Component

`<Canvas>` (`react/Canvas.tsx`) is the root component. On mount, it creates the engine (async), scene, and camera, starts the rAF loop, and mounts the reconciler. It also hosts an overlay `<div>` where `Html` components tunnel their DOM content.

### Hooks

- **useEngine()** — Access engine, scene, camera, and canvas from context.
- **useFrame(callback)** — Register a function that runs every frame before rendering. The callback receives `{ dt, elapsed, frame, engine, scene, camera }`.
- **useGLTF(url)** — Load a glTF/GLB model with Suspense support. Options include `{ meshName }` to extract a single mesh by name, and `{ clone: true }` for independent instancing.
- **useKTX2(url)** — Load a KTX2 compressed texture.
- **useColoredGeometry(geometry, palette)** — Memoized `bakePalette()` wrapper.
- **useAnimations(clips, skeleton)** — Create an AnimationMixer and return named actions.

### Events

The event system (`react/events.ts`) attaches DOM pointer listeners to the canvas and converts pointer positions to raycasts. When a raycast hit is found, it walks up the instance tree to find components with event handlers (`onClick`, `onPointerOver`, `onPointerOut`, `onPointerDown`, `onPointerUp`, `onPointerMove`) and dispatches synthetic events with 3D intersection data (hit point, normal, distance, UV).

### Html Component

`<Html>` (`react/Html.tsx`) renders React DOM content as an overlay positioned at a 3D coordinate. It participates in the scene graph (inheriting parent transforms via a `<group>`) and tunnels its DOM children to the Canvas's overlay div using a **tunnel pattern** (inspired by pmndrs/tunnel-rat). Each frame, it projects its world position through the camera and updates its CSS transform.

### BakeShadows

`<BakeShadows />` (`react/BakeShadows.tsx`) is a declarative component that freezes the shadow map on mount and resumes real-time shadow updates on unmount.

---

## Performance Philosophy

Several design decisions throughout VoidCore are motivated by real-time performance constraints:

### Zero Allocation in Hot Paths

The render loop runs 60+ times per second. Allocating objects in the loop creates garbage that the JavaScript GC must eventually collect, causing frame hitches (stutters). VoidCore eliminates hot-path allocations through:

- **Write-into-output math** — `vec3Add(out, a, b)` instead of returning new arrays.
- **Pre-allocated scheduler state** — A single `SchedulerState` object reused every frame.
- **Pre-allocated traversal stacks** — Scene graph traversal, BVH traversal, and mesh collection use pre-sized arrays with index tracking.
- **Pre-allocated sort buffers** — Radix sort state (keys, indices, histograms) is created once.
- **Pre-allocated raycaster buffers** — Scratch matrices, vectors, and optional pre-allocated hit targets.
- **Reusable overlay math** — Projection vectors and matrices allocated once.

### Minimizing GPU State Changes

Changing the active shader, material, blend mode, or texture on the GPU is expensive. The radix sort groups draws by pipeline → material → depth, so consecutive draws often share the same state. The WebGL2 renderer additionally uses a **state cache** (tracking the currently bound program, VAO, blend mode, etc.) to skip redundant API calls.

### Vertex Packing

Smaller vertex attributes mean less data transferred from CPU to GPU and less bandwidth consumed during vertex fetching. The 3× savings on normals and 4× on colors add up with dense meshes.

### Dirty Flags and Subtree Skipping

The scene graph only recomputes matrices for nodes that actually changed. The `_subtreeDirty` flag allows entire static subtrees to be skipped during the update traversal. In a scene with 500 objects and 10 animated characters, this means visiting ~100 nodes instead of 500.

### Batched Uniform Uploads

Instead of N individual GPU uploads for N objects, per-object data is packed into contiguous buffers and uploaded in 1–2 bulk operations. This reduces API call overhead significantly.

### Squared Distance Comparisons

Distance culling and distance-based LOD decisions use `dx² + dy² + dz²` instead of `sqrt(dx² + dy² + dz²)` — comparing squared distances against squared thresholds avoids the expensive square root.

### Shadow Baking

For static scenes, the shadow map is rendered once and frozen, eliminating the entire shadow pass for subsequent frames.

### DPR Limiting

High-DPI displays (Retina, 4K) have pixel ratios of 2–3×, meaning the GPU must shade 4–9× as many pixels. VoidCore defaults to capping the effective DPR (1.25 on mobile, 1.5 on desktop) for a practical balance between visual quality and performance.

---

## Credits

This codebase was entirely AI-generated. While it is impossible to pinpoint exactly what other projects VoidCore takes inspiration from, it is clearly standing on the shoulders of [Three.js](https://threejs.org/), [React Three Fiber](https://github.com/pmndrs/react-three-fiber), and [three-mesh-bvh](https://github.com/gkjohnson/three-mesh-bvh). Huge thanks to the authors and contributors of these projects for paving the way.
