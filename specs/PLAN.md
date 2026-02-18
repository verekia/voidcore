# Voidcore — Implementation Plan

## Overview

**Voidcore** is a minimal, high-performance 3D game engine for the web. It targets stylized and low-poly games with a developer experience inspired by Three.js but engineered for drastically better performance. The hard performance target is **2000 draw calls at 60fps on recent mobile phones**.

## Design Philosophy

**Performance is architecture, not optimization.** Every structural decision — from data layout to sort strategy to uniform upload — traces back to the 16.6ms frame budget. The engine achieves performance through design, not through post-hoc profiling and patching.

**Minimal scope, done well.** Two materials (Basic + Lambert), one shadow-casting light type (directional), seven parametric primitives, one transparency technique (WBOIT), one post-processing effect (bloom). Each feature is complete and polished rather than broad and shallow.

**Familiar API, modern internals.** The public API uses factory functions and object-oriented patterns familiar to Three.js developers. Internally, the renderer operates on contiguous typed arrays for cache-friendly GPU upload. Users never see the SoA layout — they interact with objects that have named properties.

## Key Architectural Decisions

| Decision             | Choice                                                          | Rationale                                                     |
| -------------------- | --------------------------------------------------------------- | ------------------------------------------------------------- |
| Language             | TypeScript strict, no semicolons, const arrow functions         | Full type safety, clean code style                            |
| GPU primary          | WebGPU                                                          | Lower per-draw overhead, immutable pipelines                  |
| GPU fallback         | WebGL2                                                          | Broad compatibility                                           |
| Package structure    | Single `voidcore` package with `/react` subpath export          | Simple distribution, one version, one install                 |
| Public API style     | Object graph with factory functions                             | Familiar DX, debuggable                                       |
| Internal data layout | SoA typed arrays                                                | Cache-friendly, GPU-uploadable                                |
| Coordinate system    | Z-up, right-handed (X=right, Y=forward, Z=up)                   | Blender convention, prompt requirement                        |
| Materials            | Basic (unlit) + Lambert (N·L diffuse)                           | Sufficient for stylized/low-poly                              |
| Transparency         | Weighted Blended OIT                                            | No sorting, single-pass, no ordering artifacts                |
| Shadows              | 3-cascade CSM, 1024px default                                   | Smooth shadows for 200m worlds                                |
| Bloom                | Unreal-style progressive downsample/upsample                    | Per-vertex emissive control via MRT                           |
| BVH                  | Binned SAH, flat 32-byte nodes                                  | three-mesh-bvh quality target                                 |
| Animation            | GPU skinning, 4-bone weights, 32-bone limit, action-based mixer | glTF skeletal animation with crossfade, UBO always sufficient |
| Assets               | glTF 2.0 + Draco + KTX2/Basis                                   | Standard pipeline, user-provided decoders                     |
| React bindings       | Custom reconciler at `voidcore/react`                           | R3F-style declarative scene, same package                     |
| Math                 | Custom Float32Array-backed, functional API                      | Zero-alloc, Z-up baked in, GPU-uploadable                     |
| Draw call sorting    | 32-bit radix sort (`Uint32Array`)                               | O(n), stable, no BigInt overhead                              |
| Uniforms             | Dynamic offsets on shared buffer                                | Minimal binding overhead                                      |
| Instancing           | None                                                            | Per-draw overhead <2-3μs makes it unnecessary at 2000 draws   |

## Package Structure

```
voidcore/
  src/
    math/           # Vec3, Mat4, Quat, AABB, Frustum, Ray
    device/         # GPU abstraction (Device, Buffer, Texture, Pipeline, BindGroup)
    renderer/       # Render loop, draw call sorting, state management
    scene/          # Scene graph, nodes, transforms, dirty flags
    materials/      # BasicMaterial, LambertMaterial, shader variants
    geometry/       # Parametric primitives, buffer layouts
    animation/      # Skeleton, AnimationMixer, crossfade, bone attachment
    loaders/        # glTF, Draco, KTX2/Basis
    lighting/       # Directional light, ambient light, CSM shadows
    spatial/        # BVH, raycasting, frustum culling
    postfx/         # Bloom, MSAA, OIT composite
    controls/       # OrbitControls
    overlay/        # HTML overlay system
    react/          # Custom reconciler, Canvas, hooks, Html component
      reconciler.ts
      Canvas.tsx
      hooks.ts
      Html.tsx
      components.ts
  package.json
```

The React bindings live inside the main package and are exposed via a `/react` subpath export in `package.json`:

```json
{
  "name": "voidcore",
  "exports": {
    ".": "./dist/index.js",
    "./react": "./dist/react/index.js"
  },
  "peerDependencies": {
    "react": "^18.0.0 || ^19.0.0",
    "react-dom": "^18.0.0 || ^19.0.0"
  },
  "peerDependenciesMeta": {
    "react": { "optional": true },
    "react-dom": { "optional": true }
  }
}
```

Users who don't use React pay zero cost — `react` and `react-dom` are optional peer dependencies, and the `/react` export is never imported unless explicitly used. Tree-shaking eliminates all React code from non-React bundles.

```typescript
// Vanilla usage — no React dependency loaded
import { createEngine, createScene, createMesh } from 'voidcore'

// React usage — imports reconciler, Canvas, hooks
import { Canvas, useFrame, useGLTF } from 'voidcore/react'
```

## v1 Feature Scope

All of the following ship in v1:

- WebGL2 and WebGPU backends with auto-detection
- Basic (unlit) and Lambert (N·L diffuse) materials
- Material index palette system (32 entries) with per-vertex color/emissive
- Vertex colors (multiplicative blending)
- glTF 2.0 with Draco mesh compression (user-provided decoder)
- KTX2/Basis Universal texture compression (user-provided transcoder, ASTC > BC7 > ETC2 > RGBA8)
- Color textures, AO textures
- Per-vertex bloom via MRT emissive output (Unreal-style progressive)
- 4x MSAA (configurable)
- SAH-based BVH raycasting (three-mesh-bvh quality)
- GPU-skinned meshes with skeletal animation and crossfade
- Frustum culling (AABB, hierarchical)
- Weighted Blended OIT transparency
- Orbit controls (mouse + touch)
- 7 parametric geometry primitives (plane, box, sphere, cone, cylinder, capsule, circle)
- Directional light + ambient light
- 3-cascade CSM shadows for 200x200m low-poly worlds
- Parent/child scene graph with Groups
- Bone attachment for static meshes
- HTML overlay system (CSS-positioned DOM elements at 3D positions)
- React bindings (R3F-style, `voidcore/react` subpath)
- Frame statistics API

## Out of Scope for v1

- Point lights, spot lights
- PBR materials (metallic-roughness, normal maps, IBL)
- Morph targets / blend shapes
- Physics integration
- Text rendering
- Particle systems
- Post-processing beyond bloom
- PCSS / VSM / ESM shadow techniques
- Multiple shadow-casting lights
- Automatic instancing
