# Voidcore — Renderer (Remaining Work)

## Conditional Vertex Attribute Fetch

The current implementation always allocates and binds all 4 attribute slots (position, normal, UV, materialIndex) even when the geometry doesn't have UV or materialIndex data. Skinned attributes (joints/weights) are correctly conditional, but the static attributes are not.

The spec calls for separate-per-attribute buffers where unused attributes are simply not allocated or bound:

```
HAS_COLOR_TEXTURE    = 0x01
HAS_AO_TEXTURE       = 0x02
HAS_VERTEX_COLORS    = 0x04
HAS_MATERIAL_INDEX   = 0x08
HAS_SKINNING         = 0x10
HAS_EMISSIVE         = 0x20
SHADOW_RECEIVE       = 0x40
IS_TRANSPARENT       = 0x80
```

Shaders would be compiled as variants via `#ifdef` guards keyed on this bitmask:

```glsl
#ifdef HAS_COLOR_TEXTURE
  uniform sampler2D u_colorMap;
#endif
#ifdef HAS_AO_TEXTURE
  uniform sampler2D u_aoMap;
#endif
#ifdef HAS_VERTEX_COLORS
  in vec4 a_color;
#endif
#ifdef HAS_MATERIAL_INDEX
  in float a_materialIndex;
#endif
```

The feature mask is derived per-mesh from the combination of material config and geometry attributes. Variants would be compiled lazily on first use and cached by bitmask. Typical scene uses 10–30 unique variants.

This is a significant refactor — it requires changing both vertex buffer allocation in the renderer and the shader source for both WGSL and GLSL backends.

## Unified Bind Group Layout for Shadow Pass

The shadow pass currently uses a separate `shadowBGL` bind group layout instead of reusing the per-frame bind group (slot 0) that the main passes use. The spec describes a single unified per-frame layout shared across all passes.

The three-slot layout:

| Slot | Name         | Update Frequency               | Contents                                                             |
| ---- | ------------ | ------------------------------ | -------------------------------------------------------------------- |
| 0    | Per-frame    | Once per frame                 | Camera VP matrix, light data, ambient, shadow VP matrix, bias params |
| 1    | Per-material | Per material switch            | Material UBO, color/AO textures, samplers                            |
| 2    | Per-object   | Per draw call (dynamic offset) | World matrix, optional bone matrices                                 |

Currently the shadow pass duplicates slot 0 with its own layout. Unifying them would reduce bind group creation overhead and simplify the renderer code.

## WebGL2 Pipeline as State Bundle

The WebGPU backend correctly uses immutable `GPURenderPipeline` objects — each unique combination of (shader + blend + depth + cull) is a cached pipeline object, and switching between them is cheap.

The WebGL2 backend does not have an equivalent. Instead it sets blend/depth/cull state ad-hoc during rendering for each draw. The spec calls for a frozen "pipeline" record that maps to the state cache diff:

```typescript
interface WebGL2Pipeline {
  program: WebGLProgram
  depthWrite: boolean
  depthFunc: number
  blendEnabled: boolean
  blendSrc: number
  blendDst: number
  cullFace: number | null
}
```

Applying a pipeline would compare each field against the cached state and call only the GL functions that actually changed, rather than setting everything every draw.

## Render Bundles (WebGPU)

Not implemented. For mostly-static scenes, WebGPU render bundles can pre-record draw commands for geometry that doesn't change between frames, replaying them with near-zero JS overhead.

- Typical split: ~1800 static + ~200 dynamic objects
- Bundles would be invalidated on scene structure changes, material changes, or visibility changes
- The sort-based renderer is the primary optimization strategy — bundles are an optional acceleration on top

This is lower priority since the sorted draw path already hits the performance target on both backends.
