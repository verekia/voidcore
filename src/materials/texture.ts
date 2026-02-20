// Texture – Holds decoded pixel data for GPU texture maps (color maps, AO maps, etc.).
//
// A Texture stores raw RGBA8 pixel data along with its dimensions. This data is uploaded
// to the GPU by the renderer when the texture is first used, and cached for subsequent frames.
// Textures are typically created by asset loaders (e.g. loadKTX2) and assigned to material
// properties like `colorMap` or `aoMap`.
//
// new Texture({ width, height, data }) – Creates a texture from raw RGBA pixel data.

export interface TextureData {
  width: number
  height: number
  data: Uint8Array
}

let _nextTextureId = 0

export class Texture {
  readonly _id: number
  width: number
  height: number
  data: Uint8Array

  constructor(data: TextureData) {
    this._id = _nextTextureId++
    this.width = data.width
    this.height = data.height
    this.data = data.data
  }
}
