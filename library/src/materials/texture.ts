// Texture – Holds pixel data for GPU texture maps (color maps, AO maps, etc.).
//
// A Texture stores pixel data along with its dimensions and format. For uncompressed textures,
// the data is raw RGBA8. For compressed textures (ASTC, BC7, BC3, ETC2), the data contains
// GPU-native compressed blocks that are uploaded directly — halving memory and bandwidth vs RGBA8.
//
// The `format` field tells the renderer which GPU texture format to use:
//   'rgba8'      – Uncompressed RGBA (32 bpp), universal fallback
//   'astc-4x4'   – ASTC 4×4 blocks (8 bpp), best for modern mobile GPUs
//   'bc7'        – BC7 blocks (8 bpp), best for modern desktop GPUs
//   'bc3'        – BC3/DXT5 blocks (8 bpp), wider desktop support than BC7
//   'etc2-rgba8' – ETC2 RGBA blocks (8 bpp), native on all mobile WebGL2/GLES3 devices
//
// Textures are typically created by asset loaders (e.g. loadKTX2) and assigned to material
// properties like `colorMap` or `aoMap`.
//
// new Texture({ width, height, data, format? }) – Creates a texture.

export type CompressedTextureFormat = 'astc-4x4' | 'bc7' | 'bc3' | 'etc2-rgba8'

export type TextureFormat = 'rgba8' | CompressedTextureFormat

export interface TextureData {
  width: number
  height: number
  data: Uint8Array
  /** GPU texture format. Defaults to 'rgba8' (uncompressed). */
  format?: TextureFormat
}

let _nextTextureId = 0

export class Texture {
  readonly _id: number
  width: number
  height: number
  data: Uint8Array
  format: TextureFormat

  constructor(data: TextureData) {
    this._id = _nextTextureId++
    this.width = data.width
    this.height = data.height
    this.data = data.data
    this.format = data.format ?? 'rgba8'
  }
}
