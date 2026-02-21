// KTX2 Loader – Decodes KTX2 compressed textures using the Basis Universal transcoder.
//
// KTX2 is a GPU texture container format that stores Basis Universal compressed data
// (either ETC1S or UASTC supercompressed). The Basis transcoder (a WASM module) can
// transcode from these intermediate formats into GPU-native compressed formats at load time.
//
// Instead of always decompressing to uncompressed RGBA8 (32 bpp), this loader picks the
// best GPU-native compressed format supported by the device — halving texture memory and
// bandwidth on devices with ASTC, BC7, BC3, or ETC2 support:
//
//   Priority: ASTC 4×4 > BC7 > ETC2 RGBA > BC3 > RGBA8 (fallback)
//
// The caller passes `supportedFormats` (from renderer.compressedTextureFormats) so the
// loader can pick the best match. If none are provided, it falls back to RGBA8.
//
// The loader follows the same pattern as Draco decoding:
//   1. Load the Basis transcoder WASM module via a <script> tag (cached after first load)
//   2. Fetch the .ktx2 file as an ArrayBuffer
//   3. Transcode to the best supported GPU-compressed format (or RGBA8 fallback)
//   4. Return a Texture object with format metadata ready for GPU upload
//
// loadKTX2(url, transcoderPath, supportedFormats?) – Loads a .ktx2 file and returns a Texture.

import { Texture } from '../materials/texture'

import type { CompressedTextureFormat, TextureFormat } from '../materials/texture'

// Basis Universal transcoder format codes
const BASIS_FORMAT_ASTC_4x4 = 10
const BASIS_FORMAT_BC7 = 6
const BASIS_FORMAT_BC3 = 3
const BASIS_FORMAT_ETC2_RGBA = 1
const BASIS_FORMAT_RGBA32 = 13

// Maps our CompressedTextureFormat names to Basis transcoder format codes
const FORMAT_TO_BASIS: Record<CompressedTextureFormat, number> = {
  'astc-4x4': BASIS_FORMAT_ASTC_4x4,
  bc7: BASIS_FORMAT_BC7,
  bc3: BASIS_FORMAT_BC3,
  'etc2-rgba8': BASIS_FORMAT_ETC2_RGBA,
}

let basisPromise: Promise<any> | null = null

const loadBasisTranscoder = (transcoderPath: string): Promise<any> => {
  if (basisPromise) return basisPromise

  basisPromise = new Promise<any>((resolve, reject) => {
    const url = transcoderPath.endsWith('/') ? transcoderPath : transcoderPath + '/'
    const script = document.createElement('script')
    script.src = `${url}basis_transcoder.js`
    script.onload = () => {
      const factory = (globalThis as any).BASIS
      if (!factory) {
        reject(new Error('BASIS not found after script load'))
        return
      }
      factory({ locateFile: (f: string) => `${url}${f}` })
        .then((module: any) => {
          module.initializeBasis()
          resolve(module)
        })
        .catch(reject)
    }
    script.onerror = () => reject(new Error('Failed to load Basis transcoder'))
    document.head.appendChild(script)
  })

  return basisPromise
}

/**
 * Selects the best Basis transcoder target format from the list of device-supported formats.
 * Returns the Basis format code and our TextureFormat name.
 * The `supportedFormats` array is already in priority order (from the renderer).
 */
const selectBestFormat = (
  supportedFormats?: readonly CompressedTextureFormat[],
): { basisFormat: number; textureFormat: TextureFormat } => {
  if (supportedFormats) {
    for (const fmt of supportedFormats) {
      const basisCode = FORMAT_TO_BASIS[fmt]
      if (basisCode !== undefined) {
        return { basisFormat: basisCode, textureFormat: fmt }
      }
    }
  }
  return { basisFormat: BASIS_FORMAT_RGBA32, textureFormat: 'rgba8' }
}

export const loadKTX2 = async (
  url: string,
  transcoderPath: string,
  supportedFormats?: readonly CompressedTextureFormat[],
): Promise<Texture> => {
  const basis = await loadBasisTranscoder(transcoderPath)

  const response = await fetch(url)
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`)
  const buf = await response.arrayBuffer()
  const data = new Uint8Array(buf)

  const ktx2File = new basis.KTX2File(data)
  const width = ktx2File.getWidth()
  const height = ktx2File.getHeight()

  if (!ktx2File.startTranscoding()) {
    ktx2File.close()
    ktx2File.delete()
    throw new Error('KTX2 transcoding failed to start')
  }

  const { basisFormat, textureFormat } = selectBestFormat(supportedFormats)
  const imageSize = ktx2File.getImageTranscodedSizeInBytes(0, 0, 0, basisFormat)
  const pixels = new Uint8Array(imageSize)

  if (!ktx2File.transcodeImage(pixels, 0, 0, 0, basisFormat, 0, -1, -1)) {
    ktx2File.close()
    ktx2File.delete()
    throw new Error('KTX2 image transcoding failed')
  }

  ktx2File.close()
  ktx2File.delete()

  return new Texture({ width, height, data: pixels, format: textureFormat })
}
