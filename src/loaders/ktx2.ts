// KTX2 Loader – Decodes KTX2 compressed textures using the Basis Universal transcoder.
//
// KTX2 is a GPU texture container format that stores Basis Universal compressed data.
// The Basis transcoder (a WASM module) decompresses the texture data at load time.
// This loader always transcodes to RGBA8 for maximum compatibility across all GPU backends.
//
// The loader follows the same pattern as Draco decoding:
//   1. Load the Basis transcoder WASM module via a <script> tag (cached after first load)
//   2. Fetch the .ktx2 file as an ArrayBuffer
//   3. Transcode the compressed data to RGBA8 pixels
//   4. Return a Texture object ready for use as a material map
//
// loadKTX2(url, transcoderPath) – Loads a .ktx2 file and returns a decoded Texture.

import { Texture } from '../materials/texture'

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

export const loadKTX2 = async (url: string, transcoderPath: string): Promise<Texture> => {
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

  // Transcode to RGBA32 (format code 13) for maximum compatibility
  const cTFRGBA32 = 13
  const imageSize = ktx2File.getImageTranscodedSizeInBytes(0, 0, 0, cTFRGBA32)
  const rgba = new Uint8Array(imageSize)

  if (!ktx2File.transcodeImage(rgba, 0, 0, 0, cTFRGBA32, 0, -1, -1)) {
    ktx2File.close()
    ktx2File.delete()
    throw new Error('KTX2 image transcoding failed')
  }

  ktx2File.close()
  ktx2File.delete()

  return new Texture({ width, height, data: rgba })
}
