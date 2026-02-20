import { useMemo } from 'react'

import { LambertMaterial, mergeGeometries, prebuildBVH } from '../src/index'
import { useGLTF, useKTX2 } from '../src/react/index'
import { staticBundleSrc, cityAoSrc } from './assets'

const EDEN_COLORS: [number, number, number][] = [
  [0.78, 0.44, 0.25],
  [0.85, 0.68, 0.3],
  [0.75, 0.75, 0.75],
  [0.725, 0.38, 0.09],
  [0.25, 0.55, 0.2],
  [0.212, 0.212, 0.212],
  [0.75, 0.75, 0.75],
  [0.15, 0.15, 0.18],
  [0.0, 0.75, 0.7],
  [0.55, 0.5, 0.42],
  [0.85, 0.35, 0.55],
  [0.95, 0.95, 0.95],
  [0.8, 0.65, 0.15],
  [0.65, 0.65, 0.62],
  [0.3, 0.65, 0.2],
  [0.5, 0.32, 0.15],
  [0.5, 0.25, 0.65],
  [0.9, 0.8, 0.2],
  [0.15, 0.15, 0.18],
  [0.75, 0.75, 0.75],
  [0.95, 0.95, 0.95],
  [0.6, 0.4, 0.22],
  [0.6, 0.4, 0.22],
  [0.0, 0.75, 0.7],
  [0.9, 0.8, 0.2],
  [0.85, 0.35, 0.55],
  [0.3, 0.36, 0.3],
]

const EDEN_PALETTE = EDEN_COLORS.map((color, i) => ({
  color,
  ...(i === 23 ? { emissive: [0.0, 0.75, 0.7] as [number, number, number], emissiveIntensity: 2.5 } : {}),
}))

const EdenMesh = ({ onReady }: { onReady?: () => void }) => {
  const gltf = useGLTF(staticBundleSrc, { draco: { decoderPath: '/draco-1.5.7/' } })
  const aoTexture = useKTX2(cityAoSrc, '/basis-1.50/')

  const { geometry, material } = useMemo(() => {
    const edenMeshes = gltf.meshes.filter(m => m.name === 'Eden')
    const geometry = mergeGeometries(edenMeshes.map(m => m.geometry))
    prebuildBVH(geometry)
    onReady?.()
    return {
      geometry: mergeGeometries(edenMeshes.map(m => m.geometry)),
      material: new LambertMaterial({ palette: EDEN_PALETTE, aoMap: aoTexture, aoIntensity: 2 }),
    }
  }, [gltf, aoTexture, onReady])

  return <mesh geometry={geometry} material={material} name="eden" castShadow receiveShadow position={[-50, -70, 0]} />
}

export default EdenMesh
