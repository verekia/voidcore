import { Suspense, useMemo } from 'react'

import { LambertMaterial, prebuildBVH, BakeShadows, useKTX2, useColoredStaticGeometry } from 'voidcore'

import { cityAoSrc } from './assets'

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
  ...(i === 23 ? { emissive: [0.0, 0.75, 0.7] as [number, number, number], emissiveIntensity: 1.5 } : {}),
}))

const EdenMesh = ({ onReady }: { onReady?: () => void }) => {
  const geometry = useColoredStaticGeometry('Eden', EDEN_PALETTE)
  const aoTexture = useKTX2(cityAoSrc)

  const material = useMemo(() => {
    prebuildBVH(geometry)
    onReady?.()
    return new LambertMaterial({
      aoMap: aoTexture,
      aoIntensity: 2,
      emissiveBrightness: 0.5,
    })
  }, [geometry, aoTexture, onReady])

  return (
    <Suspense fallback={null}>
      <mesh geometry={geometry} material={material} name="eden" castShadow receiveShadow position={[-50, -70, 0]} />
      <BakeShadows />
    </Suspense>
  )
}

export default EdenMesh
