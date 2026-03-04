import { Suspense, useMemo } from 'react'

import {
  LambertMaterial,
  Mesh,
  prebuildBVH,
  BakeShadows,
  useKTX2,
  useGLTF,
  useColoredGeometry,
  useGrass,
  type PaletteEntry,
} from 'voidcore'

import { staticBundleSrc, cityAoSrc, grassAoSrc, grassNormalSrc, groundAoSrc, groundNormalSrc } from './assets'

const EdenMesh = ({ onReady }: { onReady?: () => void }) => {
  const aoTexture = useKTX2(cityAoSrc)
  const grassAoTexture = useKTX2(grassAoSrc)
  const groundAoTexture = useKTX2(groundAoSrc)
  const grassNormalTexture = useKTX2(grassNormalSrc)
  const groundNormalTexture = useKTX2(groundNormalSrc)

  const palette = useMemo<PaletteEntry[]>(
    () => [
      {
        color: [0.78, 0.44, 0.25],
        tiledAo: groundAoTexture,
        tiledAoIntensity: 0.3,
        tiledAoScale: 150,
        tiledNormal: groundNormalTexture,
        tiledNormalIntensity: 0.3,
        tiledNormalScale: 150,
      },
      {
        color: [0.85, 0.68, 0.3],
        tiledAo: groundAoTexture,
        tiledAoIntensity: 0.3,
        tiledAoScale: 150,
        tiledNormal: groundNormalTexture,
        tiledNormalIntensity: 0.5,
        tiledNormalScale: 150,
      },
      { color: [0.75, 0.75, 0.75] },
      {
        color: [0.725, 0.38, 0.09],
        tiledAo: groundAoTexture,
        tiledAoIntensity: 0.3,
        tiledAoScale: 150,
        tiledNormal: groundNormalTexture,
        tiledNormalIntensity: 0.5,
        tiledNormalScale: 150,
      },
      {
        color: [0.25, 0.55, 0.2],
        tiledAo: groundAoTexture,
        tiledAoIntensity: 0.3,
        tiledAoScale: 150,
        tiledNormal: groundNormalTexture,
        tiledNormalIntensity: 0.5,
        tiledNormalScale: 150,
      },
      {
        color: [0.212, 0.212, 0.212],
        tiledAo: groundAoTexture,
        tiledAoIntensity: 0.3,
        tiledAoScale: 150,
        tiledNormal: groundNormalTexture,
        tiledNormalIntensity: 0.5,
        tiledNormalScale: 150,
      },
      { color: [0.75, 0.75, 0.75] },
      { color: [0.15, 0.15, 0.18] },
      {
        color: [0.0, 0.75, 0.7],
        emissive: [0.0, 0.75, 0.7],
        emissiveIntensity: 1.5,
      },
      {
        color: [0.55, 0.5, 0.42],
        tiledAo: groundAoTexture,
        tiledAoIntensity: 0.3,
        tiledAoScale: 150,
        tiledNormal: groundNormalTexture,
        tiledNormalIntensity: 0.5,
        tiledNormalScale: 150,
      },
      {
        color: [0.85, 0.35, 0.55],
        emissive: [0.85, 0.35, 0.55],
        emissiveIntensity: 0.7,
      },
      { color: [0.95, 0.95, 0.95] },
      { color: [0.6, 0.3, 0.15] },
      {
        color: [0.65, 0.65, 0.62],
        tiledAo: groundAoTexture,
        tiledAoIntensity: 0.3,
        tiledAoScale: 150,
        tiledNormal: groundNormalTexture,
        tiledNormalIntensity: 0.5,
        tiledNormalScale: 150,
      },
      {
        color: [0.4, 0.6, 0.133],
        color2: [0.2, 0.4, 0.05],
        noiseScale: 0.08,
        tiledAo: grassAoTexture,
        tiledAoIntensity: 0.8,
        tiledAoScale: 100,
        tiledNormal: grassNormalTexture,
        tiledNormalIntensity: 0.6,
        tiledNormalScale: 100,
      },
      {
        color: [0.5, 0.32, 0.15],
        tiledAo: groundAoTexture,
        tiledAoIntensity: 0.3,
        tiledAoScale: 150,
        tiledNormal: groundNormalTexture,
        tiledNormalIntensity: 0.5,
        tiledNormalScale: 150,
      },
      { color: [0.5, 0.25, 0.65] },
      {
        color: [0.9, 0.8, 0.2],
        tiledAo: groundAoTexture,
        tiledAoIntensity: 0.3,
        tiledAoScale: 150,
        tiledNormal: groundNormalTexture,
        tiledNormalIntensity: 0.5,
        tiledNormalScale: 150,
      },
      { color: [0.15, 0.15, 0.18] },
      {
        color: [0.75, 0.75, 0.75],
        tiledAo: groundAoTexture,
        tiledAoIntensity: 0.3,
        tiledAoScale: 150,
        tiledNormal: groundNormalTexture,
        tiledNormalIntensity: 0.5,
        tiledNormalScale: 150,
      },
      { color: [0.95, 0.95, 0.95] },
      {
        color: [0.6, 0.4, 0.22],
        tiledAo: groundAoTexture,
        tiledAoIntensity: 0.3,
        tiledAoScale: 150,
        tiledNormal: groundNormalTexture,
        tiledNormalIntensity: 0.5,
        tiledNormalScale: 150,
      },
      {
        color: [0.6, 0.4, 0.22],
        tiledAo: groundAoTexture,
        tiledAoIntensity: 0.3,
        tiledAoScale: 150,
        tiledNormal: groundNormalTexture,
        tiledNormalIntensity: 0.5,
        tiledNormalScale: 150,
      },
      {
        color: [0.0, 0.75, 0.7],
        emissive: [0.0, 0.75, 0.7],
        emissiveIntensity: 1.5,
      },
      {
        color: [0.9, 0.8, 0.2],
        tiledAo: groundAoTexture,
        tiledAoIntensity: 0.3,
        tiledAoScale: 150,
        tiledNormal: groundNormalTexture,
        tiledNormalIntensity: 0.5,
        tiledNormalScale: 150,
      },
      { color: [0.85, 0.35, 0.55] },
      {
        color: [0.3, 0.36, 0.3],
        tiledAo: groundAoTexture,
        tiledAoIntensity: 0.3,
        tiledAoScale: 150,
        tiledNormal: groundNormalTexture,
        tiledNormalIntensity: 0.5,
        tiledNormalScale: 150,
      },
    ],
    [grassAoTexture, groundAoTexture, grassNormalTexture, groundNormalTexture],
  )

  // Load raw mesh (with materialIndices) for grass generation
  const edenMesh = useGLTF(staticBundleSrc, { meshName: 'Eden' }) as Mesh
  const grassGridMesh = useGLTF(staticBundleSrc, {
    meshName: 'EdenGrassGrid',
  }) as Mesh
  const geometry = useColoredGeometry(edenMesh.geometry, palette)

  const grass = useGrass(grassGridMesh.geometry, palette, {
    bladesPerPatch: 15,
    patchRadius: 0.65,
    radius: 80,
    minWidth: 0.1,
    maxWidth: 0.2,
    color: [0.4, 0.6, 0.133],
    color2: [0.2, 0.4, 0.05],
  })

  const material = useMemo(() => {
    prebuildBVH(geometry)
    onReady?.()
    return new LambertMaterial({
      aoMap: aoTexture,
      aoIntensity: 2,
      emissiveBrightness: 0,
      darkness: 0.5,
    })
  }, [geometry, aoTexture, onReady])

  return (
    <Suspense fallback={null}>
      <mesh
        name="eden"
        geometry={geometry}
        material={material}
        castShadow
        receiveShadow
        position={[-50, -70, 0]}
        outline={0.04}
      />
      <mesh geometry={grass.geometry} material={grass.material} position={[-50, -70, 0]} />
      <BakeShadows />
    </Suspense>
  )
}

export default EdenMesh
