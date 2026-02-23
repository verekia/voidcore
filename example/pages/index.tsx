import { Suspense, useCallback, useEffect, useState } from 'react'

import { Canvas } from 'voidcore'

import BackendSwitch from '../components/BackendSwitch'
import CameraControls from '../components/CameraControls'
import Characters from '../components/Characters'
import ColoredCubes from '../components/ColoredCubes'
import EdenMesh from '../components/EdenMesh'
import FloatSwitch from '../components/FloatSwitch'
import Lighting from '../components/Lighting'
import RotatingCube from '../components/RotatingCube'
import StatsOverlay from '../components/StatsOverlay'

import type { Engine } from 'voidcore'

// ─── Configuration ──────────────────────────────────────────────────────────

const CHARACTER_COUNT = 400

// ─── Page ───────────────────────────────────────────────────────────────────

const IndexPage = () => {
  const [engine, setEngine] = useState<Engine | null>(null)
  const [edenReady, setEdenReady] = useState(false)
  const onEdenReady = useCallback(() => setEdenReady(true), [])

  useEffect(() => {
    const fn = async () => {
      const eruda = await import('eruda')
      eruda.default.init()
    }
    fn()
  }, [])

  const params = typeof window !== 'undefined' ? new URLSearchParams(window.location.search) : null
  const forceWebGL = params?.has('webgl')
  const forceF32 = params?.has('f32')

  return (
    <>
      <Canvas
        shadows={{ resolution: 1024 }}
        bloom={{ levels: 3 }}
        antialias
        maxFps={60}
        backend={forceWebGL ? 'webgl2' : 'auto'}
        floatPrecision={forceF32 ? 'float32' : 'auto'}
        camera={{ fov: 55, near: 0.1, far: 500, position: [0, -230, 60] }}
        onCreated={({ engine }) => setEngine(engine)}
        className="fixed top-0 left-0 h-screen w-screen"
      >
        <mesh>
          <sphereGeometry args={[{ radius: 400 }]} />
          <basicMaterial color={[0.2, 0.5, 1]} side="back" />
        </mesh>
        <Lighting />
        <CameraControls />
        <RotatingCube />
        <ColoredCubes count={10} />
        <Suspense fallback={null}>
          <EdenMesh onReady={onEdenReady} />
        </Suspense>
        {edenReady && (
          <Suspense fallback={null}>
            <Characters count={CHARACTER_COUNT} />
          </Suspense>
        )}
      </Canvas>
      <StatsOverlay engine={engine} />
      <FloatSwitch engine={engine} />
      <BackendSwitch engine={engine} />
    </>
  )
}

export default IndexPage
