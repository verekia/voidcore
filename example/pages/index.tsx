import { Suspense, useCallback, useEffect, useState } from 'react'

import { Canvas } from 'voidcore'

import BackendSwitch from '../components/BackendSwitch'
import CameraControls from '../components/CameraControls'
import Characters from '../components/Characters'
import ColoredCubes from '../components/ColoredCubes'
import EdenMesh from '../components/EdenMesh'
import Lighting from '../components/Lighting'
import RotatingCube from '../components/RotatingCube'
import StatsOverlay from '../components/StatsOverlay'

import type { Engine, Geometry } from 'voidcore'

// ─── Configuration ──────────────────────────────────────────────────────────

const CHARACTER_COUNT = 400

// ─── Page ───────────────────────────────────────────────────────────────────

const IndexPage = () => {
  const [engine, setEngine] = useState<Engine | null>(null)
  const [edenReady, setEdenReady] = useState(false)
  const [edenGeometry, setEdenGeometry] = useState<Geometry | undefined>()
  const [debugGIProbes, setDebugGIProbes] = useState(false)
  const [giEnabled, setGIEnabled] = useState(true)
  const onEdenReady = useCallback(() => setEdenReady(true), [])
  const onEdenGeometry = useCallback((geo: Geometry) => setEdenGeometry(geo), [])

  useEffect(() => {
    const fn = async () => {
      const eruda = await import('eruda')
      eruda.default.init()
    }
    fn()
  }, [])

  const params = typeof window !== 'undefined' ? new URLSearchParams(window.location.search) : null
  const forceWebGL = params?.has('webgl')

  return (
    <>
      <Canvas
        shadows={{ resolution: 1024 }}
        bloom={{ levels: 3 }}
        antialias
        maxFps={60}
        backend={forceWebGL ? 'webgl2' : 'auto'}
        camera={{ fov: 55, near: 0.1, far: 2000, position: [0, -230, 60] }}
        onCreated={({ engine }) => setEngine(engine)}
        className="fixed top-0 left-0 h-screen w-screen"
      >
        <mesh>
          <sphereGeometry args={[{ radius: 300 }]} />
          <basicMaterial color={[0.2, 0.5, 1]} side="back" />
        </mesh>
        <Lighting debugGIProbes={debugGIProbes} giEnabled={giEnabled} edenGeometry={edenGeometry} />
        <CameraControls />
        <RotatingCube />
        <ColoredCubes count={10} />
        <Suspense fallback={null}>
          <EdenMesh onReady={onEdenReady} onGeometry={onEdenGeometry} />
        </Suspense>
        {edenReady && (
          <Suspense fallback={null}>
            <Characters count={CHARACTER_COUNT} />
          </Suspense>
        )}
      </Canvas>
      <StatsOverlay engine={engine} />
      <BackendSwitch engine={engine} />
      <button
        onClick={() => setGIEnabled(v => !v)}
        className={`fixed top-12 right-2.5 z-[1000] cursor-pointer rounded px-3 py-2 font-mono text-sm ${
          giEnabled ? 'bg-green-600/80 text-white' : 'bg-black/60 text-white'
        }`}
      >
        GI {giEnabled ? 'ON' : 'OFF'}
      </button>
      <button
        onClick={() => setDebugGIProbes(v => !v)}
        className={`fixed top-24 right-2.5 z-[1000] cursor-pointer rounded px-3 py-2 font-mono text-sm ${
          debugGIProbes ? 'bg-yellow-600/80 text-white' : 'bg-black/60 text-white'
        }`}
      >
        GI Probes {debugGIProbes ? 'ON' : 'OFF'}
      </button>
    </>
  )
}

export default IndexPage
