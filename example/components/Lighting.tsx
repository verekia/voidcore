import { useEffect, useMemo, useRef, useState, useLayoutEffect } from 'react'

import { bakeGIProbes, GIProbeHelper, useGIProbes, type GIProbeGrid } from 'voidcore'

import type { DirectionalLight, Geometry } from 'voidcore'

const EDEN_POSITION: [number, number, number] = [-50, -70, 0]

const GIProbeDebug = ({ grid, bakeKey }: { grid: GIProbeGrid; bakeKey?: unknown }) => {
  const helper = useMemo(() => new GIProbeHelper(grid, { radius: 0.4 }), [grid, bakeKey])
  return <primitive object={helper.mesh} />
}

const Lighting = ({
  debugGIProbes = false,
  giEnabled = true,
  edenGeometry,
}: {
  debugGIProbes?: boolean
  giEnabled?: boolean
  edenGeometry?: Geometry
}) => {
  const lightRef = useRef<DirectionalLight>(null)

  const grid = useGIProbes({
    boundsMin: [-55, -75, -2],
    boundsMax: [55, 85, 25],
    resolution: [22, 32, 4],
    intensity: 1.0,
  })

  // Toggle GI on/off by setting intensity (read each frame by the renderer)
  useLayoutEffect(() => {
    grid.intensity = giEnabled ? 1.0 : 0.0
  }, [grid, giEnabled])

  // Bake probes from actual scene geometry when it becomes available
  const [bakeCount, setBakeCount] = useState(0)
  useEffect(() => {
    if (!edenGeometry) return
    bakeGIProbes(grid, [{ geometry: edenGeometry, position: EDEN_POSITION }], {
      maxDistance: 15,
      skyColor: [0.4, 0.6, 0.9],
      skyIntensity: 0.15,
    })
    setBakeCount(c => c + 1)
  }, [grid, edenGeometry])

  return (
    <>
      <ambientLight args={[{ color: [1, 1, 1], intensity: 0.8 }]} />
      <directionalLight
        ref={lightRef}
        args={[{ color: [1, 1, 1], intensity: 0.5 }]}
        position={[10, 40, 100]}
        castShadow
        shadowMapSize={130}
        shadowNear={-20}
        shadowFar={25}
        shadowBias={0}
        shadowSlopeBias={0}
      />
      {debugGIProbes && bakeCount > 0 && <GIProbeDebug grid={grid} bakeKey={bakeCount} />}
    </>
  )
}

export default Lighting
