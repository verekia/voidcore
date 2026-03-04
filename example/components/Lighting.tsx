import { useMemo, useRef } from 'react'

import { GIProbeHelper, useGIProbes, type GIProbeGrid } from 'voidcore'

import type { DirectionalLight } from 'voidcore'

// Populate GI probes with spatially-varying ambient color.
// Ground level gets green bounce from grass, upper probes get sky blue,
// and there's a warm tint from the sun direction (+Y, +Z).
const populateProbes = (grid: GIProbeGrid) => {
  const [rx, ry, rz] = grid.resolution
  for (let iz = 0; iz < rz; iz++) {
    for (let iy = 0; iy < ry; iy++) {
      for (let ix = 0; ix < rx; ix++) {
        const tz = iz / Math.max(rz - 1, 1) // 0 = ground, 1 = top

        // Base ambient: blend from warm ground to cool sky
        const r = 0.08 + tz * 0.04
        const g = 0.1 + tz * 0.06
        const b = 0.06 + tz * 0.12

        // Green bounce from grass at ground level
        const grassBounce = Math.max(0, 1 - tz * 2.5) * 0.06

        grid.setProbe(ix, iy, iz, {
          color: [r, g + grassBounce, b],
          direction: [0.2, 0.5, 0.8],
          directionalColor: [0.06, 0.04, 0.02],
        })
      }
    }
  }
}

const GIProbeDebug = ({ grid }: { grid: GIProbeGrid }) => {
  const helper = useMemo(() => new GIProbeHelper(grid, { radius: 0.6 }), [grid])
  return <primitive object={helper.mesh} />
}

const Lighting = ({ debugGIProbes = false }: { debugGIProbes?: boolean }) => {
  const lightRef = useRef<DirectionalLight>(null)

  const grid = useGIProbes({
    boundsMin: [-55, -75, -2],
    boundsMax: [55, 85, 25],
    resolution: [12, 17, 4],
    intensity: 1.0,
    populate: populateProbes,
  })

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
      {debugGIProbes && <GIProbeDebug grid={grid} />}
    </>
  )
}

export default Lighting
