import { useRef } from 'react'

import type { DirectionalLight } from 'voidcore'

const Lighting = () => {
  const lightRef = useRef<DirectionalLight>(null)

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
      {/* <DirectionalLightHelper lightRef={lightRef} /> */}
    </>
  )
}

export default Lighting
