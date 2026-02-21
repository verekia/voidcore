import { useEffect, useRef } from 'react'

import { OrbitControls, useEngine, useFrame } from 'voidcore'

const CameraControls = () => {
  const { camera, canvas } = useEngine()
  const controlsRef = useRef<OrbitControls | null>(null)

  useEffect(() => {
    const controls = new OrbitControls(camera, canvas, {
      target: [0, 0, 1.5],
      dampingFactor: 0.08,
      minDistance: 2,
      maxDistance: 230,
    })
    controlsRef.current = controls
    return () => controls.dispose()
  }, [camera, canvas])

  useFrame(({ dt }) => {
    controlsRef.current?.update(dt)
  })

  return null
}

export default CameraControls
