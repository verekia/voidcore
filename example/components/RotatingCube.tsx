import { useMemo, useRef } from 'react'

import { quatFromAxisAngle, Html, useFrame } from 'voidcore'

const RotatingCube = () => {
  const meshRef = useRef<any>(null)
  const cubeAxis = useMemo(() => new Float32Array([0, 0, 1]), [])

  useFrame(({ elapsed }) => {
    const mesh = meshRef.current
    if (!mesh) return
    quatFromAxisAngle(mesh.rotation, cubeAxis, elapsed)
    mesh.markTransformDirty()
  })

  return (
    <mesh ref={meshRef} position={[80, 0, 60]} scale={5}>
      <boxGeometry args={[{ width: 10, height: 10, depth: 10 }]} />
      <lambertMaterial args={[{ color: [0.2, 0.6, 1.0] }]} />
      <Html center>
        <div className="rounded bg-black/60 px-2.5 py-1 font-mono text-base font-bold whitespace-nowrap text-white">
          Big Cube
        </div>
      </Html>
    </mesh>
  )
}

export default RotatingCube
