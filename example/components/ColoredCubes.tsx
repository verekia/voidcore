import { useMemo, useRef } from 'react'

import { type CustomShader, quatFromAxisAngle, useFrame } from 'voidcore'

const CUBE_COLORS: [number, number, number][] = [
  [1.0, 0.2, 0.2],
  [0.2, 1.0, 0.3],
  [0.2, 0.4, 1.0],
  [1.0, 0.8, 0.1],
  [1.0, 0.4, 0.0],
  [0.8, 0.2, 1.0],
  [0.0, 0.9, 0.9],
  [1.0, 0.3, 0.6],
  [0.4, 1.0, 0.7],
  [1.0, 1.0, 1.0],
]

const COLORED_RING_RADIUS = 30
const COLORED_Z = 25
const COLORED_CUBE_SIZE = 10

const ColoredCube = ({ index }: { index: number }) => {
  const meshRef = useRef<any>(null)
  const angle = (index / 10) * Math.PI * 2
  const position = useMemo<[number, number, number]>(
    () => [Math.cos(angle) * COLORED_RING_RADIUS, Math.sin(angle) * COLORED_RING_RADIUS, COLORED_Z],
    [angle],
  )
  const { axis, speed } = useMemo(() => {
    const ax = Math.random() - 0.5
    const ay = Math.random() - 0.5
    const az = Math.random() - 0.5
    const len = Math.sqrt(ax * ax + ay * ay + az * az)
    return {
      axis: new Float32Array([ax / len, ay / len, az / len]),
      speed: 1.5 + Math.random() * 2.0,
    }
  }, [])

  useFrame(({ elapsed }) => {
    const mesh = meshRef.current
    if (!mesh) return
    quatFromAxisAngle(mesh.rotation, axis, elapsed * speed)
    mesh.markTransformDirty()
  })

  return (
    <mesh ref={meshRef} position={position} castShadow={false}>
      <boxGeometry args={[{ width: COLORED_CUBE_SIZE, height: COLORED_CUBE_SIZE, depth: COLORED_CUBE_SIZE }]} />
      <lambertMaterial args={[{ color: CUBE_COLORS[index]!, opacity: 0.7, transparent: true }]} />
    </mesh>
  )
}

const ringShader = {
  uniforms: { time: 0 },
  fragmentWGSL: /* wgsl */ `
    let centered = in.uv - vec2<f32>(0.5, 0.5);
    let dist = length(centered);
    let ring = smoothstep(0.35, 0.36, dist) * (1.0 - smoothstep(0.46, 0.47, dist));
    finalColor = vec3<f32>(1.0, 0.15, 0.1);
    alpha = ring * (0.5 + 0.5 * sin(uniforms.time * 3.0));
  `,
  fragmentGLSL: /* glsl */ `
    vec2 centered = v_uv - vec2(0.5);
    float dist = length(centered);
    float ring = smoothstep(0.35, 0.36, dist) * (1.0 - smoothstep(0.46, 0.47, dist));
    finalColor = vec3(1.0, 0.15, 0.1);
    alpha = ring * (0.5 + 0.5 * sin(uniforms.time * 3.0));
  `,
} satisfies CustomShader

const PulsingRingSprite = () => {
  useFrame(({ elapsed }) => {
    ringShader.uniforms.time = elapsed
  })

  return (
    <sprite position={[0, 0, COLORED_Z + 5]} scale={[20, 20, 1]}>
      <spriteMaterial args={[{ customShader: ringShader }]} />
    </sprite>
  )
}

const ColoredCubes = ({ count }: { count: number }) => (
  <>
    {Array.from({ length: count }, (_, i) => (
      <ColoredCube key={i} index={i} />
    ))}
    <PulsingRingSprite />
  </>
)

export default ColoredCubes
