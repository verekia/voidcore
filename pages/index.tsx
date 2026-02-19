import { Suspense, useEffect, useMemo, useRef, useState } from 'react'

import {
  OrbitControls,
  Raycaster,
  Skeleton,
  AnimationMixer,
  Mesh,
  LambertMaterial,
  Group,
  Node,
  quatFromAxisAngle,
} from '../src/index.ts'
import { Canvas, Html, useEngine, useFrame, useGLTF } from '../src/react/index.ts'

import type { Engine } from '../src/index.ts'
import type { AnimationClip, AnimationAction } from '../src/index.ts'

// ─── Asset URLs ──────────────────────────────────────────────────────────────

const staticBundleSrc = new URL('../public/static-bundle.glb', import.meta.url).href
const playerBundleSrc = new URL('../public/player-bundle.glb', import.meta.url).href

// ─── Preload GLBs in parallel (avoids Suspense waterfall) ────────────────────

useGLTF.preload(staticBundleSrc, { draco: { decoderPath: '/draco-1.5.7/' } })
useGLTF.preload(playerBundleSrc, { draco: { decoderPath: '/draco-1.5.7/' } })

// ─── Configuration ──────────────────────────────────────────────────────────

const CHARACTER_COUNT = 800
const GRID_SPACING = 5
const SPHERE_RADIUS = 300
const SPHERE_CENTER_Z = 10 - SPHERE_RADIUS

// ─── Sun ────────────────────────────────────────────────────────────────────

const Sun = () => (
  <directionalLight args={[{ color: [1, 0.95, 0.9], intensity: 1.2 }]} position={[30, 30, 50]} castShadow />
)

// ─── Camera Controls ────────────────────────────────────────────────────────

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

// ─── Sphere Terrain ─────────────────────────────────────────────────────────

const SphereTerrain = () => (
  <mesh position={[0, 0, SPHERE_CENTER_Z]} castShadow={false}>
    <sphereGeometry args={[{ radius: SPHERE_RADIUS, widthSegments: 64, heightSegments: 64 }]} />
    <lambertMaterial args={[{ color: [0.12, 0.14, 0.18] }]} />
  </mesh>
)

// ─── Rotating Cube ──────────────────────────────────────────────────────────

const RotatingCube = () => {
  const meshRef = useRef<any>(null)
  const cubeAxis = useMemo(() => new Float32Array([0, 0, 1]), [])

  useFrame(({ elapsed }) => {
    const mesh = meshRef.current
    if (!mesh) return
    quatFromAxisAngle(mesh.rotation, cubeAxis, elapsed)
    mesh._dirtyLocal = true
  })

  return (
    <mesh ref={meshRef} position={[80, 0, 60]} scale={5} castShadow>
      <boxGeometry args={[{ width: 10, height: 10, depth: 10 }]} />
      <lambertMaterial args={[{ color: [0.2, 0.6, 1.0] }]} />
      <Html center>
        <div
          style={{
            color: '#fff',
            font: 'bold 16px monospace',
            background: 'rgba(0,0,0,0.6)',
            padding: '4px 10px',
            borderRadius: '4px',
            whiteSpace: 'nowrap',
          }}
        >
          Big Cube
        </div>
      </Html>
    </mesh>
  )
}

// ─── Transparent Cube ───────────────────────────────────────────────────────

const TRANSPARENT_COLORS: [number, number, number][] = [
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

const TRANSPARENT_RING_RADIUS = 30
const TRANSPARENT_Z = 25
const TRANSPARENT_CUBE_SIZE = 10

const TransparentCube = ({ index }: { index: number }) => {
  const meshRef = useRef<any>(null)
  const angle = (index / 10) * Math.PI * 2
  const position = useMemo<[number, number, number]>(
    () => [Math.cos(angle) * TRANSPARENT_RING_RADIUS, Math.sin(angle) * TRANSPARENT_RING_RADIUS, TRANSPARENT_Z],
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
    mesh._dirtyLocal = true
  })

  return (
    <mesh ref={meshRef} position={position}>
      <boxGeometry
        args={[{ width: TRANSPARENT_CUBE_SIZE, height: TRANSPARENT_CUBE_SIZE, depth: TRANSPARENT_CUBE_SIZE }]}
      />
      <lambertMaterial args={[{ color: TRANSPARENT_COLORS[index]!, opacity: 0.4 }]} />
    </mesh>
  )
}

const TransparentCubes = ({ count }: { count: number }) => (
  <>
    {Array.from({ length: count }, (_, i) => (
      <TransparentCube key={i} index={i} />
    ))}
  </>
)

// ─── Characters (imperative for performance) ────────────────────────────────

const CLIP_DURATION = 2
const CROSSFADE_DURATION = 0.3
const ORBIT_RADIUS = 5
const ORBIT_SPEED = 3

const Characters = ({ count }: { count: number }) => {
  const { scene } = useEngine()
  const groupRef = useRef<any>(null)

  const megaxeGltf = useGLTF(staticBundleSrc, { draco: { decoderPath: '/draco-1.5.7/' } })
  const playerGltf = useGLTF(playerBundleSrc, { draco: { decoderPath: '/draco-1.5.7/' } })

  // Imperative state stored in ref to avoid re-renders
  const stateRef = useRef<{
    roots: Node[]
    mixers: AnimationMixer[]
    charActions: AnimationAction[][]
    charCurrentClip: number[]
    charNextSwitch: number[]
    charOriginX: number[]
    charOriginY: number[]
    charOrbitAngle: number[]
  } | null>(null)

  useEffect(() => {
    const group = groupRef.current
    if (!group || stateRef.current) return

    // Find megaxe template mesh
    let megaxeTemplate: Mesh | null = null
    for (const mesh of megaxeGltf.meshes) {
      if (mesh.name.toLowerCase().includes('megaxe')) {
        megaxeTemplate = mesh
        break
      }
    }

    const megaxeMaterial = new LambertMaterial({
      palette: [
        { color: [0.95, 0.93, 0.9] },
        { color: [0.08, 0.08, 0.1] },
        { color: [0, 0.9, 0.85], emissive: [0, 1, 0.9], emissiveIntensity: 2.5 },
      ],
    })

    // Gather animation clips
    const refSkeleton = playerGltf.skeletons[0]!
    const playerClips: AnimationClip[] = []
    for (const name of ['SlashRight', 'Jump', 'Run']) {
      const clip = playerGltf.animations.find(a => a.name === name)
      if (clip) playerClips.push(clip)
    }
    if (playerClips.length === 0) {
      playerClips.push(...playerGltf.animations.slice(0, 3))
    }

    // Clone helper
    const cloneNodeTree = (source: Node): { root: Node; nodeMap: Map<Node, Node> } => {
      const nodeMap = new Map<Node, Node>()
      const cloneNode = (src: Node): Node => {
        let clone: Node
        if (src instanceof Mesh && src.name === 'Body') {
          clone = new Mesh(src.geometry, src.material)
        } else {
          clone = new Group()
        }
        clone.name = src.name
        clone.visible = src.visible
        clone.frustumCulled = src.frustumCulled
        clone.position[0] = src.position[0]!
        clone.position[1] = src.position[1]!
        clone.position[2] = src.position[2]!
        clone.rotation[0] = src.rotation[0]!
        clone.rotation[1] = src.rotation[1]!
        clone.rotation[2] = src.rotation[2]!
        clone.rotation[3] = src.rotation[3]!
        clone.scale[0] = src.scale[0]!
        clone.scale[1] = src.scale[1]!
        clone.scale[2] = src.scale[2]!
        clone._dirtyLocal = true
        nodeMap.set(src, clone)
        for (const child of src.children) {
          clone.add(cloneNode(child))
        }
        return clone
      }
      return { root: cloneNode(source), nodeMap }
    }

    // Spawn characters
    const cols = Math.ceil(Math.sqrt(count))
    const totalCycle = playerClips.length * CLIP_DURATION

    const roots: Node[] = []
    const mixers: AnimationMixer[] = []
    const charActions: AnimationAction[][] = []
    const charCurrentClip: number[] = []
    const charNextSwitch: number[] = []
    const charOriginX: number[] = []
    const charOriginY: number[] = []
    const charOrbitAngle: number[] = []

    for (let i = 0; i < count; i++) {
      const { root, nodeMap } = cloneNodeTree(playerGltf.scene)
      const clonedBones = refSkeleton.bones.map(b => nodeMap.get(b)!)
      const skeleton = new Skeleton(clonedBones, refSkeleton.boneInverseBindMatrices)

      for (const origMesh of playerGltf.meshes) {
        const clonedNode = nodeMap.get(origMesh)
        if (clonedNode && clonedNode instanceof Mesh) {
          clonedNode.skeleton = skeleton
        }
      }

      const col = i % cols
      const row = Math.floor(i / cols)
      root.position[0] = (col - (cols - 1) / 2) * GRID_SPACING
      root.position[1] = (row - (cols - 1) / 2) * GRID_SPACING
      root._dirtyLocal = true

      if (megaxeTemplate) {
        const handBone = skeleton.getBone('Hand.R')
        if (handBone) {
          const axeClone = new Mesh(megaxeTemplate.geometry, megaxeMaterial)
          axeClone.rotation[0] = 0
          axeClone.rotation[1] = 0
          axeClone.rotation[2] = 1
          axeClone.rotation[3] = 0
          axeClone._dirtyLocal = true
          handBone.add(axeClone)
        }
      }

      roots.push(root)
      group.add(root)

      if (playerClips.length > 0) {
        const mixer = new AnimationMixer(skeleton)
        const actions: AnimationAction[] = []
        for (const clip of playerClips) {
          actions.push(mixer.clipAction(clip))
        }

        const offset = Math.random() * totalCycle
        const startClip = Math.floor(offset / CLIP_DURATION) % playerClips.length
        const timeIntoClip = offset % CLIP_DURATION

        actions[startClip]!.play()
        actions[startClip]!.weight = 1
        actions[startClip]!.time = timeIntoClip

        mixers.push(mixer)
        charActions.push(actions)
        charCurrentClip.push(startClip)
        charNextSwitch.push(CLIP_DURATION - timeIntoClip)
      }
    }

    // Raycast characters onto sphere
    scene.updateGraph()
    const raycaster = new Raycaster()
    const rayDir: [number, number, number] = [0, 0, -1]
    const RAY_START_Z = 100

    // Find the sphere mesh in the scene
    let sphereMesh: Mesh | null = null
    scene.traverse((node: Node) => {
      if (node.type === 'mesh' && (node as Mesh).geometry && !sphereMesh) {
        const m = node as Mesh
        // The sphere is at SPHERE_CENTER_Z and has the large radius
        if (Math.abs(m.position[2]! - SPHERE_CENTER_Z) < 1) {
          sphereMesh = m
        }
      }
    })

    for (const root of roots) {
      const x = root.position[0]!
      const y = root.position[1]!

      if (sphereMesh) {
        raycaster.set(new Float32Array([x, y, RAY_START_Z]), new Float32Array(rayDir))
        const hits = raycaster.intersectObject(sphereMesh)
        if (hits.length > 0) {
          root.position[2] = hits[0]!.point[2]!
        }
      }

      charOriginX.push(x)
      charOriginY.push(y)
      const a = Math.random() * Math.PI * 2
      charOrbitAngle.push(a)
      root.position[0] = x + Math.cos(a) * ORBIT_RADIUS
      root.position[1] = y + Math.sin(a) * ORBIT_RADIUS
      root._dirtyLocal = true
    }

    stateRef.current = {
      roots,
      mixers,
      charActions,
      charCurrentClip,
      charNextSwitch,
      charOriginX,
      charOriginY,
      charOrbitAngle,
    }
  }, [megaxeGltf, playerGltf, count, scene])

  useFrame(({ dt, elapsed }) => {
    const state = stateRef.current
    if (!state) return

    const { roots, mixers, charActions, charCurrentClip, charNextSwitch, charOriginX, charOriginY, charOrbitAngle } =
      state

    for (let i = 0; i < roots.length; i++) {
      charOrbitAngle[i] = charOrbitAngle[i]! + ORBIT_SPEED * dt
      const a = charOrbitAngle[i]!
      roots[i]!.position[0] = charOriginX[i]! + Math.cos(a) * ORBIT_RADIUS
      roots[i]!.position[1] = charOriginY[i]! + Math.sin(a) * ORBIT_RADIUS
      roots[i]!._dirtyLocal = true
    }

    for (let i = 0; i < mixers.length; i++) {
      if (elapsed >= charNextSwitch[i]!) {
        const actions = charActions[i]!
        const cur = charCurrentClip[i]!
        const next = (cur + 1) % actions.length
        actions[cur]!.crossFadeTo(actions[next]!, CROSSFADE_DURATION)
        charCurrentClip[i] = next
        charNextSwitch[i] = elapsed + CLIP_DURATION
      }
      mixers[i]!.update(dt)
    }
  })

  return <group ref={groupRef} />
}

// ─── Stats Overlay ──────────────────────────────────────────────────────────

const StatsOverlay = ({ engine }: { engine: Engine | null }) => {
  const [statsText, setStatsText] = useState('')

  useEffect(() => {
    if (!engine) return
    const interval = setInterval(() => {
      const stats = engine.getStats()
      const shadowSuffix = stats.shadowDrawCalls > 0 ? ` (+${stats.shadowDrawCalls} CSM)` : ''
      const dpr = Math.min(window.devicePixelRatio, engine.maxDpr)
      setStatsText(
        `${engine.backend.toUpperCase()} | ${Math.round(stats.fps)} FPS | DPR: ${dpr.toFixed(2)} | Draw calls: ${stats.drawCalls}${shadowSuffix}`,
      )
    }, 500)
    return () => clearInterval(interval)
  }, [engine])

  if (!statsText) return null

  return (
    <div
      style={{
        position: 'fixed',
        top: 10,
        left: 10,
        color: '#fff',
        font: '14px monospace',
        background: 'rgba(0,0,0,0.6)',
        padding: '8px 12px',
        borderRadius: '4px',
        zIndex: 1000,
        pointerEvents: 'none',
      }}
    >
      {statsText}
    </div>
  )
}

// ─── Backend Switch ─────────────────────────────────────────────────────────

const BackendSwitch = ({ engine }: { engine: Engine | null }) => {
  if (!engine) return null

  const isWebGPU = engine.backend === 'webgpu'

  return (
    <a
      href={isWebGPU ? '?webgl=1' : '/'}
      style={{
        position: 'fixed',
        top: 10,
        right: 10,
        color: '#fff',
        font: '14px monospace',
        background: 'rgba(0,0,0,0.6)',
        padding: '8px 12px',
        borderRadius: '4px',
        zIndex: 1000,
        textDecoration: 'underline',
        cursor: 'pointer',
      }}
    >
      Switch to {isWebGPU ? 'WebGL2' : 'WebGPU'}
    </a>
  )
}

// ─── Page ───────────────────────────────────────────────────────────────────

const IndexPage = () => {
  const [engine, setEngine] = useState<Engine | null>(null)

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
        shadows
        bloom={{ intensity: 0.8 }}
        antialias
        maxFps={60}
        backend={forceWebGL ? 'webgl2' : 'auto'}
        camera={{ fov: 55, near: 0.1, far: 500, position: [0, -230, 60] }}
        ambientLight={{ color: [0.5, 0.5, 0.6], intensity: 0.4 }}
        onCreated={({ engine }) => setEngine(engine)}
        style={{ width: '100vw', height: '100vh', position: 'fixed', top: 0, left: 0 }}
      >
        <Suspense fallback={null}>
          <Sun />
          <CameraControls />
          <SphereTerrain />
          <Characters count={CHARACTER_COUNT} />
          <RotatingCube />
          <TransparentCubes count={10} />
        </Suspense>
      </Canvas>
      <StatsOverlay engine={engine} />
      <BackendSwitch engine={engine} />
    </>
  )
}

export default IndexPage
