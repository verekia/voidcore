import { memo, useRef, useMemo } from 'react'

import { Raycaster, cloneScene, LambertMaterial, Mesh } from '../src/index'
import { useEngine, useFrame, useGLTF, useAnimations } from '../src/react/index'
import { staticBundleSrc, playerBundleSrc } from './assets'

import type { AnimationClip, Geometry } from '../src/index'
import type { GLTFResult } from '../src/loaders/gltf'

const CLIP_DURATION = 2
const CROSSFADE_DURATION = 0.3
const ORBIT_RADIUS = 5
const GRID_SPACING = 5

const megaxeMaterial = new LambertMaterial({
  emissiveBrightness: 0.3,
  palette: [
    { color: [0.95, 0.93, 0.9] },
    { color: [0.08, 0.08, 0.1] },
    { color: [1, 1, 1], emissive: [0, 1, 0.9], emissiveIntensity: 1.5 },
  ],
})

interface CharacterProps {
  x: number
  y: number
  megaxeGeometry: Geometry | null
  playerGltf: GLTFResult
  clips: AnimationClip[]
  startAngle: number
  startClipIndex: number
  startTimeIntoClip: number
}

const Character = memo(
  ({ x, y, megaxeGeometry, playerGltf, clips, startAngle, startClipIndex, startTimeIntoClip }: CharacterProps) => {
    const { scene } = useEngine()

    const { root, skeleton } = useMemo(() => {
      const { root, skeletons } = cloneScene(playerGltf.scene, playerGltf.skeletons, {
        meshFilter: m => m.name === 'Body',
      })
      const skeleton = skeletons[0]!

      if (megaxeGeometry) {
        const handBone = skeleton.getBone('Hand.R')
        if (handBone) {
          const axe = new Mesh(megaxeGeometry, megaxeMaterial)
          axe.outline = 0.05
          axe.setRotation(0, 0, 1, 0)
          handBone.add(axe)
        }
      }

      root.traverse(node => {
        if (node instanceof Mesh) node.outline = 0.05
      })

      return { root, skeleton }
    }, [playerGltf, megaxeGeometry])

    const { actions } = useAnimations(clips, skeleton)
    const actionList = useMemo(() => clips.map(c => actions[c.name]!), [clips, actions])

    const stateRef = useRef({
      initialized: false,
      angle: startAngle,
      z: -50,
      zResolved: false,
      currentClip: startClipIndex,
      nextSwitch: CLIP_DURATION - startTimeIntoClip,
    })

    useFrame(({ elapsed }) => {
      const s = stateRef.current

      if (!s.initialized) {
        s.initialized = true
        if (actionList[startClipIndex]) {
          actionList[startClipIndex].play()
          actionList[startClipIndex].weight = 1
          actionList[startClipIndex].time = startTimeIntoClip
        }
      }

      if (!s.zResolved) {
        s.zResolved = true
        const edenMesh = scene.getByName('eden') as Mesh | undefined
        if (edenMesh) {
          const raycaster = new Raycaster()
          raycaster.set(new Float32Array([x, y, 200]), new Float32Array([0, 0, -1]))
          const hits = raycaster.intersectObject(edenMesh)
          if (hits.length > 0) s.z = hits[0]!.point[2]!
        }
      }

      // s.angle += ORBIT_SPEED * dt
      root.setPosition(x + Math.cos(s.angle) * ORBIT_RADIUS, y + Math.sin(s.angle) * ORBIT_RADIUS, s.z)

      if (elapsed >= s.nextSwitch && actionList.length > 1) {
        const cur = s.currentClip
        const next = (cur + 1) % actionList.length
        actionList[cur]!.crossFadeTo(actionList[next]!, CROSSFADE_DURATION)
        s.currentClip = next
        s.nextSwitch = elapsed + CLIP_DURATION
      }
    })

    return <primitive object={root} />
  },
)

const Characters = ({ count }: { count: number }) => {
  const megaxeGltf = useGLTF(staticBundleSrc, { draco: { decoderPath: '/draco-1.5.7/' } })
  const playerGltf = useGLTF(playerBundleSrc, { draco: { decoderPath: '/draco-1.5.7/' } })

  const megaxeGeometry = useMemo(() => {
    for (const mesh of megaxeGltf.meshes) {
      if (mesh.name.toLowerCase().includes('megaxe')) return mesh.geometry
    }
    return null
  }, [megaxeGltf])

  const clips = useMemo(() => {
    const result: AnimationClip[] = []
    for (const name of ['SlashRight', 'Jump', 'Run']) {
      const clip = playerGltf.animations.find(a => a.name === name)
      if (clip) result.push(clip)
    }
    if (result.length === 0) result.push(...playerGltf.animations.slice(0, 3))
    return result
  }, [playerGltf])

  const characters = useMemo(() => {
    const cols = Math.ceil(Math.sqrt(count))
    const totalCycle = clips.length * CLIP_DURATION
    const result = []
    for (let i = 0; i < count; i++) {
      const col = i % cols
      const row = Math.floor(i / cols)
      const offset = Math.random() * totalCycle
      result.push({
        x: (col - (cols - 1) / 2) * GRID_SPACING,
        y: (row - (cols - 1) / 2) * GRID_SPACING,
        startAngle: Math.random() * Math.PI * 2,
        startClipIndex: Math.floor(offset / CLIP_DURATION) % Math.max(clips.length, 1),
        startTimeIntoClip: offset % CLIP_DURATION,
      })
    }
    return result
  }, [count, clips])

  return (
    <group>
      {characters.map((c, i) => (
        <Character
          key={i}
          x={c.x}
          y={c.y}
          megaxeGeometry={megaxeGeometry}
          playerGltf={playerGltf}
          clips={clips}
          startAngle={c.startAngle}
          startClipIndex={c.startClipIndex}
          startTimeIntoClip={c.startTimeIntoClip}
        />
      ))}
    </group>
  )
}

export default Characters
