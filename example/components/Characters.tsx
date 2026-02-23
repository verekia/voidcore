import { memo, useRef, useMemo } from 'react'

import {
  Raycaster,
  cloneScene,
  LambertMaterial,
  Mesh,
  useEngine,
  useFrame,
  useGLTF,
  useAnimations,
  bakePalette,
  mergeStaticIntoSkinned,
  mat4Create,
  mat4Compose,
  VEC3_ZERO,
  VEC3_ONE,
} from 'voidcore'

import { staticBundleSrc, playerBundleSrc } from './assets'

import type { AnimationClip, Geometry, GLTFResult, MeshOutline, PaletteEntry } from 'voidcore'

const CLIP_DURATION = 2
const CROSSFADE_DURATION = 0.3
const ORBIT_RADIUS = 5
const GRID_SPACING = 3

const megaxePalette: PaletteEntry[] = [
  { color: [0.95, 0.93, 0.9] },
  { color: [0.08, 0.08, 0.1] },
  { color: [1, 1, 1], emissive: [0, 1, 0.9], emissiveIntensity: 1.5 },
]

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

    const { root, skeleton, meshes } = useMemo(() => {
      const { root, skeletons } = cloneScene(playerGltf.scene, playerGltf.skeletons, {
        meshFilter: m => m.name === 'Body',
      })
      const skeleton = skeletons[0]!
      const meshes: Mesh[] = []

      root.traverse(node => {
        if (node instanceof Mesh) {
          node.outline = { thickness: 0.03, color: [0, 0, 0] }
          meshes.push(node)
        }
      })

      if (megaxeGeometry && meshes.length > 0) {
        const bodyMesh = meshes[0]!
        const bodyPalette = bodyMesh.material.palette
        if (bodyPalette) {
          bodyMesh.geometry = bakePalette(bodyMesh.geometry, bodyPalette)
        }
        const bakedAxe = bakePalette(megaxeGeometry, megaxePalette)
        const handBone = skeleton.getBone('Hand.R')
        if (handBone) {
          const handBoneIndex = skeleton.bones.indexOf(handBone)
          if (handBoneIndex >= 0) {
            const ibm = skeleton.boneInverseBindMatrices[handBoneIndex]!
            const axeLocal = mat4Compose(mat4Create(), VEC3_ZERO, new Float32Array([0, 0, 1, 0]), VEC3_ONE)
            bodyMesh.geometry = mergeStaticIntoSkinned(bodyMesh.geometry, bakedAxe, handBoneIndex, ibm, axeLocal)
            bodyMesh.material = new LambertMaterial({ emissiveBrightness: 0.3 })
          }
        }
      }

      return { root, skeleton, meshes }
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
      flashFramesLeft: 0,
      nextFlash: Math.random() * 3 + 1,
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
      root.setScale(0.6)

      // Random red outline flash
      if (s.flashFramesLeft <= 0 && elapsed >= s.nextFlash) {
        s.flashFramesLeft = 10
        s.nextFlash = elapsed + Math.random() * 3 + 1
      }
      const flashColor: [number, number, number] = s.flashFramesLeft > 0 ? [1, 0, 0] : [0, 0, 0]
      if (s.flashFramesLeft > 0) s.flashFramesLeft--
      for (const m of meshes) {
        ;(m.outline as MeshOutline).color = flashColor
      }

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
  const megaxeGltf = useGLTF(staticBundleSrc)
  const playerGltf = useGLTF(playerBundleSrc)

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
