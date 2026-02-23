import { memo, useRef, useMemo } from 'react'

import {
  Raycaster,
  LambertMaterial,
  Mesh,
  useEngine,
  useFrame,
  useGLTF,
  useAnimations,
  useColoredGeometry,
  useColoredStaticGeometry,
  mergeStaticIntoSkinned,
  mat4Create,
  mat4Compose,
  VEC3_ZERO,
  VEC3_ONE,
} from 'voidcore'

import { playerBundleSrc } from './assets'

import type { AnimationClip, MeshOutline, PaletteEntry } from 'voidcore'

const CLIP_DURATION = 2
const CROSSFADE_DURATION = 0.3
const ORBIT_RADIUS = 2
const ORBIT_SPEED = 1
const GRID_SPACING = 5

const bodyPalette: PaletteEntry[] = [{ color: [1, 1, 1] }, { color: [1, 1, 1] }]

const megaxePalette: PaletteEntry[] = [
  { color: [0.95, 0.93, 0.9] },
  { color: [0.08, 0.08, 0.1] },
  { color: [1, 1, 1], emissive: [0, 1, 0.9], emissiveIntensity: 1.5 },
]

interface CharacterProps {
  x: number
  y: number
  startAngle: number
  timeOffset: number
}

const characterMaterial = new LambertMaterial({ emissiveBrightness: 0.2 })
const raycaster = new Raycaster()
const rayOrigin = new Float32Array([0, 0, 200])
const rayDir = new Float32Array([0, 0, -1])

const Character = memo(({ x, y, startAngle, timeOffset }: CharacterProps) => {
  const { scene } = useEngine()

  const axeGeometry = useColoredStaticGeometry('megaxe', megaxePalette)

  const {
    root,
    mesh: bodyMesh,
    skeleton,
    animations,
  } = useGLTF(playerBundleSrc, {
    meshName: 'Body',
    clone: true,
  })

  const bakedBody = useColoredGeometry(bodyMesh.geometry, bodyPalette)

  const clips = useMemo(() => {
    const result: AnimationClip[] = []
    for (const name of ['SlashRight', 'Jump', 'Run']) {
      const clip = animations.find(a => a.name === name)
      if (clip) result.push(clip)
    }
    if (result.length === 0) result.push(...animations.slice(0, 3))
    return result
  }, [animations])

  const startClipIndex = clips.length > 0 ? Math.floor(timeOffset / CLIP_DURATION) % clips.length : 0
  const startTimeIntoClip = timeOffset % CLIP_DURATION

  useMemo(() => {
    bodyMesh.outline = { thickness: 0.03, color: [0, 0, 0] }

    const handBone = skeleton.getBone('Hand.R')
    if (handBone) {
      const handBoneIndex = skeleton.bones.indexOf(handBone)
      if (handBoneIndex >= 0) {
        const ibm = skeleton.boneInverseBindMatrices[handBoneIndex]!
        const axeLocal = mat4Compose(mat4Create(), VEC3_ZERO, new Float32Array([0, 0, 1, 0]), VEC3_ONE)
        bodyMesh.geometry = mergeStaticIntoSkinned(bakedBody, axeGeometry, handBoneIndex, ibm, axeLocal)
        bodyMesh.material = characterMaterial
      }
    }
  }, [bodyMesh, skeleton, bakedBody, axeGeometry])

  const { actions } = useAnimations(clips, skeleton)
  const actionList = useMemo(() => clips.map(c => actions[c.name]!), [clips, actions])

  const stateRef = useRef({
    initialized: false,
    angle: startAngle,
    z: -50,
    currentClip: startClipIndex,
    nextSwitch: CLIP_DURATION - startTimeIntoClip,
    flashFramesLeft: 0,
    nextFlash: Math.random() * 3 + 1,
  })

  useFrame(({ elapsed, dt }) => {
    const s = stateRef.current

    if (!s.initialized) {
      s.initialized = true
      if (actionList[startClipIndex]) {
        actionList[startClipIndex].play()
        actionList[startClipIndex].weight = 1
        actionList[startClipIndex].time = startTimeIntoClip
      }
    }

    s.angle += ORBIT_SPEED * dt

    const posX = x + Math.cos(s.angle) * ORBIT_RADIUS
    const posY = y + Math.sin(s.angle) * ORBIT_RADIUS

    const edenMesh = scene.getByName('eden') as Mesh | undefined
    if (edenMesh) {
      rayOrigin[0] = posX
      rayOrigin[1] = posY
      raycaster.set(rayOrigin, rayDir)
      const hits = raycaster.intersectObject(edenMesh)
      s.z = hits.length > 0 ? hits[0]!.point[2]! : -50
    }

    root.setPosition(posX, posY, s.z)
    root.setScale(0.6)

    // Random red outline flash
    if (s.flashFramesLeft <= 0 && elapsed >= s.nextFlash) {
      s.flashFramesLeft = 10
      s.nextFlash = elapsed + Math.random() * 3 + 1
    }
    const flashColor: [number, number, number] = s.flashFramesLeft > 0 ? [1, 0, 0] : [0, 0, 0]
    if (s.flashFramesLeft > 0) s.flashFramesLeft--
    ;(bodyMesh.outline as MeshOutline).color = flashColor

    if (elapsed >= s.nextSwitch && actionList.length > 1) {
      const cur = s.currentClip
      const next = (cur + 1) % actionList.length
      actionList[cur]!.crossFadeTo(actionList[next]!, CROSSFADE_DURATION)
      s.currentClip = next
      s.nextSwitch = elapsed + CLIP_DURATION
    }
  })

  return <primitive object={root} />
})

const Characters = ({ count }: { count: number }) => {
  const characters = useMemo(() => {
    const cols = Math.ceil(Math.sqrt(count))
    const result = []
    for (let i = 0; i < count; i++) {
      const col = i % cols
      const row = Math.floor(i / cols)
      result.push({
        x: (col - (cols - 1) / 2) * GRID_SPACING,
        y: (row - (cols - 1) / 2) * GRID_SPACING,
        startAngle: Math.random() * Math.PI * 2,
        timeOffset: Math.random() * CLIP_DURATION * 3,
      })
    }
    return result
  }, [count])

  return (
    <group>
      {characters.map((c, i) => (
        <Character key={i} x={c.x} y={c.y} startAngle={c.startAngle} timeOffset={c.timeOffset} />
      ))}
    </group>
  )
}

export default Characters
