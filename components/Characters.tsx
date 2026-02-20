import { useEffect, useRef } from 'react'

import { Raycaster, Skeleton, AnimationMixer, Mesh, LambertMaterial, Group, Node } from '../src/index'
import { useEngine, useFrame, useGLTF } from '../src/react/index'
import { staticBundleSrc, playerBundleSrc } from './assets'

import type { AnimationClip, AnimationAction } from '../src/index'

const CLIP_DURATION = 2
const CROSSFADE_DURATION = 0.3
const ORBIT_RADIUS = 5
const ORBIT_SPEED = 3
const GRID_SPACING = 5

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
        clone.setPosition(src.position[0]!, src.position[1]!, src.position[2]!)
        clone.setRotation(src.rotation[0]!, src.rotation[1]!, src.rotation[2]!, src.rotation[3]!)
        clone.setScale(src.scale[0]!, src.scale[1]!, src.scale[2]!)
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
      root.setPosition((col - (cols - 1) / 2) * GRID_SPACING, (row - (cols - 1) / 2) * GRID_SPACING, 0)

      if (megaxeTemplate) {
        const handBone = skeleton.getBone('Hand.R')
        if (handBone) {
          const axeClone = new Mesh(megaxeTemplate.geometry, megaxeMaterial)
          axeClone.setRotation(0, 0, 1, 0)
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

    // Raycast characters onto Eden mesh
    scene.updateGraph()
    const raycaster = new Raycaster()
    const rayDir: [number, number, number] = [0, 0, -1]
    const RAY_START_Z = 200

    // Find the Eden mesh in the scene
    let edenMesh: Mesh | null = null
    scene.traverse((node: Node) => {
      if (node.type === 'mesh' && node.name === 'eden' && !edenMesh) {
        edenMesh = node as Mesh
      }
    })

    for (const root of roots) {
      const x = root.position[0]!
      const y = root.position[1]!

      let z = -50
      if (edenMesh) {
        raycaster.set(new Float32Array([x, y, RAY_START_Z]), new Float32Array(rayDir))
        const hits = raycaster.intersectObject(edenMesh)
        if (hits.length > 0) {
          z = hits[0]!.point[2]!
        }
      }

      charOriginX.push(x)
      charOriginY.push(y)
      const a = Math.random() * Math.PI * 2
      charOrbitAngle.push(a)
      root.setPosition(x + Math.cos(a) * ORBIT_RADIUS, y + Math.sin(a) * ORBIT_RADIUS, z)
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
      roots[i]!.setPositionX(charOriginX[i]! + Math.cos(a) * ORBIT_RADIUS)
      roots[i]!.setPositionY(charOriginY[i]! + Math.sin(a) * ORBIT_RADIUS)
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

export default Characters
