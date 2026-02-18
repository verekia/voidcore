import {
  createEngine,
  createScene,
  createPerspectiveCamera,
  createDirectionalLight,
  createLambertMaterial,
  createOrbitControls,
  loadGLTF,
  createAnimationMixer,
  Skeleton,
  Mesh,
  Group,
  Node,
} from './index.ts'

import type { AnimationClip, AnimationAction } from './index.ts'

// ─── Configuration ──────────────────────────────────────────────────
const CHARACTER_COUNT = 1000
const GRID_SPACING = 5

export const main = async (canvas: HTMLCanvasElement) => {
  // ─── Engine ─────────────────────────────────────────────────────
  const engine = await createEngine(canvas, {
    antialias: true,
    bloom: { intensity: 0.8 },
  })

  // ─── Scene ──────────────────────────────────────────────────────
  const scene = createScene()
  scene.ambientLight = { color: [0.5, 0.5, 0.6], intensity: 0.4 }

  // ─── Directional Light ──────────────────────────────────────────
  const sun = createDirectionalLight({ color: [1, 0.95, 0.9], intensity: 1.2 })
  sun.position[0] = 30
  sun.position[1] = 30
  sun.position[2] = 50
  sun._dirtyLocal = true
  scene.add(sun)

  // ─── Camera + Orbit Controls ────────────────────────────────────
  const camera = createPerspectiveCamera({ fov: 55, near: 0.1, far: 500 })
  camera.position[0] = 0
  camera.position[1] = -170
  camera.position[2] = 60
  camera._dirtyLocal = true
  camera._dirtyWorld = true

  const controls = createOrbitControls(camera, canvas, {
    target: [0, 0, 1.5],
    dampingFactor: 0.08,
    minDistance: 2,
    maxDistance: 200,
  })

  // ─── Load assets ───────────────────────────────────────────────
  const [megaxeGltf, playerGltf] = await Promise.all([
    loadGLTF('/static-bundle.glb', engine, { draco: { decoderPath: '/draco-1.5.7/' } }),
    loadGLTF('/player-bundle.glb', engine, { draco: { decoderPath: '/draco-1.5.7/' } }),
  ])

  // Find megaxe template mesh
  let megaxeTemplate: Mesh | null = null
  for (const mesh of megaxeGltf.meshes) {
    if (mesh.name.toLowerCase().includes('megaxe')) {
      megaxeTemplate = mesh
      break
    }
  }

  const megaxeMaterial = createLambertMaterial({
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

  // ─── Clone helper ─────────────────────────────────────────────
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

  // ─── Spawn characters ─────────────────────────────────────────
  const CLIP_DURATION = 2 // seconds per clip before crossfade
  const CROSSFADE_DURATION = 0.3

  const mixers: ReturnType<typeof createAnimationMixer>[] = []
  const charActions: AnimationAction[][] = []
  const charCurrentClip: number[] = []
  const charNextSwitch: number[] = []

  const cols = Math.ceil(Math.sqrt(CHARACTER_COUNT))
  const rows = Math.ceil(CHARACTER_COUNT / cols)

  // Total cycle length for staggering offsets
  const totalCycle = playerClips.length * CLIP_DURATION

  for (let i = 0; i < CHARACTER_COUNT; i++) {
    const { root, nodeMap } = cloneNodeTree(playerGltf.scene)

    // Create skeleton with cloned bones
    const clonedBones = refSkeleton.bones.map(b => nodeMap.get(b)!)
    const skeleton = new Skeleton(clonedBones, refSkeleton.boneInverseBindMatrices)

    // Assign skeleton to skinned meshes
    for (const origMesh of playerGltf.meshes) {
      const clonedNode = nodeMap.get(origMesh)
      if (clonedNode && clonedNode instanceof Mesh) {
        clonedNode.skeleton = skeleton
      }
    }

    // Position in grid
    const col = i % cols
    const row = Math.floor(i / cols)
    root.position[0] = (col - (cols - 1) / 2) * GRID_SPACING
    root.position[1] = (row - (rows - 1) / 2) * GRID_SPACING
    root._dirtyLocal = true

    // Attach megaxe clone
    if (megaxeTemplate) {
      const handBone = skeleton.getBone('Hand.R')
      if (handBone) {
        const axeClone = new Mesh(megaxeTemplate.geometry, megaxeMaterial)
        axeClone._dirtyLocal = true
        handBone.add(axeClone)
      }
    }

    scene.add(root)

    // Animation — all characters cycle through all clips with staggered offsets
    if (playerClips.length > 0) {
      const mixer = createAnimationMixer(skeleton)
      const actions: AnimationAction[] = []
      for (const clip of playerClips) {
        actions.push(mixer.clipAction(clip))
      }

      // Stagger: random offset into the full cycle
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

  // ─── Stats overlay ────────────────────────────────────────────
  const statsDiv = document.createElement('div')
  statsDiv.style.cssText =
    'position:fixed;top:10px;left:10px;color:#fff;font:14px monospace;background:rgba(0,0,0,0.6);padding:8px 12px;border-radius:4px;z-index:1000;pointer-events:none'
  document.body.appendChild(statsDiv)

  // ─── Render Loop ──────────────────────────────────────────────
  let elapsed = 0
  engine.onFrame(dt => {
    controls.update(dt)
    elapsed += dt

    for (let i = 0; i < mixers.length; i++) {
      // Crossfade to next clip when it's time
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

    const stats = engine.getStats()
    statsDiv.textContent = `Draw calls: ${stats.drawCalls}`
  })

  engine.start(scene, camera)

  return engine
}
