import {
  createEngine,
  createScene,
  createPerspectiveCamera,
  createDirectionalLight,
  createLambertMaterial,
  createBasicMaterial,
  createOrbitControls,
  loadGLTF,
  createAnimationMixer,
  createSphereGeometry,
  createMesh,
  createRaycaster,
  Skeleton,
  Mesh,
  Group,
  Node,
} from './index.ts'

import type { AnimationClip, AnimationAction } from './index.ts'

// ─── Configuration ──────────────────────────────────────────────────
const CHARACTER_COUNT = 900
const GRID_SPACING = 5

export const main = async (canvas: HTMLCanvasElement) => {
  // ─── Engine ─────────────────────────────────────────────────────
  const params = new URLSearchParams(window.location.search)
  const forceWebGL = params.has('webgl')

  const engine = await createEngine(canvas, {
    antialias: true,
    bloom: { intensity: 0.8 },
    backend: forceWebGL ? 'webgl2' : 'auto',
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
  camera.position[1] = -230
  camera.position[2] = 60
  camera._dirtyLocal = true
  camera._dirtyWorld = true

  const controls = createOrbitControls(camera, canvas, {
    target: [0, 0, 1.5],
    dampingFactor: 0.08,
    minDistance: 2,
    maxDistance: 230,
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

  const roots: Node[] = []
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
        axeClone.rotation[0] = 0
        axeClone.rotation[1] = 0
        axeClone.rotation[2] = 1
        axeClone.rotation[3] = 0
        axeClone._dirtyLocal = true
        handBone.add(axeClone)
      }
    }

    roots.push(root)
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

  // ─── Sphere terrain ───────────────────────────────────────────
  // Sphere top at z=10, large enough that all characters sit on the upper part
  const SPHERE_RADIUS = 300
  const SPHERE_CENTER_Z = 10 - SPHERE_RADIUS // = -290

  const sphereGeo = createSphereGeometry({ radius: SPHERE_RADIUS, widthSegments: 64, heightSegments: 64 })
  const sphereMat = createBasicMaterial({ color: [0.12, 0.14, 0.18] })
  const sphere = createMesh(sphereGeo, sphereMat)
  sphere.position[2] = SPHERE_CENTER_Z
  sphere._dirtyLocal = true
  scene.add(sphere)

  // Update world matrices so the sphere's _worldMatrix is ready for raycasting
  scene.updateGraph()

  // Raycast each character downward from above to land it on the sphere surface
  const raycaster = createRaycaster()
  const rayDir: [number, number, number] = [0, 0, -1]
  const RAY_START_Z = 100 // well above sphere top (z=10)

  const ORBIT_RADIUS = 5
  const ORBIT_SPEED = 3 // rad/s — one full circle in ~1 second

  const charOriginX: number[] = []
  const charOriginY: number[] = []
  const charOrbitAngle: number[] = []

  for (const root of roots) {
    const x = root.position[0]!
    const y = root.position[1]!
    raycaster.set(new Float32Array([x, y, RAY_START_Z]), new Float32Array(rayDir))
    const hits = raycaster.intersectObject(sphere)
    if (hits.length > 0) {
      root.position[2] = hits[0]!.point[2]!
    }

    // Store orbit origin and assign a random starting angle so characters spread out
    charOriginX.push(x)
    charOriginY.push(y)
    const angle = Math.random() * Math.PI * 2
    charOrbitAngle.push(angle)

    // Apply initial orbit offset
    root.position[0] = x + Math.cos(angle) * ORBIT_RADIUS
    root.position[1] = y + Math.sin(angle) * ORBIT_RADIUS
    root._dirtyLocal = true
  }

  // ─── Stats overlay ────────────────────────────────────────────
  const statsDiv = document.createElement('div')
  statsDiv.style.cssText =
    'position:fixed;top:10px;left:10px;color:#fff;font:14px monospace;background:rgba(0,0,0,0.6);padding:8px 12px;border-radius:4px;z-index:1000;pointer-events:none'
  document.body.appendChild(statsDiv)

  const switchLink = document.createElement('a')
  const isWebGPU = engine.backend === 'webgpu'
  switchLink.textContent = `Switch to ${isWebGPU ? 'WebGL2' : 'WebGPU'}`
  switchLink.href = isWebGPU ? '?webgl=1' : '/'
  switchLink.style.cssText =
    'position:fixed;top:10px;right:10px;color:#fff;font:14px monospace;background:rgba(0,0,0,0.6);padding:8px 12px;border-radius:4px;z-index:1000;text-decoration:underline;cursor:pointer'
  document.body.appendChild(switchLink)

  // ─── Render Loop ──────────────────────────────────────────────
  let elapsed = 0
  engine.onFrame(dt => {
    controls.update(dt)
    elapsed += dt

    for (let i = 0; i < roots.length; i++) {
      // Circular orbit around each character's origin point
      charOrbitAngle[i]! += ORBIT_SPEED * dt
      const a = charOrbitAngle[i]!
      roots[i]!.position[0] = charOriginX[i]! + Math.cos(a) * ORBIT_RADIUS
      roots[i]!.position[1] = charOriginY[i]! + Math.sin(a) * ORBIT_RADIUS
      roots[i]!._dirtyLocal = true
    }

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
    statsDiv.textContent = `${engine.backend.toUpperCase()} | ${Math.round(stats.fps)} FPS | Draw calls: ${stats.drawCalls}`
  })

  engine.start(scene, camera)

  return engine
}
