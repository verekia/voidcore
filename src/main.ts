import {
  Engine,
  Scene,
  PerspectiveCamera,
  DirectionalLight,
  LambertMaterial,
  OrbitControls,
  loadGLTF,
  AnimationMixer,
  SphereGeometry,
  BoxGeometry,
  Mesh,
  Raycaster,
  quatFromAxisAngle,
  Skeleton,
  Group,
  Node,
  createOverlayManager,
} from './index.ts'

import type { AnimationClip, AnimationAction } from './index.ts'

// ─── Configuration ──────────────────────────────────────────────────
const CHARACTER_COUNT = 800
const GRID_SPACING = 5

export const main = async (canvas: HTMLCanvasElement) => {
  // ─── Engine ─────────────────────────────────────────────────────
  const params = new URLSearchParams(window.location.search)
  const forceWebGL = params.has('webgl')

  const engine = await Engine.create(canvas, {
    antialias: true,
    bloom: { intensity: 0.8 },
    shadows: true,
    backend: forceWebGL ? 'webgl2' : 'auto',
  })

  // ─── Scene ──────────────────────────────────────────────────────
  const scene = new Scene()
  scene.ambientLight = { color: [0.5, 0.5, 0.6], intensity: 0.4 }

  // ─── Directional Light ──────────────────────────────────────────
  const sun = new DirectionalLight({ color: [1, 0.95, 0.9], intensity: 1.2 })
  sun.setPosition(30, 30, 50)
  scene.add(sun)

  // ─── Camera + Orbit Controls ────────────────────────────────────
  const camera = new PerspectiveCamera({ fov: 55, near: 0.1, far: 500 })
  camera.setPosition(0, -230, 60)

  const controls = new OrbitControls(camera, canvas, {
    target: [0, 0, 1.5],
    dampingFactor: 0.08,
    minDistance: 2,
    maxDistance: 230,
  })

  // ─── Load assets ───────────────────────────────────────────────
  const [megaxeGltf, playerGltf] = await Promise.all([
    loadGLTF('/static-bundle.glb', { draco: { decoderPath: '/draco-1.5.7/' } }),
    loadGLTF('/player-bundle.glb', { draco: { decoderPath: '/draco-1.5.7/' } }),
  ])

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

  // ─── Spawn characters ─────────────────────────────────────────
  const CLIP_DURATION = 2 // seconds per clip before crossfade
  const CROSSFADE_DURATION = 0.3

  const roots: Node[] = []
  const mixers: AnimationMixer[] = []
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
    root.setPosition((col - (cols - 1) / 2) * GRID_SPACING, (row - (rows - 1) / 2) * GRID_SPACING, 0)

    // Attach megaxe clone
    if (megaxeTemplate) {
      const handBone = skeleton.getBone('Hand.R')
      if (handBone) {
        const axeClone = new Mesh(megaxeTemplate.geometry, megaxeMaterial)
        axeClone.setRotation(0, 0, 1, 0)
        handBone.add(axeClone)
      }
    }

    roots.push(root)
    scene.add(root)

    // Animation — all characters cycle through all clips with staggered offsets
    if (playerClips.length > 0) {
      const mixer = new AnimationMixer(skeleton)
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

  const sphereGeo = new SphereGeometry({ radius: SPHERE_RADIUS, widthSegments: 64, heightSegments: 64 })
  const sphereMat = new LambertMaterial({ color: [0.12, 0.14, 0.18] })
  const sphere = new Mesh(sphereGeo, sphereMat)
  sphere.castShadow = false
  sphere.setPosition(0, 0, SPHERE_CENTER_Z)
  scene.add(sphere)

  // Update world matrices so the sphere's _worldMatrix is ready for raycasting
  scene.updateGraph()

  // Raycast each character downward from above to land it on the sphere surface
  const raycaster = new Raycaster()
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
    const z = hits.length > 0 ? hits[0]!.point[2]! : root.position[2]!

    // Store orbit origin and assign a random starting angle so characters spread out
    charOriginX.push(x)
    charOriginY.push(y)
    const angle = Math.random() * Math.PI * 2
    charOrbitAngle.push(angle)

    // Apply initial orbit offset
    root.setPosition(x + Math.cos(angle) * ORBIT_RADIUS, y + Math.sin(angle) * ORBIT_RADIUS, z)
  }

  // ─── Big Rotating Cube ──────────────────────────────────────────
  const cubeGeo = new BoxGeometry({ width: 10, height: 10, depth: 10 })
  const cubeMat = new LambertMaterial({ color: [0.2, 0.6, 1.0] })
  const cube = new Mesh(cubeGeo, cubeMat)
  cube.setPosition(80, 0, 60)
  cube.setScale(5)
  cube.castShadow = true
  scene.add(cube)

  // ─── HTML Overlay ─────────────────────────────────────────────────
  const overlay = createOverlayManager(canvas)
  const labelDiv = document.createElement('div')
  labelDiv.textContent = 'Big Cube'
  labelDiv.style.cssText =
    'color:#fff;font:bold 16px monospace;background:rgba(0,0,0,0.6);padding:4px 10px;border-radius:4px;white-space:nowrap'
  overlay.add({ element: labelDiv, node: cube, center: true })

  // ─── Colored Cubes ──────────────────────────────────────────────────
  const COLORED_COUNT = 10
  const COLORED_RING_RADIUS = 30
  const COLORED_Z = 25
  const COLORED_CUBE_SIZE = 10

  const coloredCubeGeo = new BoxGeometry({
    width: COLORED_CUBE_SIZE,
    height: COLORED_CUBE_SIZE,
    depth: COLORED_CUBE_SIZE,
  })

  const cubeColors: [number, number, number][] = [
    [1.0, 0.2, 0.2], // red
    [0.2, 1.0, 0.3], // green
    [0.2, 0.4, 1.0], // blue
    [1.0, 0.8, 0.1], // yellow
    [1.0, 0.4, 0.0], // orange
    [0.8, 0.2, 1.0], // purple
    [0.0, 0.9, 0.9], // cyan
    [1.0, 0.3, 0.6], // pink
    [0.4, 1.0, 0.7], // mint
    [1.0, 1.0, 1.0], // white
  ]

  const coloredCubes: Mesh[] = []
  const coloredAxes: Float32Array[] = []
  const coloredSpeeds: number[] = []

  for (let i = 0; i < COLORED_COUNT; i++) {
    const angle = (i / COLORED_COUNT) * Math.PI * 2
    const mat = new LambertMaterial({ color: cubeColors[i]! })
    const mesh = new Mesh(coloredCubeGeo, mat)
    mesh.setPosition(Math.cos(angle) * COLORED_RING_RADIUS, Math.sin(angle) * COLORED_RING_RADIUS, COLORED_Z)
    scene.add(mesh)
    coloredCubes.push(mesh)

    // Each cube gets a unique rotation axis and speed
    const ax = Math.random() - 0.5
    const ay = Math.random() - 0.5
    const az = Math.random() - 0.5
    const len = Math.sqrt(ax * ax + ay * ay + az * az)
    coloredAxes.push(new Float32Array([ax / len, ay / len, az / len]))
    coloredSpeeds.push(1.5 + Math.random() * 2.0)
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

  // ─── Scheduler ──────────────────────────────────────────────
  engine.maxFps = 60
  const cubeAxis = new Float32Array([0, 0, 1])

  // Update callback (runs before render)
  engine.register(
    ({ dt, elapsed }) => {
      controls.update(dt)

      for (let i = 0; i < roots.length; i++) {
        // Circular orbit around each character's origin point
        charOrbitAngle[i] = charOrbitAngle[i]! + ORBIT_SPEED * dt
        const a = charOrbitAngle[i]!
        roots[i]!.setPositionX(charOriginX[i]! + Math.cos(a) * ORBIT_RADIUS)
        roots[i]!.setPositionY(charOriginY[i]! + Math.sin(a) * ORBIT_RADIUS)
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

      // Rotate cube around Z axis (Z-up)
      quatFromAxisAngle(cube.rotation, cubeAxis, elapsed)
      cube.markTransformDirty()

      // Rotate colored cubes individually
      for (let i = 0; i < coloredCubes.length; i++) {
        quatFromAxisAngle(coloredCubes[i]!.rotation, coloredAxes[i]!, elapsed * coloredSpeeds[i]!)
        coloredCubes[i]!.markTransformDirty()
      }
    },
    { priority: -1 },
  )

  // Render callback
  let lastStatsUpdate = -1
  const backendLabel = engine.backend.toUpperCase()
  engine.register(
    ({ elapsed }) => {
      engine.render(scene, camera)
      overlay.update(camera, canvas.clientWidth, canvas.clientHeight)
      if (elapsed - lastStatsUpdate >= 0.5) {
        lastStatsUpdate = elapsed
        const stats = engine.getStats()
        const shadowSuffix = stats.shadowDrawCalls > 0 ? ` (+${stats.shadowDrawCalls} CSM)` : ''
        const dpr = Math.min(window.devicePixelRatio, engine.maxDpr)
        statsDiv.textContent = `${backendLabel} | ${Math.round(stats.fps)} FPS | DPR: ${dpr.toFixed(2)} | Draw calls: ${stats.drawCalls}${shadowSuffix}`
      }
    },
    { priority: 0 },
  )

  engine.start()

  return engine
}
