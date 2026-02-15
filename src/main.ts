import { createRaycastHit } from './engine/bvh.ts'
import { mergeGeometries } from './engine/geometry.ts'
import { loadGLTF } from './engine/gltf.ts'
import { createRenderer } from './engine/gpu.ts'
import { loadKTX2 } from './engine/ktx2.ts'
import { Mesh } from './engine/mesh.ts'
import { OrbitControls } from './engine/orbit-controls.ts'
import { Scene } from './engine/scene.ts'
import {
  createSkeleton,
  createSkinInstance,
  updateSkinInstance,
  transitionTo,
  findBoneNodeIndex,
} from './engine/skin.ts'

import type { GltfAnimation } from './engine/gltf.ts'
import type { Backend } from './engine/gpu.ts'
import type { Skeleton, SkinInstance } from './engine/skin.ts'

// Grid configuration
const GRID_SIZE = 20
const GRID_SPACING = 5

// Animation cycling: each character transitions between anims on a staggered timer
const ANIM_CYCLE_BASE = 3.0 // seconds between animation transitions
const ANIM_CYCLE_JITTER = 2.0 // random extra seconds
const CROSSFADE_DURATION = 0.2 // 200ms crossfade

// Orbit motion
const ORBIT_RADIUS = 2.5
const BASE_ORBIT_SPEED = 0.5 // radians per second

export const main = async (canvas: HTMLCanvasElement) => {
  const savedBackend = localStorage.getItem('voidcore-backend') as Backend | null
  const scene = await Scene.create(canvas, { backend: savedBackend ?? undefined })
  console.log(`VoidCore renderer: ${scene.renderer.backend}`)

  // Load megaxe mesh from static bundle
  const megaxeColors = new Map<number, [number, number, number]>([
    [0, [0.95, 0.95, 0.95]],
    [1, [0.1, 0.1, 0.1]],
    [2, [0, 0.9, 0.8]],
  ])

  const megaxeBloom = new Map<number, number>([
    [2, 1.0], // Material index 2 (teal parts) gets bloom
  ])

  // Eden per-primitive colors (indexed by primitive order in the GLB)
  // prettier-ignore
  const EDEN_COLORS: [number, number, number][] = [
    [0.78, 0.44, 0.25],   // 0
    [0.85, 0.68, 0.3],    // 1
    [0.75, 0.75, 0.75],   // 2
    [0.725, 0.38, 0.09],  // 3
    [0.25, 0.55, 0.2],    // 4
    [0.212, 0.212, 0.212], // 5
    [0.75, 0.75, 0.75],   // 6
    [0.15, 0.15, 0.18],   // 7
    [0.0, 0.75, 0.7],     // 8
    [0.55, 0.5, 0.42],    // 9
    [0.85, 0.35, 0.55],   // 10
    [0.95, 0.95, 0.95],   // 11
    [0.8, 0.65, 0.15],    // 12
    [0.65, 0.65, 0.62],   // 13
    [0.3, 0.65, 0.2],     // 14
    [0.5, 0.32, 0.15],    // 15
    [0.5, 0.25, 0.65],    // 16
    [0.9, 0.8, 0.2],      // 17
    [0.15, 0.15, 0.18],   // 18
    [0.75, 0.75, 0.75],   // 19
    [0.95, 0.95, 0.95],   // 20
    [0.6, 0.4, 0.22],     // 21
    [0.6, 0.4, 0.22],     // 22
    [0.0, 0.75, 0.7],     // 23  (bloom)
    [0.9, 0.8, 0.2],      // 24
    [0.85, 0.35, 0.55],   // 25
    [0.3, 0.36, 0.3],     // 26
  ]
  const EDEN_BLOOM_PRIM = 23 // primitive index 23 gets bloom

  let megaxeId: number | null = null
  let edenGeoId: number | null = null
  let aoTexId: number | null = null
  let edenHasUVs = false
  try {
    const gltf = await loadGLTF('/static-bundle.glb', { materialColors: megaxeColors, materialBloom: megaxeBloom })
    const megaxeMesh = gltf.meshes.find(m => m.name === 'megaxe')
    if (megaxeMesh && megaxeMesh.primitives.length > 0) {
      const prim = megaxeMesh.primitives[0]!
      megaxeId = scene.registerGeometry(prim.geometry)
      console.log(`Loaded megaxe mesh (${prim.geometry.vertices.length / 10} vertices)`)
    }

    // Load Eden mesh (multiple primitives, one per material — merge into single geometry)
    const edenMesh = gltf.meshes.find(m => m.name === 'Eden')
    if (edenMesh && edenMesh.primitives.length > 0) {
      const merged = mergeGeometries(
        edenMesh.primitives.map((prim, i) => ({
          vertices: prim.geometry.vertices,
          indices: prim.geometry.indices,
          color: EDEN_COLORS[i] ?? [1, 1, 1],
          bloom: i === EDEN_BLOOM_PRIM ? 1.0 : undefined,
          uvs: prim.uvs,
        })),
      )
      edenHasUVs = !!merged.uvs

      if (edenHasUVs) {
        // Load AO texture and register textured geometry
        try {
          const aoTex = await loadKTX2('/city-ao.ktx2', '/basis-1.50/')
          aoTexId = scene.registerTexture(aoTex.data, aoTex.width, aoTex.height)
          edenGeoId = scene.registerTexturedGeometry(merged, merged.uvs!)
          console.log(
            `Loaded Eden textured (${edenMesh.primitives.length} primitives merged, ${merged.vertices.length / 10} vertices, AO ${aoTex.width}×${aoTex.height})`,
          )
        } catch (e) {
          console.warn('Failed to load AO texture, falling back to non-textured:', e)
          edenGeoId = scene.registerGeometry(merged)
        }
      } else {
        edenGeoId = scene.registerGeometry(merged)
        console.log(
          `Loaded Eden (${edenMesh.primitives.length} primitives merged, ${merged.vertices.length / 10} vertices)`,
        )
      }
    }
  } catch (e) {
    console.warn('Failed to load static-bundle.glb:', e)
  }

  // Load player character (skinned mesh)
  let skeleton: Skeleton | undefined
  let animations: GltfAnimation[] = []
  let skinnedGeoId: number | undefined
  let handBoneIdx = -1

  // Animation clip indices
  let idleIdx = 0
  let runIdx = -1
  let slashIdx = -1

  try {
    const playerGltf = await loadGLTF('/player-bundle.glb')
    const bodyMesh = playerGltf.meshes.find(m => m.name === 'Body' && m.skinIndex !== undefined)

    if (bodyMesh && bodyMesh.primitives.length > 0) {
      const prim = bodyMesh.primitives[0]!
      const skin = playerGltf.skins[bodyMesh.skinIndex!]!
      skeleton = createSkeleton(skin, playerGltf.nodeTransforms)
      animations = playerGltf.animations

      // Find animation clip indices
      for (let i = 0; i < animations.length; i++) {
        const name = animations[i]!.name.toLowerCase()
        if (name === 'jump') idleIdx = i
        else if (name === 'run') runIdx = i
        else if (name === 'slash') slashIdx = i
      }

      // Register skinned geometry (shared by all characters)
      skinnedGeoId = scene.registerSkinnedGeometry(prim.geometry, prim.skinJoints!, prim.skinWeights!)

      // Find hand bone for megaxe attachment
      if (megaxeId !== null) {
        try {
          handBoneIdx = findBoneNodeIndex(skeleton, 'Hand.R')
        } catch {
          console.warn('Hand.R bone not found')
        }
      }

      console.log(
        `Loaded player (${prim.geometry.vertices.length / 10} verts, ${skeleton.jointCount} joints, ${animations.length} anims: ${animations.map(a => a.name).join(', ')})`,
      )
    }
  } catch (e) {
    console.warn('Failed to load player-bundle.glb:', e)
  }

  if (!skeleton || skinnedGeoId === undefined) {
    console.error('Failed to load player assets')
    return
  }

  // Build list of available animation indices for cycling
  const availableAnims: number[] = [idleIdx]
  if (runIdx >= 0) availableAnims.push(runIdx)
  if (slashIdx >= 0) availableAnims.push(slashIdx)

  // Add Eden map mesh
  let edenMeshRef: Mesh | null = null
  if (edenGeoId !== null) {
    edenMeshRef = scene.add(
      new Mesh({
        geometryId: edenGeoId,
        position: [-53.5, -69.5, 0],
        color: [1, 1, 1, 1],
        scale: [1, 1, 1],
        aoMap: aoTexId ?? undefined,
        aoIntensity: 2,
      }),
    )
    scene.buildBVH(edenGeoId)
  }

  // Spawn grid of characters
  const skinInstances: SkinInstance[] = []
  const animTimers: number[] = [] // time until next transition
  const currentAnimSlot: number[] = [] // index into availableAnims
  const bodyMeshes: Mesh[] = []
  const weaponMeshes: (Mesh | null)[] = []
  const spawnX: number[] = []
  const spawnY: number[] = []
  const orbitSpeeds: number[] = []
  const orbitPhases: number[] = []

  const totalChars = GRID_SIZE * GRID_SIZE
  const gridOffset = ((GRID_SIZE - 1) * GRID_SPACING) / 2

  for (let row = 0; row < GRID_SIZE; row++) {
    for (let col = 0; col < GRID_SIZE; col++) {
      const charIdx = row * GRID_SIZE + col

      // Staggered starting animation
      const startAnimSlot = charIdx % availableAnims.length
      const startClipIdx = availableAnims[startAnimSlot]!

      const inst = createSkinInstance(skeleton, startClipIdx)

      // Offset animation time so characters aren't all in sync
      const clipDuration = animations[startClipIdx]!.duration
      inst.time = clipDuration > 0 ? (charIdx * 0.13) % clipDuration : 0

      // Initial pose
      updateSkinInstance(inst, animations, 0)
      skinInstances.push(inst)

      const x = col * GRID_SPACING - gridOffset
      const y = row * GRID_SPACING - gridOffset
      spawnX.push(x)
      spawnY.push(y)
      orbitSpeeds.push(BASE_ORBIT_SPEED + (charIdx % 7) * 0.1)
      orbitPhases.push(charIdx * 1.37)

      // Add skinned body mesh
      const body = scene.add(
        new Mesh({
          geometryId: skinnedGeoId,
          position: [x, y, -40],
          color: [1, 1, 1, 1],
          scale: [1, 1, 1],
          skinInstance: inst,
        }),
      )
      bodyMeshes.push(body)

      // Attach megaxe to hand bone
      if (megaxeId !== null && handBoneIdx >= 0) {
        const weapon = scene.add(
          new Mesh({
            geometryId: megaxeId,
            position: [x, y, -40],
            rotation: [0, 0, Math.PI],
            color: [1, 1, 1, 1],
            scale: [1, 1, 1],
            boneAttachment: { skinInstance: inst, boneNodeIndex: handBoneIdx },
          }),
        )
        weaponMeshes.push(weapon)
      } else {
        weaponMeshes.push(null)
      }

      // Stagger transition timers so they don't all switch at once
      animTimers.push(ANIM_CYCLE_BASE + Math.random() * ANIM_CYCLE_JITTER + charIdx * 0.01)
      currentAnimSlot.push(startAnimSlot)
    }
  }

  const totalEntities = totalChars + (megaxeId !== null && handBoneIdx >= 0 ? totalChars : 0)

  console.log(
    `Spawned ${totalChars} characters (${totalEntities} entities)${megaxeId !== null && handBoneIdx >= 0 ? ' with megaxes' : ''}`,
  )

  // Lighting
  scene.setDirectionalLight([0, -1, -1], [0.6, 0.6, 0.6])
  scene.setAmbientLight([0.7, 0.7, 0.7])

  // Bloom
  scene.setBloom({ enabled: true, intensity: 5.0, radius: 1.0 })

  // Camera
  const camera = scene.camera
  camera.fov = Math.PI / 4
  camera.near = 0.1
  camera.far = 500

  const orbitRadius = GRID_SIZE * GRID_SPACING * 1.1
  camera.eye[0] = 0
  camera.eye[1] = -orbitRadius
  camera.eye[2] = orbitRadius * 0.4
  camera.target[0] = 0
  camera.target[1] = 0
  camera.target[2] = 0

  let controls = new OrbitControls(camera, canvas)
  let currentCanvas = canvas

  const resize = () => {
    const dpr = window.devicePixelRatio || 1
    const w = Math.floor(currentCanvas.clientWidth * dpr)
    const h = Math.floor(currentCanvas.clientHeight * dpr)
    if (currentCanvas.width !== w || currentCanvas.height !== h) {
      scene.resize(w, h)
    }
  }
  window.addEventListener('resize', resize)
  resize()

  // Stats overlay
  const stats = document.createElement('div')
  stats.style.cssText =
    'position:fixed;top:8px;left:8px;color:#fff;font:12px monospace;background:rgba(0,0,0,0.6);padding:4px 8px;border-radius:4px;pointer-events:none;z-index:999'
  document.body.appendChild(stats)
  let frameCount = 0
  let lastFpsTime = performance.now()
  let fps = 0

  // Backend toggle button
  let switching = false
  const toggle = document.createElement('button')
  const updateToggle = () => {
    const current = scene.renderer.backend
    const other: Backend = current === 'webgpu' ? 'webgl' : 'webgpu'
    toggle.textContent = `${current.toUpperCase()} | Switch to ${other.toUpperCase()}`
  }
  updateToggle()
  toggle.style.cssText =
    'position:fixed;top:8px;right:8px;color:#fff;font:12px monospace;background:rgba(0,0,0,0.6);padding:4px 8px;border-radius:4px;border:1px solid rgba(255,255,255,0.3);cursor:pointer;z-index:999'
  toggle.addEventListener('click', async () => {
    if (switching) return
    switching = true
    toggle.textContent = 'Switching...'
    const target: Backend = scene.renderer.backend === 'webgpu' ? 'webgl' : 'webgpu'

    // Fully destroy old renderer before creating anything new
    scene.destroy()
    controls.dispose()

    // Create fresh canvas and renderer with zero overlap
    const newCanvas = document.createElement('canvas')
    currentCanvas.replaceWith(newCanvas)
    currentCanvas = newCanvas
    const newRenderer = await createRenderer(newCanvas, target)
    scene.setRenderer(newCanvas, newRenderer)

    localStorage.setItem('voidcore-backend', target)
    controls = new OrbitControls(camera, newCanvas)
    resize()
    updateToggle()
    switching = false
  })
  document.body.appendChild(toggle)

  // Raycast hit result (reused every frame)
  const rayHit = createRaycastHit()
  const edenFilter = (m: Mesh) => m === edenMeshRef

  // Animation loop
  let lastTime = performance.now()
  let orbitTime = 0

  const frame = (now: number) => {
    const dt = Math.min((now - lastTime) / 1000, 0.05)
    lastTime = now
    orbitTime += dt

    // FPS counter
    frameCount++
    if (now - lastFpsTime >= 1000) {
      fps = frameCount
      frameCount = 0
      lastFpsTime = now
    }

    // Update animation timers and transitions
    for (let i = 0; i < totalChars; i++) {
      animTimers[i]! -= dt
      if (animTimers[i]! <= 0) {
        // Cycle to next animation
        currentAnimSlot[i] = (currentAnimSlot[i]! + 1) % availableAnims.length
        const nextClip = availableAnims[currentAnimSlot[i]!]!
        transitionTo(skinInstances[i]!, nextClip, CROSSFADE_DURATION)
        animTimers[i] = ANIM_CYCLE_BASE + Math.random() * ANIM_CYCLE_JITTER
      }
    }

    // Update all skin instances
    for (const inst of skinInstances) {
      updateSkinInstance(inst, animations, dt)
    }

    // Orbit characters around spawn points + raycast ground clamping
    for (let i = 0; i < totalChars; i++) {
      const angle = orbitTime * orbitSpeeds[i]! + orbitPhases[i]!
      const cx = spawnX[i]! + Math.cos(angle) * ORBIT_RADIUS
      const cy = spawnY[i]! + Math.sin(angle) * ORBIT_RADIUS

      const body = bodyMeshes[i]!
      body.position[0] = cx
      body.position[1] = cy

      // Raycast downward to find ground (reset Z each frame to avoid stale values on miss)
      if (edenMeshRef && scene.raycast(cx, cy, 20, 0, 0, -1, rayHit, edenFilter, 50)) {
        body.position[2] = rayHit.pointZ
      } else {
        body.position[2] = -40
      }

      body.setDirty()

      // Sync weapon position
      const weapon = weaponMeshes[i]
      if (weapon) {
        weapon.position[0] = cx
        weapon.position[1] = cy
        weapon.position[2] = body.position[2]!
      }
    }

    controls.update()
    resize()
    scene.render()

    stats.textContent = `${fps} fps | ${scene.drawCalls} draws | ${scene.visibleCount}/${totalEntities} visible | ${totalChars} skinned`

    requestAnimationFrame(frame)
  }

  requestAnimationFrame(frame)
}
