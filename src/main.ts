import { loadGLTF } from './engine/gltf.ts'
import { Mesh } from './engine/mesh.ts'
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
const GRID_SIZE = 20 // 20x20 = 400 characters
const GRID_SPACING = 2.5

// Animation cycling: each character transitions between anims on a staggered timer
const ANIM_CYCLE_BASE = 3.0 // seconds between animation transitions
const ANIM_CYCLE_JITTER = 2.0 // random extra seconds
const CROSSFADE_DURATION = 0.2 // 200ms crossfade

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

  let megaxeId: number | null = null
  try {
    const gltf = await loadGLTF('/static-bundle.glb', { materialColors: megaxeColors })
    const megaxeMesh = gltf.meshes.find(m => m.name === 'megaxe')
    if (megaxeMesh && megaxeMesh.primitives.length > 0) {
      const prim = megaxeMesh.primitives[0]!
      megaxeId = scene.registerGeometry(prim.geometry)
      console.log(`Loaded megaxe mesh (${prim.geometry.vertices.length / 9} vertices)`)
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
        if (name.includes('idle')) idleIdx = i
        else if (name.includes('run')) runIdx = i
        else if (name.includes('slash')) slashIdx = i
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
        `Loaded player (${prim.geometry.vertices.length / 9} verts, ${skeleton.jointCount} joints, ${animations.length} anims: ${animations.map(a => a.name).join(', ')})`,
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

  // Spawn grid of characters
  const skinInstances: SkinInstance[] = []
  const animTimers: number[] = [] // time until next transition
  const currentAnimSlot: number[] = [] // index into availableAnims

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

      // Add skinned body mesh
      scene.add(
        new Mesh({
          geometryId: skinnedGeoId,
          position: [x, y, 0],
          color: [1, 1, 1, 1],
          scale: [1, 1, 1],
          skinInstance: inst,
        }),
      )

      // Attach megaxe to hand bone
      if (megaxeId !== null && handBoneIdx >= 0) {
        scene.add(
          new Mesh({
            geometryId: megaxeId,
            position: [x, y, 0],
            color: [1, 1, 1, 1],
            scale: [1, 1, 1],
            boneAttachment: { skinInstance: inst, boneNodeIndex: handBoneIdx },
          }),
        )
      }

      // Stagger transition timers so they don't all switch at once
      animTimers.push(ANIM_CYCLE_BASE + Math.random() * ANIM_CYCLE_JITTER + charIdx * 0.01)
      currentAnimSlot.push(startAnimSlot)
    }
  }

  console.log(`Spawned ${totalChars} characters${megaxeId !== null && handBoneIdx >= 0 ? ' with megaxes' : ''}`)

  // Lighting
  scene.setDirectionalLight([0.5, 0.3, -1], [1, 0.95, 0.9])
  scene.setAmbientLight([0.15, 0.15, 0.2])

  // Camera orbit
  const camera = scene.camera
  camera.fov = Math.PI / 4
  camera.near = 0.1
  camera.far = 500

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
    const newCanvas = document.createElement('canvas')
    currentCanvas.replaceWith(newCanvas)
    currentCanvas = newCanvas
    await scene.switchBackend(newCanvas, target)
    localStorage.setItem('voidcore-backend', target)
    resize()
    updateToggle()
    switching = false
  })
  document.body.appendChild(toggle)

  // Animation loop
  let lastTime = performance.now()

  const frame = (now: number) => {
    const dt = Math.min((now - lastTime) / 1000, 0.05)
    lastTime = now

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

    // Orbit camera (Z-up)
    const time = now / 1000
    const orbitRadius = GRID_SIZE * GRID_SPACING * 0.7
    const orbitSpeed = 0.15
    camera.eye[0] = Math.sin(time * orbitSpeed) * orbitRadius
    camera.eye[1] = Math.cos(time * orbitSpeed) * orbitRadius
    camera.eye[2] = orbitRadius * 0.4 + Math.sin(time * 0.2) * 3
    camera.target[0] = 0
    camera.target[1] = 0
    camera.target[2] = 1

    resize()
    scene.render()

    stats.textContent = `${fps} fps | ${scene.drawCalls} draws | ${scene.visibleCount}/${totalChars} visible | ${totalChars} skinned`

    requestAnimationFrame(frame)
  }

  requestAnimationFrame(frame)
}
