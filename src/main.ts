import {
  createEngine,
  createScene,
  createPerspectiveCamera,
  createDirectionalLight,
  createMesh,
  createLambertMaterial,
  createPlaneGeometry,
  createBoxGeometry,
  createSphereGeometry,
  createConeGeometry,
  createCylinderGeometry,
  createCapsuleGeometry,
  createCircleGeometry,
  createOrbitControls,
  loadGLTF,
  createAnimationMixer,
} from './index.ts'

import type { Skeleton } from './animation/skeleton.ts'
import type { AnimationClip, AnimationAction } from './index.ts'

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
  camera.position[0] = 8
  camera.position[1] = -12
  camera.position[2] = 8
  camera._dirtyLocal = true
  camera._dirtyWorld = true

  const controls = createOrbitControls(camera, canvas, {
    target: [0, 0, 1.5],
    dampingFactor: 0.08,
    minDistance: 2,
    maxDistance: 60,
  })

  // ─── Load megaxe from static-bundle.glb ─────────────────────────
  let megaxeMesh: ReturnType<typeof createMesh> | null = null

  try {
    const gltf = await loadGLTF('/static-bundle.glb', engine, {
      draco: { decoderPath: '/draco-1.5.7/' },
    })

    // Find the megaxe mesh and add only that to the scene
    for (const mesh of gltf.meshes) {
      if (mesh.name.toLowerCase().includes('megaxe')) {
        // Override material with palette:
        // 0: near white, 1: near black, 2: bright teal with bloom
        mesh.material = createLambertMaterial({
          palette: [
            { color: [0.95, 0.93, 0.9] }, // 0: near white
            { color: [0.08, 0.08, 0.1] }, // 1: near black
            { color: [0, 0.9, 0.85], emissive: [0, 1, 0.9], emissiveIntensity: 2.5 }, // 2: bright teal with bloom
          ],
        })
        mesh.frustumCulled = false
        mesh.position[2] = 1.5
        mesh._dirtyLocal = true
        scene.add(mesh)
        megaxeMesh = mesh
      }
    }
  } catch (e) {
    console.warn('Failed to load static-bundle.glb:', e)
  }

  // ─── Load player-bundle.glb (skinned character) ─────────────────
  let playerSkeleton: Skeleton | null = null
  let playerClips: AnimationClip[] = []
  let mixer: ReturnType<typeof createAnimationMixer> | null = null
  let actions: AnimationAction[] = []
  let currentActionIndex = 0
  let nextSwitchTime = 0
  const clipDuration = 2 // seconds per clip before crossfade
  const crossfadeDuration = 0.3

  try {
    const playerGltf = await loadGLTF('/player-bundle.glb', engine, {
      draco: { decoderPath: '/draco-1.5.7/' },
    })

    scene.add(playerGltf.scene)

    if (playerGltf.skeletons.length > 0) {
      playerSkeleton = playerGltf.skeletons[0]!

      // Attach megaxe to Hand.R bone
      if (megaxeMesh && playerSkeleton) {
        const handBone = playerSkeleton.getBone('Hand.R')
        if (handBone) {
          // Reparent megaxe under hand bone
          if (megaxeMesh.parent) megaxeMesh.parent.remove(megaxeMesh)
          handBone.add(megaxeMesh)
          // Reset transform so it sits at bone origin
          megaxeMesh.position[0] = 0
          megaxeMesh.position[1] = 0
          megaxeMesh.position[2] = 0
          megaxeMesh.rotation[0] = 0
          megaxeMesh.rotation[1] = 0
          megaxeMesh.rotation[2] = 0
          megaxeMesh.rotation[3] = 1
          megaxeMesh.scale[0] = 1
          megaxeMesh.scale[1] = 1
          megaxeMesh.scale[2] = 1
          megaxeMesh._dirtyLocal = true
        }
      }
    }

    // Find animation clips by name
    const clipNames = ['SlashRight', 'Jump', 'Run']
    for (const name of clipNames) {
      const clip = playerGltf.animations.find(a => a.name === name)
      if (clip) playerClips.push(clip)
    }

    // Fallback: use whatever animations are available
    if (playerClips.length === 0) {
      playerClips = playerGltf.animations.slice(0, 3)
    }

    // Set up mixer with actions
    if (playerSkeleton && playerClips.length > 0) {
      mixer = createAnimationMixer(playerSkeleton)
      for (const clip of playerClips) {
        actions.push(mixer.clipAction(clip))
      }
      // Start the first action
      actions[0]!.play()
      actions[0]!.weight = 1
      nextSwitchTime = clipDuration
    }
  } catch (e) {
    console.warn('Failed to load player-bundle.glb:', e)
  }

  // ─── Geometry Showcase ──────────────────────────────────────────

  const geoMaterial = createLambertMaterial({ color: [0.7, 0.5, 0.9] })
  const geoMaterial2 = createLambertMaterial({ color: [0.4, 0.8, 0.6] })
  const geoMaterial3 = createLambertMaterial({ color: [0.9, 0.6, 0.3] })

  const spacing = 2.5
  const startX = -spacing * 3

  // Ground plane
  const ground = createMesh(
    createPlaneGeometry({ width: 30, height: 30 }),
    createLambertMaterial({ color: [0.25, 0.25, 0.28] }),
  )
  scene.add(ground)

  // Plane (vertical display)
  const planeMesh = createMesh(createPlaneGeometry({ width: 1.5, height: 1.5 }), geoMaterial)
  planeMesh.position[0] = startX
  planeMesh.position[1] = 5
  planeMesh.position[2] = 1.5
  // Tilt the plane so it's visible
  const halfPi = Math.PI / 4
  const s = Math.sin(halfPi / 2)
  planeMesh.rotation[0] = s // rotate around X to face camera
  planeMesh.rotation[3] = Math.cos(halfPi / 2)
  planeMesh._dirtyLocal = true
  scene.add(planeMesh)

  // Box
  const boxMesh = createMesh(createBoxGeometry({ width: 1.2, height: 1.2, depth: 1.2 }), geoMaterial2)
  boxMesh.position[0] = startX + spacing
  boxMesh.position[1] = 5
  boxMesh.position[2] = 1
  boxMesh._dirtyLocal = true
  scene.add(boxMesh)

  // Sphere
  const sphereMesh = createMesh(
    createSphereGeometry({ radius: 0.7, widthSegments: 24, heightSegments: 12 }),
    geoMaterial3,
  )
  sphereMesh.position[0] = startX + spacing * 2
  sphereMesh.position[1] = 5
  sphereMesh.position[2] = 1
  sphereMesh._dirtyLocal = true
  scene.add(sphereMesh)

  // Cone
  const coneMesh = createMesh(createConeGeometry({ radius: 0.6, height: 1.4, radialSegments: 24 }), geoMaterial)
  coneMesh.position[0] = startX + spacing * 3
  coneMesh.position[1] = 5
  coneMesh.position[2] = 0
  coneMesh._dirtyLocal = true
  scene.add(coneMesh)

  // Cylinder
  const cylMesh = createMesh(
    createCylinderGeometry({ radiusTop: 0.5, radiusBottom: 0.5, height: 1.5, radialSegments: 24 }),
    geoMaterial2,
  )
  cylMesh.position[0] = startX + spacing * 4
  cylMesh.position[1] = 5
  cylMesh.position[2] = 0.75
  cylMesh._dirtyLocal = true
  scene.add(cylMesh)

  // Capsule
  const capMesh = createMesh(createCapsuleGeometry({ radius: 0.4, height: 1.6, radialSegments: 16 }), geoMaterial3)
  capMesh.position[0] = startX + spacing * 5
  capMesh.position[1] = 5
  capMesh.position[2] = 0.8
  capMesh._dirtyLocal = true
  scene.add(capMesh)

  // Circle
  const circleMesh = createMesh(createCircleGeometry({ radius: 0.7, segments: 24 }), geoMaterial)
  circleMesh.position[0] = startX + spacing * 6
  circleMesh.position[1] = 5
  circleMesh.position[2] = 1.5
  // Tilt the circle
  circleMesh.rotation[0] = s
  circleMesh.rotation[3] = Math.cos(halfPi / 2)
  circleMesh._dirtyLocal = true
  scene.add(circleMesh)

  // ─── Render Loop ────────────────────────────────────────────────
  engine.onFrame((dt, elapsed) => {
    controls.update(dt)

    // Slowly rotate the geometry showcase items
    const time = performance.now() / 1000
    boxMesh.rotation[2] = Math.sin(time * 0.3) * 0.3
    boxMesh.rotation[3] = Math.cos(time * 0.3) * 0.3 + 0.7
    boxMesh._dirtyLocal = true

    sphereMesh.position[2] = 1 + Math.sin(time * 0.5) * 0.3
    sphereMesh._dirtyLocal = true

    // Animate player character with crossfades
    if (mixer && actions.length > 1) {
      // Check if it's time to crossfade to the next clip
      if (elapsed >= nextSwitchTime) {
        const nextIndex = (currentActionIndex + 1) % actions.length
        const current = actions[currentActionIndex]!
        const next = actions[nextIndex]!
        current.crossFadeTo(next, crossfadeDuration)
        currentActionIndex = nextIndex
        nextSwitchTime = elapsed + clipDuration
      }
      mixer.update(dt)
    } else if (mixer) {
      mixer.update(dt)
    }
  })

  engine.start(scene, camera)

  return engine
}
