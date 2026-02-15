import { createBoxGeometry, createSphereGeometry } from './engine/geometry.ts'
import { Mesh } from './engine/mesh.ts'
import { Scene } from './engine/scene.ts'

import type { Backend } from './engine/gpu.ts'

export const main = async (canvas: HTMLCanvasElement) => {
  // Create scene (restore saved backend preference, or auto-pick)
  const savedBackend = localStorage.getItem('voidcore-backend') as Backend | null
  const scene = await Scene.create(canvas, { backend: savedBackend ?? undefined })
  console.log(`VoidCore renderer: ${scene.renderer.backend}`)

  // Register geometries
  const boxGeo = createBoxGeometry(1, 1, 1)
  const sphereGeo = createSphereGeometry(0.5, 16, 12)
  const groundGeo = createBoxGeometry(40, 40, 0.2)

  const boxId = scene.registerGeometry(boxGeo)
  const sphereId = scene.registerGeometry(sphereGeo)
  const groundId = scene.registerGeometry(groundGeo)

  // Ground plane
  scene.add(
    new Mesh({
      geometryId: groundId,
      position: [0, 0, -0.1],
      color: [0.3, 0.3, 0.35, 1],
    }),
  )

  // Random cubes and spheres
  const animatedMeshes: Mesh[] = []
  const entityCount = 4000

  for (let i = 0; i < entityCount; i++) {
    const isSphere = Math.random() > 0.5
    const x = (Math.random() - 0.5) * 30
    const y = (Math.random() - 0.5) * 30
    const z = Math.random() * 5 + 0.5
    const r = Math.random() * 0.8 + 0.2
    const g = Math.random() * 0.8 + 0.2
    const b = Math.random() * 0.8 + 0.2

    const mesh = new Mesh({
      geometryId: isSphere ? sphereId : boxId,
      position: [x, y, z],
      color: [r, g, b, 1],
      scale: isSphere
        ? [0.5 + Math.random(), 0.5 + Math.random(), 0.5 + Math.random()]
        : [0.5 + Math.random() * 1.5, 0.5 + Math.random() * 1.5, 0.5 + Math.random() * 1.5],
    })

    scene.add(mesh)

    if (Math.random() > 0.7) {
      animatedMeshes.push(mesh)
    }
  }

  // Lighting
  scene.setDirectionalLight([0.5, 0.3, -1], [1, 0.95, 0.9])
  scene.setAmbientLight([0.15, 0.15, 0.2])

  // Camera orbit
  const camera = scene.camera
  camera.fov = Math.PI / 4
  camera.near = 0.1
  camera.far = 500

  // Track current canvas for resize
  let currentCanvas = canvas

  // Handle resize
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
    // Create a new canvas (can't reuse a canvas that already has a different context)
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
  let time = 0

  const frame = () => {
    time += 0.016

    // FPS counter
    frameCount++
    const now = performance.now()
    if (now - lastFpsTime >= 1000) {
      fps = frameCount
      frameCount = 0
      lastFpsTime = now
    }

    // Orbit camera (Z-up)
    const orbitRadius = 20
    const orbitSpeed = 0.2
    camera.eye[0] = Math.sin(time * orbitSpeed) * orbitRadius
    camera.eye[1] = Math.cos(time * orbitSpeed) * orbitRadius
    camera.eye[2] = 8 + Math.sin(time * 0.3) * 3
    camera.target[0] = 0
    camera.target[1] = 0
    camera.target[2] = 1

    // Animate some meshes
    for (const mesh of animatedMeshes) {
      mesh.rotation[2] = (mesh.rotation[2] ?? 0) + 0.02
      mesh.setDirty()
    }

    resize()
    scene.render()

    stats.textContent = `${fps} fps | ${scene.drawCalls} draws | ${scene.visibleCount}/${entityCount + 1} visible`

    requestAnimationFrame(frame)
  }

  requestAnimationFrame(frame)
}
