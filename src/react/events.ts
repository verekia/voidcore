// Events – Pointer event system for VoidCore React bindings.
//
// Attaches DOM listeners to the canvas and converts pointer positions to raycasts.
// Walks hit results to find instances with event handlers (onClick, onPointerOver, etc.)
// and dispatches synthetic events with 3D intersection data.

import { instanceMap } from './reconciler'

import type { FloatArray } from '../float'
import type { RaycastHit } from '../raycasting/index'
import type { Mesh } from '../scene/mesh'
import type { Node } from '../scene/node'
import type { VoidStore } from './context'
import type { VoidInstance } from './reconciler'

export interface VoidEvent {
  object: any
  point: FloatArray
  normal: FloatArray
  distance: number
  uv: FloatArray | null
  nativeEvent: PointerEvent | MouseEvent
  stopPropagation: () => void
}

let hoveredInstance: VoidInstance | null = null

const collectMeshes = (node: Node, meshes: Mesh[]) => {
  if (node.type === 'mesh' && node.visible) {
    meshes.push(node as Mesh)
  }
  for (const child of node.children) {
    collectMeshes(child, meshes)
  }
}

const toNDC = (e: PointerEvent | MouseEvent, canvas: HTMLCanvasElement): [number, number] => {
  const rect = canvas.getBoundingClientRect()
  const x = ((e.clientX - rect.left) / rect.width) * 2 - 1
  const y = -((e.clientY - rect.top) / rect.height) * 2 + 1
  return [x, y]
}

const raycast = (store: VoidStore, ndc: [number, number]): RaycastHit[] => {
  store.raycaster.setFromCamera(ndc, store.camera)
  const meshes: Mesh[] = []
  collectMeshes(store.scene, meshes)
  return store.raycaster.intersectObjects(meshes)
}

const findInstanceHandler = (
  hit: RaycastHit,
  eventName: string,
): { instance: VoidInstance; handler: (event: any) => void } | null => {
  // Walk up from the hit object's instance to find one with the handler
  let node: any = hit.object
  while (node) {
    const inst = instanceMap.get(node)
    if (inst && inst.eventHandlers[eventName]) {
      return { instance: inst, handler: inst.eventHandlers[eventName] }
    }
    node = node.parent
  }
  return null
}

const createSyntheticEvent = (hit: RaycastHit, nativeEvent: PointerEvent | MouseEvent): VoidEvent => {
  const event: VoidEvent = {
    object: hit.object,
    point: hit.point,
    normal: hit.normal,
    distance: hit.distance,
    uv: hit.uv,
    nativeEvent,
    stopPropagation: () => {
      ;(event as any)._stopped = true
    },
  }
  return event
}

export const setupEvents = (store: VoidStore): (() => void) => {
  const canvas = store.canvas

  const onPointerMove = (e: PointerEvent) => {
    store.pointer.x = ((e.clientX - canvas.getBoundingClientRect().left) / canvas.clientWidth) * 2 - 1
    store.pointer.y = -((e.clientY - canvas.getBoundingClientRect().top) / canvas.clientHeight) * 2 + 1

    const ndc = toNDC(e, canvas)
    const hits = raycast(store, ndc)

    let newHovered: VoidInstance | null = null
    if (hits.length > 0) {
      const hit = hits[0]!
      const found = findInstanceHandler(hit, 'onPointerOver')
      if (found) newHovered = found.instance

      // Dispatch onPointerMove
      const moveHandler = findInstanceHandler(hit, 'onPointerMove')
      if (moveHandler) {
        moveHandler.handler(createSyntheticEvent(hit, e))
      }
    }

    // Enter/Leave detection
    if (newHovered !== hoveredInstance) {
      if (hoveredInstance && hoveredInstance.eventHandlers.onPointerOut) {
        hoveredInstance.eventHandlers.onPointerOut({ nativeEvent: e })
      }
      if (newHovered && newHovered.eventHandlers.onPointerOver && hits.length > 0) {
        newHovered.eventHandlers.onPointerOver(createSyntheticEvent(hits[0]!, e))
      }
      hoveredInstance = newHovered
    }
  }

  const handlePointerEvent = (eventName: string) => (e: PointerEvent | MouseEvent) => {
    const ndc = toNDC(e, canvas)
    const hits = raycast(store, ndc)
    if (hits.length > 0) {
      const hit = hits[0]!
      const found = findInstanceHandler(hit, eventName)
      if (found) {
        found.handler(createSyntheticEvent(hit, e))
      }
    }
  }

  const onClick = handlePointerEvent('onClick')
  const onPointerDown = handlePointerEvent('onPointerDown')
  const onPointerUp = handlePointerEvent('onPointerUp')

  canvas.addEventListener('pointermove', onPointerMove)
  canvas.addEventListener('click', onClick)
  canvas.addEventListener('pointerdown', onPointerDown)
  canvas.addEventListener('pointerup', onPointerUp)

  return () => {
    canvas.removeEventListener('pointermove', onPointerMove)
    canvas.removeEventListener('click', onClick)
    canvas.removeEventListener('pointerdown', onPointerDown)
    canvas.removeEventListener('pointerup', onPointerUp)
    hoveredInstance = null
  }
}
