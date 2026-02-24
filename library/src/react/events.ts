// Events – Pointer event system for VoidCore React bindings.
//
// Attaches DOM listeners to the canvas and converts pointer positions to raycasts.
// Walks hit results to find instances with event handlers (onClick, onPointerOver, etc.)
// and dispatches synthetic events with 3D intersection data.
//
// Performance: pointer move fires at display refresh rate, so the hot path avoids
// allocations by reusing a pre-allocated mesh array and NDC tuple. The canvas
// bounding rect is read once per event (not per helper call).

import { instanceMap } from './reconciler'

import type { RaycastHit } from '../raycasting/index'
import type { Mesh } from '../scene/mesh'
import type { Node } from '../scene/node'
import type { VoidStore } from './context'
import type { VoidInstance } from './reconciler'

export interface VoidEvent {
  object: any
  point: Float32Array
  normal: Float32Array
  distance: number
  uv: Float32Array | null
  nativeEvent: PointerEvent | MouseEvent
  stopPropagation: () => void
}

let hoveredInstance: VoidInstance | null = null

// Pre-allocated buffers reused every pointer event (avoids per-event allocation)
const _meshes: Mesh[] = []
const _ndc: [number, number] = [0, 0]

const collectMeshes = (node: Node, meshes: Mesh[]) => {
  if (node.type === 'mesh' && node.visible) {
    meshes.push(node as Mesh)
  }
  const children = node.children
  for (let i = 0; i < children.length; i++) {
    collectMeshes(children[i]!, meshes)
  }
}

const raycast = (store: VoidStore, ndc: [number, number]): RaycastHit[] => {
  store.raycaster.setFromCamera(ndc, store.camera)
  _meshes.length = 0
  collectMeshes(store.scene, _meshes)
  return store.raycaster.intersectObjects(_meshes)
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
    // Read bounding rect once per event (not per helper call)
    const rect = canvas.getBoundingClientRect()
    store.pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
    store.pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1

    _ndc[0] = store.pointer.x
    _ndc[1] = store.pointer.y
    const hits = raycast(store, _ndc)

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
    const rect = canvas.getBoundingClientRect()
    _ndc[0] = ((e.clientX - rect.left) / rect.width) * 2 - 1
    _ndc[1] = -((e.clientY - rect.top) / rect.height) * 2 + 1
    const hits = raycast(store, _ndc)
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
