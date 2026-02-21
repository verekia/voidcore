// Reconciler – Custom React reconciler for VoidCore's scene graph.
//
// Uses react-reconciler to map React elements to engine objects (Mesh, Group, DirectionalLight,
// geometry, material, etc.). Supports mutation mode: elements are created, appended to parents,
// removed, and updated as React's diffing algorithm dictates.
//
// Key concepts:
//   VoidInstance – Wrapper around an engine object that tracks parent/child/attach relationships.
//   catalogue    – Maps JSX tag names to engine constructors.
//   applyProps   – Sets engine properties from React props (position, rotation, color, etc.).

import Reconciler from 'react-reconciler'
import { DefaultEventPriority } from 'react-reconciler/constants.js'

import { catalogue, GEOMETRY_TYPES, MATERIAL_TYPES } from './types'

// ─── Instance wrapper ─────────────────────────────────────────────────────────

export interface VoidInstance {
  object: any
  type: string
  parent: VoidInstance | null
  children: VoidInstance[]
  attach: string | null
  previousAttachValue: any
  eventHandlers: Record<string, (event: any) => void>
}

// Reverse lookup: engine object → VoidInstance (for events)
export const instanceMap = new WeakMap<object, VoidInstance>()

// ─── Props helpers ────────────────────────────────────────────────────────────

const EVENT_PROPS = new Set([
  'onClick',
  'onPointerOver',
  'onPointerOut',
  'onPointerDown',
  'onPointerUp',
  'onPointerMove',
])

const RESERVED_PROPS = new Set(['key', 'ref', 'children', 'attach', 'args', 'object'])

export const applyProps = (
  instance: VoidInstance,
  newProps: Record<string, any>,
  oldProps: Record<string, any> = {},
) => {
  const obj = instance.object

  for (const key of Object.keys(newProps)) {
    if (RESERVED_PROPS.has(key)) continue

    const newVal = newProps[key]
    const oldVal = oldProps[key]

    // Skip unchanged values
    if (newVal === oldVal) continue

    // Event handlers
    if (EVENT_PROPS.has(key)) {
      if (newVal) {
        instance.eventHandlers[key] = newVal
      } else {
        delete instance.eventHandlers[key]
      }
      continue
    }

    // Transform arrays: use setters which handle dirty flags automatically
    if (key === 'position' && Array.isArray(newVal)) {
      obj.setPosition(newVal[0], newVal[1], newVal[2])
      continue
    }

    if (key === 'rotation' && Array.isArray(newVal)) {
      obj.setRotation(newVal[0], newVal[1], newVal[2], newVal[3])
      continue
    }

    if (key === 'scale') {
      if (typeof newVal === 'number') {
        obj.setScale(newVal)
      } else if (Array.isArray(newVal)) {
        obj.setScale(newVal[0], newVal[1], newVal[2])
      }
      continue
    }

    // Color arrays on materials/lights
    if (key === 'color' && Array.isArray(newVal)) {
      obj.color = newVal
      if (obj.needsUpdate !== undefined) obj.needsUpdate = true
      continue
    }

    // Direct assignments: visible, castShadow, receiveShadow, name, intensity, fov, near, far, etc.
    if (key in obj) {
      obj[key] = newVal
      if (obj.needsUpdate !== undefined && (key === 'vertexColors' || key === 'opacity' || key === 'transparent')) {
        obj.needsUpdate = true
      }
      if (key === 'fov' || key === 'near' || key === 'far') {
        obj._projectionDirty = true
      }
    }
  }

  // Remove old event handlers that aren't in new props
  for (const key of Object.keys(oldProps)) {
    if (EVENT_PROPS.has(key) && !(key in newProps)) {
      delete instance.eventHandlers[key]
    }
  }
}

// ─── Child management ─────────────────────────────────────────────────────────

const appendChild = (parent: VoidInstance, child: VoidInstance) => {
  child.parent = parent
  parent.children.push(child)

  if (child.attach) {
    child.previousAttachValue = parent.object[child.attach]
    parent.object[child.attach] = child.object
    if (parent.object.needsUpdate !== undefined) parent.object.needsUpdate = true
  } else if (parent.object.add) {
    parent.object.add(child.object)
  }
}

const removeChild = (parent: VoidInstance, child: VoidInstance) => {
  const idx = parent.children.indexOf(child)
  if (idx !== -1) parent.children.splice(idx, 1)
  child.parent = null

  if (child.attach) {
    parent.object[child.attach] = child.previousAttachValue
    if (parent.object.needsUpdate !== undefined) parent.object.needsUpdate = true
  } else if (parent.object.remove) {
    parent.object.remove(child.object)
  }

  // Dispose geometries and materials
  disposeRecursive(child)
}

const disposeRecursive = (instance: VoidInstance) => {
  if (instance.object.dispose) {
    instance.object.dispose()
  }
  for (const child of instance.children) {
    disposeRecursive(child)
  }
}

// ─── Reconciler config ────────────────────────────────────────────────────────

// The reconciler host config uses `as any` because the @types/react-reconciler types
// require many platform-specific methods that aren't relevant for a 3D scene graph renderer.
export const reconciler = Reconciler({
  // Required React 19 fields
  NotPendingTransition: null as any,
  HostTransitionContext: { $$typeof: Symbol.for('react.context'), _currentValue: null } as any,
  resetFormInstance() {},
  requestPostPaintCallback() {},
  shouldAttemptEagerTransition() {
    return false
  },
  trackSchedulerEvent() {},
  resolveEventType() {
    return null
  },
  resolveEventTimeStamp() {
    return Date.now()
  },
  maySuspendCommit() {
    return false
  },
  preloadInstance() {
    return true
  },
  startSuspendingCommit() {},
  suspendInstance() {},
  waitForCommitToBeReady() {
    return null
  },

  supportsMutation: true,
  supportsPersistence: false,
  supportsHydration: false,

  isPrimaryRenderer: false,

  createInstance(type: string, props: Record<string, any>) {
    let object: any

    if (type === 'primitive') {
      object = props.object
      if (!object) throw new Error('<primitive> requires an `object` prop')
    } else {
      const Ctor = catalogue[type]
      if (!Ctor) throw new Error(`Unknown element type: ${type}`)

      const args = props.args ?? []
      object = new Ctor(...args)
    }

    // Determine default attach
    let attach = props.attach ?? null
    if (!attach) {
      if (GEOMETRY_TYPES.has(type)) attach = 'geometry'
      else if (MATERIAL_TYPES.has(type)) attach = 'material'
    }

    const instance: VoidInstance = {
      object,
      type,
      parent: null,
      children: [],
      attach,
      previousAttachValue: undefined,
      eventHandlers: {},
    }

    instanceMap.set(object, instance)
    applyProps(instance, props)

    return instance
  },

  createTextInstance() {
    throw new Error('Text is not supported in VoidCore scene graph. Wrap text in an <Html> component.')
  },

  appendInitialChild(parent: VoidInstance, child: VoidInstance) {
    appendChild(parent, child)
  },

  appendChild(parent: VoidInstance, child: VoidInstance) {
    appendChild(parent, child)
  },

  appendChildToContainer(container: VoidInstance, child: VoidInstance) {
    appendChild(container, child)
  },

  removeChild(parent: VoidInstance, child: VoidInstance) {
    removeChild(parent, child)
  },

  removeChildFromContainer(container: VoidInstance, child: VoidInstance) {
    removeChild(container, child)
  },

  insertBefore(parent: VoidInstance, child: VoidInstance, _before: VoidInstance) {
    appendChild(parent, child)
  },

  insertInContainerBefore(container: VoidInstance, child: VoidInstance, _before: VoidInstance) {
    appendChild(container, child)
  },

  commitUpdate(instance: VoidInstance, _type: string, oldProps: Record<string, any>, newProps: Record<string, any>) {
    applyProps(instance, newProps, oldProps)
  },

  finalizeInitialChildren() {
    return false
  },

  prepareForCommit() {
    return null
  },

  resetAfterCommit() {},

  getPublicInstance(instance: VoidInstance) {
    return instance.object
  },

  getRootHostContext() {
    return {}
  },

  getChildHostContext(parentContext: any) {
    return parentContext
  },

  shouldSetTextContent() {
    return false
  },

  hideInstance(instance: VoidInstance) {
    instance.object.visible = false
  },

  unhideInstance(instance: VoidInstance) {
    instance.object.visible = true
  },

  hideTextInstance() {},
  unhideTextInstance() {},

  clearContainer() {},

  scheduleTimeout: setTimeout,
  cancelTimeout: clearTimeout,
  noTimeout: -1,

  getInstanceFromNode() {
    return null
  },

  beforeActiveInstanceBlur() {},
  afterActiveInstanceBlur() {},
  prepareScopeUpdate() {},
  getInstanceFromScope() {
    return null
  },
  detachDeletedInstance() {},

  preparePortalMount() {},

  setCurrentUpdatePriority() {},
  getCurrentUpdatePriority() {
    return DefaultEventPriority
  },
  resolveUpdatePriority() {
    return DefaultEventPriority
  },
})

// Enable concurrent features
reconciler.injectIntoDevTools({
  // @ts-ignore
  bundleType: process.env.NODE_ENV === 'production' ? 0 : 1,
  rendererPackageName: 'voidcore',
  version: '0.0.0',
})
