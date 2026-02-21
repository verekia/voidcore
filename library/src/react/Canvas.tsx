// Canvas – Root React component for VoidCore's declarative API.
//
// Creates the WebGPU/WebGL2 engine, scene, and camera on mount, then renders children
// via the custom reconciler into the scene graph. Manages the rAF loop, frame callbacks,
// pointer events, and cleanup on unmount. Also hosts an overlay div where Html components
// tunnel their DOM content via the tunnel-rat pattern.

import { useEffect, useRef, useState, useSyncExternalStore, type ReactNode } from 'react'

import { Engine } from '../engine'
import { Raycaster } from '../raycasting/index'
import { PerspectiveCamera } from '../scene/camera'
import { Scene } from '../scene/scene'
import { VoidContext, type VoidStore } from './context'
import { setupEvents } from './events'
import { reconciler, type VoidInstance } from './reconciler'
import { createTunnel, type Tunnel } from './tunnel'

import type { FloatPrecision } from '../float'
import type { ShadowConfig } from '../renderer/renderer'

export interface CanvasProps {
  children?: ReactNode
  backend?: 'auto' | 'webgpu' | 'webgl2'
  antialias?: boolean
  bloom?: { intensity?: number; levels?: number }
  shadows?: boolean | ShadowConfig
  maxFps?: number
  maxDpr?: number
  /** Float precision for math types: 'auto' tries Float16Array, falls back to Float32Array. */
  floatPrecision?: FloatPrecision | 'auto'
  camera?: {
    fov?: number
    near?: number
    far?: number
    position?: [number, number, number]
  }
  onCreated?: (state: { engine: Engine; scene: Scene; camera: PerspectiveCamera }) => void
  style?: React.CSSProperties
  className?: string
}

const noop = () => {}
const logError = (err: Error) => console.error(err)

const emptyArray: ReactNode[] = []
const getServerSnapshot = () => emptyArray

const TunnelOut = ({ tunnel }: { tunnel: Tunnel }) => {
  const items = useSyncExternalStore(tunnel.subscribe, tunnel.getSnapshot, getServerSnapshot)
  return <>{items}</>
}

export const Canvas = ({
  children,
  backend = 'auto',
  antialias,
  bloom,
  shadows,
  maxFps,
  maxDpr,
  floatPrecision,
  camera: cameraProps,
  onCreated,
  style,
  className,
}: CanvasProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [store, setStore] = useState<VoidStore | null>(null)
  const rootRef = useRef<any>(null)
  const cleanupRef = useRef<(() => void) | null>(null)
  const tunnelRef = useRef<Tunnel>(null)
  if (!tunnelRef.current) tunnelRef.current = createTunnel()

  // Initialize engine on mount
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    let disposed = false

    const init = async () => {
      const engine = await Engine.create(canvas, {
        backend,
        antialias,
        bloom,
        shadows,
        floatPrecision,
      })
      if (disposed) {
        engine.dispose()
        return
      }

      if (maxFps !== undefined) engine.maxFps = maxFps
      if (maxDpr !== undefined) engine.maxDpr = maxDpr

      const scene = new Scene()

      const camera = new PerspectiveCamera({
        fov: cameraProps?.fov ?? 60,
        near: cameraProps?.near ?? 0.1,
        far: cameraProps?.far ?? 1000,
      })
      if (cameraProps?.position) {
        camera.setPosition(cameraProps.position[0], cameraProps.position[1], cameraProps.position[2])
      }

      const raycaster = new Raycaster()

      const storeValue: VoidStore = {
        engine,
        scene,
        camera,
        canvas,
        raycaster,
        pointer: { x: 0, y: 0 },
        frameCallbacks: new Set(),
        tunnel: tunnelRef.current!,
      }

      const cleanupEvents = setupEvents(storeValue)

      const unregisterUpdate = engine.register(
        state => {
          for (const cb of storeValue.frameCallbacks) {
            cb({
              dt: state.dt,
              elapsed: state.elapsed,
              frame: state.frame,
              engine,
              scene,
              camera,
            })
          }
          scene.updateGraph()
        },
        { priority: -1 },
      )

      const unregisterRender = engine.register(
        () => {
          engine.render(scene, camera)
        },
        { priority: 0 },
      )

      // Create reconciler container
      const sceneInstance: VoidInstance = {
        object: scene,
        type: 'scene',
        parent: null,
        children: [],
        attach: null,
        previousAttachValue: undefined,
        eventHandlers: {},
      }

      rootRef.current = reconciler.createContainer(
        sceneInstance,
        0, // LegacyRoot
        null, // hydrationCallbacks
        false, // isStrictMode
        null, // concurrentUpdatesByDefaultOverride
        '', // identifierPrefix
        logError, // onUncaughtError
        logError, // onCaughtError
        logError, // onRecoverableError
        noop, // onDefaultTransitionIndicator
      )

      cleanupRef.current = () => {
        reconciler.updateContainer(null, rootRef.current, null, noop)
        rootRef.current = null
        cleanupEvents()
        unregisterUpdate()
        unregisterRender()
        engine.dispose()
      }

      setStore(storeValue)

      if (onCreated) {
        onCreated({ engine, scene, camera })
      }

      engine.start()
    }

    init()

    return () => {
      disposed = true
      cleanupRef.current?.()
      cleanupRef.current = null
      setStore(null)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Update reconciler when children or store changes
  useEffect(() => {
    if (!store || !rootRef.current) return

    reconciler.updateContainer(
      <VoidContext.Provider value={store}>{children}</VoidContext.Provider>,
      rootRef.current,
      null,
      noop,
    )
  }, [store, children])

  return (
    <div
      ref={containerRef}
      style={{ position: 'relative', width: '100%', height: '100%', ...style }}
      className={className}
    >
      <canvas ref={canvasRef} style={{ display: 'block', width: '100%', height: '100%' }} />
      <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', overflow: 'hidden' }}>
        <TunnelOut tunnel={tunnelRef.current} />
      </div>
    </div>
  )
}
