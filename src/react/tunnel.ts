// Tunnel – Bridges React elements between VoidCore's custom reconciler and react-dom.
//
// Html components inside the 3D scene graph register DOM content via tunnel.set().
// The Canvas component renders a TunnelOut that displays all registered content
// in the react-dom overlay div. Inspired by pmndrs/tunnel-rat.

import type { ReactNode } from 'react'

type Listener = () => void

export interface Tunnel {
  set: (id: string, node: ReactNode) => void
  delete: (id: string) => void
  subscribe: (cb: Listener) => () => void
  getSnapshot: () => ReactNode[]
}

export const createTunnel = (): Tunnel => {
  const entries = new Map<string, ReactNode>()
  const listeners = new Set<Listener>()
  let snapshot: ReactNode[] = []

  const notify = () => {
    snapshot = [...entries.values()]
    for (const l of listeners) l()
  }

  return {
    set(id: string, node: ReactNode) {
      entries.set(id, node)
      notify()
    },
    delete(id: string) {
      entries.delete(id)
      notify()
    },
    subscribe(cb: Listener) {
      listeners.add(cb)
      return () => {
        listeners.delete(cb)
      }
    },
    getSnapshot: () => snapshot,
  }
}
