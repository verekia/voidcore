// BakeShadows – Freezes the shadow map so it is not re-rendered every frame.
//
// Mount this component to bake (freeze) the shadow map after the first render.
// Unmount it to resume real-time shadow updates. Useful for static scenes where
// neither the light nor shadow-casting objects move.
//
// Usage:
//   <BakeShadows />

import { useEffect } from 'react'

import { useEngine } from './hooks'

export const BakeShadows = () => {
  const { engine } = useEngine()
  useEffect(() => {
    engine.shadowsBaked = true
    return () => {
      engine.shadowsBaked = false
    }
  }, [engine])
  return null
}
