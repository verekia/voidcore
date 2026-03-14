// DirectionalLightHelper (React) – Declarative wrapper around the imperative helper.
//
// Adds the shadow volume mesh directly to the scene and updates it each frame.
// Accepts a React ref to the directional light so the helper can read it at frame time
// (after the reconciler has committed the ref).
//
// Usage:
//   const lightRef = useRef<DirectionalLight>(null)
//   <directionalLight ref={lightRef} ... />
//   <DirectionalLightHelper lightRef={lightRef} />

import { useEffect, useRef } from 'react'
import type { RefObject } from 'react'

import { DirectionalLightHelper as HelperImpl } from '../helpers/directional-light-helper'
import { useEngine, useFrame } from './hooks'

import type { DirectionalLight } from '../scene/light'

export interface DirectionalLightHelperProps {
  lightRef: RefObject<DirectionalLight | null>
  color?: [number, number, number]
  opacity?: number
}

export const DirectionalLightHelper = ({ lightRef, color, opacity }: DirectionalLightHelperProps) => {
  const { scene } = useEngine()
  const helperRef = useRef<HelperImpl | null>(null)
  // Serialize array to avoid re-creating on every render due to new array references
  const colorKey = color ? `${color[0]},${color[1]},${color[2]}` : ''

  useEffect(() => {
    const helper = new HelperImpl({ color, opacity })
    helperRef.current = helper
    scene.add(helper.mesh)
    return () => {
      scene.remove(helper.mesh)
      helper.dispose()
      helperRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene, colorKey, opacity])

  useFrame(() => {
    const light = lightRef.current
    if (helperRef.current && light) {
      helperRef.current.update(light)
    }
  })

  return null
}
