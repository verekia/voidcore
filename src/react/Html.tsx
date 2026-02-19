// Html – Renders React DOM content as an overlay positioned at a 3D world coordinate.
//
// Returns a <group> to the custom reconciler for scene graph participation (inherits
// parent transforms), then tunnels DOM children to Canvas's react-dom overlay div.
// Each frame, projects the group's world position through the camera VP matrix and
// updates the wrapper div's CSS transform. Inspired by pmndrs/tunnel-rat.

import { useRef, useEffect, useId, useContext, type ReactNode, type CSSProperties } from 'react'

import { mat4Multiply, vec4Create, vec4Set, vec4TransformMat4, mat4Create } from '../math/index.ts'
import { VoidContext } from './context.ts'
import { useFrame } from './hooks.ts'

export interface HtmlProps {
  position?: [number, number, number]
  center?: boolean
  style?: CSSProperties
  className?: string
  children?: ReactNode
}

export const Html = ({ position = [0, 0, 0], center = false, style, className, children }: HtmlProps) => {
  const store = useContext(VoidContext)!
  const groupRef = useRef<any>(null)
  const divRef = useRef<HTMLDivElement>(null)
  const id = useId()

  const { camera, canvas, tunnel } = store

  // Pre-allocate math buffers
  const vpMatrix = useRef(mat4Create()).current
  const tempVec4 = useRef(vec4Create()).current

  // Register/update tunneled DOM content
  useEffect(() => {
    tunnel.set(
      id,
      <div
        key={id}
        ref={divRef}
        className={className}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          willChange: 'transform',
          pointerEvents: 'auto',
          display: 'none',
          ...style,
        }}
      >
        {children}
      </div>,
    )
    return () => tunnel.delete(id)
  }, [id, tunnel, children, className, style])

  // Project group world position to screen CSS every frame
  useFrame(() => {
    const el = divRef.current
    const group = groupRef.current
    if (!el || !group) return

    // Read world position from group's world matrix
    const wm = group._worldMatrix
    const wx = wm[12]
    const wy = wm[13]
    const wz = wm[14]

    mat4Multiply(vpMatrix, camera._projectionMatrix, camera._viewMatrix)
    vec4Set(tempVec4, wx, wy, wz, 1)
    vec4TransformMat4(tempVec4, tempVec4, vpMatrix)

    const w = tempVec4[3]!
    if (w <= 0) {
      el.style.display = 'none'
      return
    }

    const ndcX = tempVec4[0]! / w
    const ndcY = tempVec4[1]! / w

    if (Math.abs(ndcX) > 1.1 || Math.abs(ndcY) > 1.1) {
      el.style.display = 'none'
      return
    }

    el.style.display = ''
    const screenX = (ndcX * 0.5 + 0.5) * canvas.clientWidth
    const screenY = (1 - (ndcY * 0.5 + 0.5)) * canvas.clientHeight

    let transform = `translate(${screenX}px,${screenY}px)`
    if (center) transform += ' translate(-50%,-50%)'
    el.style.transform = transform
  })

  return <group ref={groupRef} position={position} />
}
