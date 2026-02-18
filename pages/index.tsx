import { useEffect, useRef } from 'react'

import { main } from '../src/main'

const IndexPage = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    let engine: any = null
    main(canvas).then(e => {
      engine = e
    })

    return () => {
      engine?.dispose()
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: '100vw',
        height: '100vh',
        display: 'block',
        position: 'fixed',
        top: 0,
        left: 0,
      }}
    />
  )
}

export default IndexPage
