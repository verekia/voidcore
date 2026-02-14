import { useEffect, useRef } from 'react'

import { main } from '../src/main'

const IndexPage = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    main(canvas)
  }, [])

  return <canvas ref={canvasRef} />
}

export default IndexPage
