import { useEffect, useState } from 'react'

import type { Engine } from 'voidcore'

const StatsOverlay = ({ engine }: { engine: Engine | null }) => {
  const [statsText, setStatsText] = useState('')

  useEffect(() => {
    if (!engine) return
    const interval = setInterval(() => {
      const stats = engine.getStats()
      const shadowSuffix = stats.shadowDrawCalls > 0 ? ` (+${stats.shadowDrawCalls} for shadows)` : ''
      const dpr = Math.min(window.devicePixelRatio, engine.maxDpr)
      const floatLabel = engine.floatPrecision === 'float16' ? 'F16' : 'F32'
      setStatsText(
        `${engine.backend.toUpperCase()} | ${floatLabel} | ${Math.round(stats.fps)} FPS | DPR: ${dpr.toFixed(2)} | Draw calls: ${stats.drawCalls}${shadowSuffix}`,
      )
    }, 500)
    return () => clearInterval(interval)
  }, [engine])

  if (!statsText) return null

  return (
    <div className="pointer-events-none fixed top-2.5 left-2.5 z-1000 rounded bg-black/60 px-3 py-2 font-mono text-sm text-white">
      {statsText}
    </div>
  )
}

export default StatsOverlay
