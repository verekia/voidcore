import type { Engine } from '../src/index'

const FloatSwitch = ({ engine }: { engine: Engine | null }) => {
  if (!engine) return null

  const isF16 = engine.floatPrecision === 'float16'

  const toggle = () => {
    const params = new URLSearchParams(window.location.search)
    if (isF16) {
      params.set('f32', '1')
    } else {
      params.delete('f32')
    }
    const qs = params.toString()
    window.location.href = qs ? `?${qs}` : window.location.pathname
  }

  return (
    <button
      onClick={toggle}
      className="fixed top-12 right-2.5 z-[1000] cursor-pointer rounded bg-black/60 px-3 py-2 font-mono text-sm text-white underline"
    >
      Switch to {isF16 ? 'F32' : 'F16'}
    </button>
  )
}

export default FloatSwitch
