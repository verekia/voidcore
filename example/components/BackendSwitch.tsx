import type { Engine } from 'voidcore'

const BackendSwitch = ({ engine }: { engine: Engine | null }) => {
  if (!engine) return null

  const isWebGPU = engine.backend === 'webgpu'

  return (
    <a
      href={isWebGPU ? '?webgl=1' : '/'}
      className="fixed top-2.5 right-2.5 z-[1000] cursor-pointer rounded bg-black/60 px-3 py-2 font-mono text-sm text-white underline"
    >
      Switch to {isWebGPU ? 'WebGL2' : 'WebGPU'}
    </a>
  )
}

export default BackendSwitch
