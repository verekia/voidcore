import { defineConfig } from 'tsup'

export default defineConfig({
  entry: ['src/index.ts', 'src/workers/geometry-worker.ts'],
  clean: true,
  format: ['esm'],
  dts: true,
  splitting: false,
})
