import '../tailwind.css'
import { useGLTF, useKTX2 } from 'voidcore'

import type { AppProps } from 'next/app'

useGLTF.setDecoderPath('/draco-1.5.7/')
useKTX2.setTranscoderPath('/basis-1.50/')

export default function App({ Component, pageProps }: AppProps) {
  return <Component {...pageProps} />
}
