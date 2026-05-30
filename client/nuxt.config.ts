/// <reference types="nuxt" />

// CHORDAI_BACKEND_URL is set at build/runtime to point at the Go server.
// Defaults:
//   local dev  → http://localhost:5555   (Go runs on the host)
//   Docker     → http://server:5555      (Docker Compose service name)
//   cluster    → http://c6-4:5555        (UiT GPU node)
const backendUrl = process.env.CHORDAI_BACKEND_URL ?? 'http://localhost:5555'

export default defineNuxtConfig({
  compatibilityDate: '2024-11-01',
  srcDir: 'app/',
  devtools: { enabled: false },
  modules: ['@nuxtjs/tailwindcss'],

  runtimeConfig: {
    public: {
      serverUrl: backendUrl,
    },
  },

  nitro: {
    devProxy: {
      '/api': {
        target: `${backendUrl}/api`,
        changeOrigin: true,
        prependPath: true,
      },
      '/ws': {
        target: `${backendUrl}/ws`,
        ws: true,
        changeOrigin: true,
        prependPath: true,
      },
    },
  },
})