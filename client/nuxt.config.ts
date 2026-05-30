export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',
  devtools: { enabled: false },
  modules: ['@nuxtjs/tailwindcss'],
  
  runtimeConfig: {
    public: {
      serverUrl: 'http://localhost:8000',
    },
  },

  nitro: {
    devProxy: {
      '/api': {
        target: 'http://c6-4:5555/api',
        changeOrigin: true,
        prependPath: true
      },
      '/ws': {
        target: 'http://c6-4:5555/ws',
        ws: true,
        changeOrigin: true,
        prependPath: true
      },
      '/verify-chords': {
        target: 'http://c6-4:8001/verify-chords',
        changeOrigin: true,
        prependPath: true
      }
    }
  }
})
