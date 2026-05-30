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
        target: 'http://c6-4:5555/api', // Sender nå til Go på GPU-noden
        changeOrigin: true,
        prependPath: true
      },
      '/ws': {
        target: 'http://c6-4:5555/ws',  // Sender nå til Go på GPU-noden
        ws: true,
        changeOrigin: true,
        prependPath: true
      }
    }
  }
})