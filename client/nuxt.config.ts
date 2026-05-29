export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',
  devtools: { enabled: false },
  modules: ['@nuxtjs/tailwindcss'],
  
  runtimeConfig: {
    public: {
      serverUrl: 'http://localhost:8000',
    },
  },

  // nitro: {
  //   devProxy: {
  //     '/api': {
  //       target: 'http://localhost:5555/api',
  //       changeOrigin: true,
  //       prependPath: true
  //     },
  //     // Siden loggene dine viste "/ws/generate", har jeg inkludert denne for WebSockets:
  //     '/ws': {
  //       target: 'http://localhost:5555/ws',
  //       ws: true,
  //       changeOrigin: true
  //     }
  //   }
  // }


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


// nitro: {
//     devProxy: {
//       '/api': {
//         target: 'http://c6-4:5555/api', // Sender nå til Go på GPU-noden
//         changeOrigin: true,
//         prependPath: false
//       },
//       '/ws': {
//         target: 'http://c6-4:5555/ws',  // Sender nå til Go på GPU-noden
//         ws: true,
//         changeOrigin: true,
//         prependPath: false
//       }
//     }
//   }

// export default defineNuxtConfig({
//   compatibilityDate: '2025-07-15',
//   devtools: { enabled: false },
//   modules: ['@nuxtjs/tailwindcss'],
  
//   runtimeConfig: {
//     public: {
//       serverUrl: 'http://c6-4:8000', 
//     },
//   },

//   nitro: {
//     devProxy: {
//       '/api': {
//         target: 'http://c6-4:8000/api', // Trafikken rutes internt fra c0-0 til c6-4
//         changeOrigin: true,
//         prependPath: true
//       },
//       '/ws': {
//         target: 'http://c6-4:8000/ws',
//         ws: true,
//         changeOrigin: true
//       }
//     }
//   }
// })