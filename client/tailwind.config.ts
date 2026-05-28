import type { Config } from 'tailwindcss'

export default {
  content: [
    './app/**/*.{vue,ts,js}',
    './pages/**/*.{vue,ts,js}',
    './components/**/*.{vue,ts,js}',
    './composables/**/*.{ts,js}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      keyframes: {
        eq1: { '0%, 100%': { height: '6px' },  '40%': { height: '28px' } },
        eq2: { '0%, 100%': { height: '20px' }, '55%': { height: '8px'  } },
        eq3: { '0%, 100%': { height: '10px' }, '35%': { height: '26px' } },
        eq4: { '0%, 100%': { height: '24px' }, '60%': { height: '6px'  } },
        eq5: { '0%, 100%': { height: '8px'  }, '45%': { height: '22px' } },
      },
      animation: {
        eq1: 'eq1 1.4s ease-in-out infinite',
        eq2: 'eq2 1.8s ease-in-out infinite',
        eq3: 'eq3 1.2s ease-in-out infinite',
        eq4: 'eq4 2.0s ease-in-out infinite',
        eq5: 'eq5 1.6s ease-in-out infinite',
      },
    },
  },
} satisfies Config
