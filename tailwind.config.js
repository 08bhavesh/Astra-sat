/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        astra: {
          bg: 'var(--astra-bg)',
          bgSecondary: 'var(--astra-bg-secondary)',
          sidebar: 'var(--astra-sidebar)',
          card: 'var(--astra-card)',
          elevated: 'var(--astra-elevated)',
          border: 'var(--astra-border)',
          teal: 'var(--astra-teal)',
          cyan: 'var(--astra-cyan)',
          blue: 'var(--astra-blue)',
          green: 'var(--astra-green)',
          amber: 'var(--astra-amber)',
          red: 'var(--astra-red)',
          text: 'var(--astra-text)',
          textSecondary: 'var(--astra-text-secondary)',
          textMuted: 'var(--astra-text-muted)',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'IBM Plex Mono', 'monospace'],
      },
    },
  },
  plugins: [],
};
