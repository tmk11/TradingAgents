/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Gold-edition palette — warm tones to nudge the visual
        // identity away from generic finance dashboards.
        gold: {
          50: '#fdf8e7',
          100: '#fbeec1',
          200: '#f7d97a',
          300: '#f4c43d',
          400: '#e9ad14',
          500: '#c89110',
          600: '#9c700b',
          700: '#705007',
          800: '#443005',
          900: '#241803',
        },
      },
      fontFamily: {
        sans: ['ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'monospace'],
      },
    },
  },
  plugins: [],
}
