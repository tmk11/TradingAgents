import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// During ``npm run dev`` the Vite server runs on 5173 and proxies
// /api requests to the FastAPI backend on 8765. Production builds
// are served by FastAPI directly so no proxy is needed there.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8765',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: false,
  },
})
