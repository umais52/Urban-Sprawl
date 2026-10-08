import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  resolve: {
    alias: [
      { find: /^react-map-gl$/, replacement: 'react-map-gl/mapbox' },
    ],
  },
  plugins: [
    tailwindcss(),
    react(),
  ],
})
