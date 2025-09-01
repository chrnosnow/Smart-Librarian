import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server:{
      proxy:{
          // requests to any path starting with /api will be forwarded to the FastAPI server
          '/api':{
//            target: 'http://localhost:8000',    // the address of the FastAPI server
            target: 'http://api:8000',
            changeOrigin: true,  // recommended for virtual hosted sites
          },
      },
  },
})
