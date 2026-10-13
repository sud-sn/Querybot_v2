// The portal's Vue pieces, built into static/dist (served as /static/dist, versioned by content by asset()).
// The built files are kept in the repository: the server never needs Node. After a change under src, run
// `npm run build` here and commit static/dist with it (tests/test_the_built_frontend_matches_its_sources.py).
import { fileURLToPath } from 'node:url'
import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [vue()],
  define: {
    __VUE_OPTIONS_API__: 'false',
    __VUE_PROD_DEVTOOLS__: 'false',
    __VUE_PROD_HYDRATION_MISMATCH_DETAILS__: 'false',
  },
  build: {
    outDir: fileURLToPath(new URL('../static/dist', import.meta.url)),
    emptyOutDir: true,
    target: 'es2020',
    sourcemap: false,
    // One script per page, named as the page loads it; anything shared between pages in its own file.
    rolldownOptions: {
      input: { dashboard: fileURLToPath(new URL('./src/dashboard/main.js', import.meta.url)) },
      output: {
        entryFileNames: '[name].js',
        chunkFileNames: 'chunks/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash][extname]',
      },
    },
  },
})
