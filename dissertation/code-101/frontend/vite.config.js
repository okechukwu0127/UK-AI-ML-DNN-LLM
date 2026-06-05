import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:5100',
      '/health': 'http://localhost:5100',
      '/detect_single': 'http://localhost:5100',
      '/batch_detect': 'http://localhost:5100',
      '/dataset_batch_detect': 'http://localhost:5100',
      '/admin': 'http://localhost:5100',
    },
  },
});
