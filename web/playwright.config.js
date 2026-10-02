// Settings for the browser tests in web/e2e/.
// They run the built app in a real Chromium browser (Microsoft Edge, which is
// already installed on Windows, so nothing has to be downloaded).
import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: 'e2e',
  timeout: 60 * 60 * 1000, // the camera test plays real-time video, so it takes a while
  workers: 1,
  reporter: 'list',
  use: {
    baseURL: 'http://localhost:4173',
  },
  webServer: {
    command: 'npm run build && npx vite preview --port 4173 --strictPort',
    url: 'http://localhost:4173',
    reuseExistingServer: true,
    timeout: 180 * 1000,
  },
});
