// Settings for the browser tests in web/e2e/.
// They run the already built app (run `npm run build` first) in a real Chromium browser (Microsoft Edge, which is
// already installed on Windows, so nothing has to be downloaded), on port 4174 so they do not
// clash with the phone preview (`npm run preview:phone`, HTTPS on 4173).
import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: 'e2e',
  timeout: 60 * 60 * 1000, // the camera test plays real-time video, so it takes a while
  workers: 1,
  reporter: 'list',
  use: {
    baseURL: 'http://localhost:4174',
  },
  webServer: {
    command: 'npx vite preview --port 4174 --strictPort',
    url: 'http://localhost:4174',
    reuseExistingServer: true,
    timeout: 180 * 1000,
  },
});
