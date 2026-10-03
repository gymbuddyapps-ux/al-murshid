// Shared helper for the browser tests: open the app in test mode with a fake camera.
import { chromium } from '@playwright/test';

// `videoFile` is a .y4m file that Chromium plays as if it were the camera.
// Without it, Chromium shows its built-in test pattern.
// `headless = false` opens a visible window; Edge's invisible mode has no service worker,
// so the offline test needs a visible one.
// `gpu = true` lets the app use the graphics chip like it does for real users (the parity
// test keeps the CPU so that its numbers can be compared with Python exactly).
export async function openApp(videoFile = null, headless = true, gpu = false) {
  const args = ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'];
  if (videoFile) args.push(`--use-file-for-fake-video-capture=${videoFile}`);

  const browser = await chromium.launch({ channel: 'msedge', headless, args });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));

  // "?e2e=1" starts the camera without a tap and does not wait for confirmations.
  await page.goto(`http://localhost:4174/?e2e=1${gpu ? '&gpu=1' : ''}`);
  await page.waitForFunction(() => window.__jisr?.ready, null, { timeout: 120000 });
  return { browser, page, errors };
}
