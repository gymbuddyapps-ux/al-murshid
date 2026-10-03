// Take screenshots of the app for a visual check (phone and desktop sizes).
// Uses a fake camera, so no real camera is needed.
//
// Usage (with the app running on http://localhost:4173):
//   node scripts/screenshot.mjs <output folder>
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from '@playwright/test';

const outDir = process.argv[2] || 'screenshots';
mkdirSync(outDir, { recursive: true });

const browser = await chromium.launch({
  channel: 'msedge',
  ignoreHTTPSErrors: true,
  args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'],
});

for (const [name, viewport] of [
  ['phone', { width: 390, height: 844 }],
  ['desktop', { width: 1280, height: 800 }],
]) {
  const context = await browser.newContext({ viewport, permissions: ['camera'], ignoreHTTPSErrors: true });
  const page = await context.newPage();
  page.on('console', (message) => {
    if (message.type() === 'error') console.log(`[${name}] console error:`, message.text());
  });
  page.on('pageerror', (error) => console.log(`[${name}] page error:`, error.message));

  await page.goto('https://localhost:4173/');
  await page
    .waitForFunction(() => window.__jisr?.ready, null, { timeout: 60000 })
    .catch(() => console.log(`[${name}] app did not become ready`));
  await page.waitForTimeout(1500);
  console.log(`[${name}] status:`, await page.textContent('#status'));
  await page.screenshot({ path: join(outDir, `${name}-camera.png`), fullPage: true });
  await context.close();
}
await browser.close();
