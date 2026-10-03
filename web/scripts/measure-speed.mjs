// Measure how many camera frames per second the app processes, in four setups:
// visible or invisible browser window, CPU-only or GPU allowed. Needs the preview server.
//   node scripts/measure-speed.mjs
import { chromium } from '@playwright/test';

for (const headless of [true, false]) {
  for (const gpu of [false, true]) {
    const browser = await chromium.launch({
      channel: 'msedge',
      headless,
      args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'],
    });
    const page = await browser.newPage();
    await page.goto(`http://localhost:4173/?e2e=1${gpu ? '&gpu=1' : ''}`);
    await page.waitForFunction(() => window.__jisr?.ready, null, { timeout: 120000 });
    const before = await page.evaluate(() => window.__jisr.framesProcessed);
    await page.waitForTimeout(10000);
    const after = await page.evaluate(() => window.__jisr.framesProcessed);
    const delegate = await page.evaluate(() => window.__jisr.delegate);
    console.log(`headless=${headless} gpu=${gpu} delegate=${delegate}: ${((after - before) / 10).toFixed(1)} frames/s`);
    await browser.close();
  }
}
