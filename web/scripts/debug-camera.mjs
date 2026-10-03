// Play one fake-camera video and print what the app saw frame by frame (for debugging).
//   node scripts/debug-camera.mjs <video.y4m> <duration_ms> [gpu]
import { chromium } from '@playwright/test';
const [video, duration, gpu] = process.argv.slice(2);
const browser = await chromium.launch({ channel: 'msedge', headless: false, args: [
  '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', `--use-file-for-fake-video-capture=${video}`] });
const page = await browser.newPage();
page.on('console', (m) => { if (m.type() === 'error') console.log('console:', m.text().slice(0, 200)); });
await page.goto(`http://localhost:4173/?e2e=1${gpu ? '&gpu=1' : ''}`);
await page.waitForFunction(() => window.__jisr?.ready, null, { timeout: 120000 });
await page.waitForTimeout(Number(duration));
const data = await page.evaluate(() => ({ trace: window.__jisr.trace, results: window.__jisr.results, delegate: window.__jisr.delegate, video: [document.getElementById('video').videoWidth, document.getElementById('video').videoHeight] }));
await browser.close();
console.log('delegate', data.delegate, 'video', data.video, 'frames', data.trace.length, 'results', data.results.length);
// summarize: runs of (pose,hands,raised)
let last = null, start = 0, count = 0, prevT = 0;
for (const f of [...data.trace, null]) {
  const key = f ? `pose=${f.pose} hands=${f.hands} raised=${f.raised} mode=${f.mode}` : null;
  if (key !== last) { if (last !== null) console.log(`${start}-${prevT}ms (${count} frames): ${last}`); last = key; start = f?.t; count = 0; }
  if (f) { count++; prevT = f.t; }
}
console.log(JSON.stringify(data.results.map((r) => ({ s: r.startMs, e: r.endMs, n: r.frames, k: r.kind, w: r.candidates[0]?.label }))));
