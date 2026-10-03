// End-to-end camera test.
//
// Held-out test clips are played to the app as a fake camera. The app does everything
// it does for a real user: finds the landmarks, notices where each sign starts and
// ends, builds the features and runs the model. For each clip we compare the app's
// best word with what the Python pipeline predicted for the same pictures.
//
// Needs data/e2e/e2e.json and the videos (made by: python scripts/make_e2e_set.py).
// Writes reports/e2e_report.json.
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { expect, test } from '@playwright/test';
import { openApp } from './browser.js';

const repo = join(import.meta.dirname, '..', '..');
const batches = JSON.parse(readFileSync(join(repo, 'data/e2e/e2e.json'), 'utf-8'));

const MIN_AGREEMENT = 0.95;

test('browser predictions agree with Python on fake-camera clips', async () => {
  const rows = [];
  const pageErrors = [];
  const framesPerSecond = [];
  let delegate = null;
  let extraDetections = 0;

  for (const batch of batches) {
    // Visible window with the GPU allowed: the video plays in real time, so the app must
    // run as fast as it does for a real user, not in the slow CPU-only test mode.
    const { browser, page, errors } = await openApp(batch.video, false, true);
    // Wait until the video has played once (it would start again from the beginning).
    await page.waitForFunction(
      (duration) => performance.now() - window.__jisr.streamStartedAt >= duration,
      batch.duration_ms - 500,
      { timeout: batch.duration_ms + 120000, polling: 250 },
    );
    const results = await page.evaluate(() => window.__jisr.results);
    const speed = await page.evaluate(() => ({
      framesProcessed: window.__jisr.framesProcessed,
      seconds: (performance.now() - window.__jisr.streamStartedAt) / 1000,
      delegate: window.__jisr.delegate,
    }));
    framesPerSecond.push(speed.framesProcessed / speed.seconds);
    delegate = speed.delegate;
    await browser.close();
    pageErrors.push(...errors);

    // Match each detected sign to the clip that was playing at that moment.
    const used = new Set();
    for (const clip of batch.clips) {
      const middle = (clip.start_ms + clip.end_ms) / 2;
      const index = results.findIndex(
        (r, i) => !used.has(i) && r.startMs - 500 <= middle && middle <= r.endMs + 500,
      );
      const result = index >= 0 ? results[index] : null;
      if (index >= 0) used.add(index);

      const browserWord = result && result.candidates.length ? result.candidates[0].label : null;
      rows.push({
        sample: clip.sample,
        true_label: clip.true_label,
        python_word: clip.python.top3[0] ?? null,
        python_kind: clip.python.kind,
        browser_word: browserWord,
        browser_kind: result ? result.kind : 'not detected',
        browser_frames: result ? result.frames : 0,
        same_word: browserWord !== null && browserWord === clip.python.top3[0],
      });
    }
    extraDetections += results.length - used.size;
    const done = rows.filter((r) => r.same_word).length;
    console.log(`${rows.length} clips played, ${done} agree`);
  }

  const share = (check) => rows.filter(check).length / rows.length;
  const report = {
    clips: rows.length,
    agreement_top_word: share((r) => r.same_word),
    agreement_ok_or_unclear: share((r) => r.browser_kind === r.python_kind),
    clips_not_detected: rows.filter((r) => r.browser_kind === 'not detected').length,
    extra_detections: extraDetections,
    browser_top_word_is_true_label: share((r) => r.browser_word === r.true_label),
    python_top_word_is_true_label: share((r) => r.python_word === r.true_label),
    browser_frames_per_second: framesPerSecond.reduce((a, b) => a + b, 0) / framesPerSecond.length,
    mediapipe_delegate: delegate,
    page_errors: pageErrors,
    rows,
  };
  writeFileSync(join(repo, 'reports/e2e_report.json'), JSON.stringify(report, null, 1));
  console.log(
    `e2e: ${rows.length} clips, top-word agreement ${(report.agreement_top_word * 100).toFixed(1)}%, ` +
      `not detected ${report.clips_not_detected}, extra ${report.extra_detections}`,
  );

  expect(pageErrors).toEqual([]);
  expect(report.agreement_top_word).toBeGreaterThanOrEqual(MIN_AGREEMENT);
});
