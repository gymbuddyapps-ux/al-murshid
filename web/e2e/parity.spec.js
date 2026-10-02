// Feature parity test.
//
// The same pictures go through the Python code and through the browser code.
// The features (32 x 140 numbers per clip) must be almost identical, and the model
// must give the same answer. A silent difference here is the most common way to lose
// accuracy between training and the real app.
//
// Needs data/e2e/parity.json (made by: python scripts/make_e2e_set.py).
// Writes reports/parity_report.json.
import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { expect, test } from '@playwright/test';
import { openApp } from './browser.js';

const repo = join(import.meta.dirname, '..', '..');
const cases = JSON.parse(readFileSync(join(repo, 'data/e2e/parity.json'), 'utf-8'));

// Limits. Features are measured in shoulder widths, so 0.02 is 2% of a shoulder width.
const MAX_MEAN_DIFFERENCE = 0.02;
const MIN_CLIPS_WITHIN_LIMIT = 0.95;
const MIN_SAME_ANSWER = 0.95;

const indexOfLargest = (values) => values.indexOf(Math.max(...values));

test('browser features match Python features', async () => {
  const { browser, page, errors } = await openApp();
  const rows = [];

  for (const clip of cases) {
    const frames = readdirSync(clip.folder)
      .sort()
      .map((name) => readFileSync(join(clip.folder, name)).toString('base64'));
    const out = await page.evaluate(([f, ms]) => window.__jisr.runClip(f, ms), [frames, clip.frame_ms]);

    let largest = 0;
    let sum = 0;
    for (let i = 0; i < clip.features.length; i++) {
      const difference = Math.abs(out.features[i] - clip.features[i]);
      largest = Math.max(largest, difference);
      sum += difference;
    }
    rows.push({
      sample: clip.sample,
      mean_difference: sum / clip.features.length,
      largest_difference: largest,
      same_answer: indexOfLargest(out.scores) === indexOfLargest(clip.scores),
      same_top_word: out.result.candidates[0].label === clip.python.top3[0],
    });
  }
  await browser.close();

  const share = (check) => rows.filter(check).length / rows.length;
  const report = {
    clips: rows.length,
    limit_mean_difference: MAX_MEAN_DIFFERENCE,
    clips_within_limit: share((r) => r.mean_difference <= MAX_MEAN_DIFFERENCE),
    mean_difference_over_all_clips: rows.reduce((a, r) => a + r.mean_difference, 0) / rows.length,
    largest_difference_over_all_clips: Math.max(...rows.map((r) => r.largest_difference)),
    same_answer: share((r) => r.same_answer),
    same_top_word: share((r) => r.same_top_word),
    page_errors: errors,
    rows,
  };
  writeFileSync(join(repo, 'reports/parity_report.json'), JSON.stringify(report, null, 1));
  console.log(
    `parity: ${rows.length} clips, within limit ${(report.clips_within_limit * 100).toFixed(1)}%, ` +
      `mean difference ${report.mean_difference_over_all_clips.toFixed(5)}, ` +
      `same answer ${(report.same_answer * 100).toFixed(1)}%`,
  );

  expect(errors).toEqual([]);
  expect(report.clips_within_limit).toBeGreaterThanOrEqual(MIN_CLIPS_WITHIN_LIMIT);
  expect(report.same_answer).toBeGreaterThanOrEqual(MIN_SAME_ANSWER);
});
