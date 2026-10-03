// Unit tests for src/features.js.
// The most important one checks that JavaScript gives the same numbers as Python.
import { readFileSync } from 'node:fs';
import { describe, expect, test } from 'vitest';
import {
  FEATURE_SIZE,
  NUM_FRAMES,
  assignHands,
  clipToFeatures,
  fillMissingPose,
  median,
  resample,
  trimRange,
} from '../src/features.js';

const fixture = JSON.parse(
  readFileSync(new URL('./fixtures/feature_parity.json', import.meta.url), 'utf-8'),
);

describe('same numbers as the Python code', () => {
  fixture.cases.forEach((testCase, index) => {
    test(`case ${index}: ${testCase.frames.length} frames, aspect ${testCase.aspect}`, () => {
      const features = clipToFeatures(testCase.frames, testCase.aspect, testCase.trim);
      if (testCase.expected === null) {
        expect(features).toBeNull();
        return;
      }
      expect(features.length).toBe(NUM_FRAMES * FEATURE_SIZE);
      let largestDifference = 0;
      for (let i = 0; i < features.length; i++) {
        largestDifference = Math.max(largestDifference, Math.abs(features[i] - testCase.expected[i]));
      }
      expect(largestDifference).toBeLessThan(1e-4);
    });
  });
});

describe('small pieces', () => {
  test('median', () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 3, 2])).toBe(2.5);
  });

  test('missing poses are filled from neighbours', () => {
    const a = [[0, 0]];
    const b = [[1, 1]];
    expect(fillMissingPose([null, a, null, b, null])).toEqual([a, a, a, b, b]);
    expect(fillMissingPose([null, null])).toBeNull();
  });

  test('a hand goes to the nearest pose wrist', () => {
    // pose list: nose, ears, shoulders, elbows, then left wrist (index 7) and right wrist (index 8)
    const pose = [[0.5, 0.2], [0.55, 0.2], [0.45, 0.2], [0.6, 0.4], [0.4, 0.4], [0.65, 0.6], [0.35, 0.6], [0.7, 0.5], [0.3, 0.8]];
    const nearLeft = [[0.71, 0.5, 0]];
    const nearRight = [[0.31, 0.8, 0]];
    expect(assignHands([nearLeft], pose)).toEqual([nearLeft, null]);
    expect(assignHands([nearRight], pose)).toEqual([null, nearRight]);
    expect(assignHands([nearRight, nearLeft], pose)).toEqual([nearLeft, nearRight]);
    expect(assignHands([], pose)).toEqual([null, null]);
  });

  test('trimming cuts off a fast raise and a fast lowering', () => {
    // Left wrist (index 4) y per frame: rises fast for 3 frames, stays, then drops fast.
    const wristY = [0.9, 0.8, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 0.8];
    const head = [[0.5, 0.2], [0.55, 0.2], [0.45, 0.2]];
    const poses = wristY.map((y) => [...head, [0.6, 0.4], [0.4, 0.4], [0, 0], [0, 0], [0.7, y], [0.3, 0.9]]);
    const times = wristY.map((_, t) => t * 40);
    // shoulder width 0.2: 0.1 per 40 ms is 12.5 widths per second, far above the limit of 2
    const trim = { trim_speed: 2.0, trim_max_ms: 500 };
    expect(trimRange(poses, times, 0.2, trim)).toEqual([3, 8]);
    // A clip with still wrists is not trimmed at all.
    const still = wristY.map(() => poses[4]);
    expect(trimRange(still, times, 0.2, trim)).toEqual([0, 10]);
    // Trimming stops after trim_max_ms even if the wrist keeps rising.
    const rising = Array.from({ length: 30 }, (_, t) => [...head, [0.6, 0.4], [0.4, 0.4], [0, 0], [0, 0], [0.7, 2 - t * 0.05], [0.3, 0.9]]);
    const risingTimes = rising.map((_, t) => t * 40);
    expect(trimRange(rising, risingTimes, 0.2, { trim_speed: 2.0, trim_max_ms: 200 })[0]).toBe(5);
  });

  test('resample picks evenly spaced frames', () => {
    const frames = Array.from({ length: 63 }, (_, i) => i);
    const picked = resample(frames);
    expect(picked.length).toBe(NUM_FRAMES);
    expect(picked.slice(0, 3)).toEqual([0, 2, 4]);
    expect(picked[31]).toBe(62);
    expect(resample(['only'])).toEqual(Array(NUM_FRAMES).fill('only'));
  });
});
