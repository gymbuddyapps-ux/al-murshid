// Unit tests for sign detection (src/segmenter.js).
import { describe, expect, test } from 'vitest';
import { Segmenter, isHandRaised } from '../src/segmenter.js';

const config = { raise_line: 1.0, start_frames: 3, end_ms: 400, min_ms: 300, max_ms: 6000 };
const FRAME_MS = 40; // 25 frames per second

// Feed a list of true/false ("is a hand raised?") to a segmenter, one frame every 40 ms.
// Returns the list of results; each frame is just its own index so clips are easy to check.
function run(raisedList, segmenter = new Segmenter(config)) {
  return raisedList.map((raised, i) => segmenter.push(i, i * FRAME_MS, raised));
}
const repeat = (value, count) => Array(count).fill(value);

describe('Segmenter', () => {
  test('stays idle when no hand is raised', () => {
    const results = run(repeat(false, 50));
    expect(results.every((r) => r.state === 'idle')).toBe(true);
  });

  test('starts only after a few raised frames in a row', () => {
    const results = run([false, true, true, true, true]);
    expect(results.map((r) => r.state)).toEqual(['idle', 'idle', 'idle', 'signing', 'signing']);
  });

  test('a short flicker does not start a sign', () => {
    const results = run([true, true, false, true, false, false]);
    expect(results.every((r) => r.state === 'idle')).toBe(true);
  });

  test('ends about 0.4 seconds after the hands drop, and returns only the raised part', () => {
    // 5 idle frames, 25 raised frames (1 second), then hands down.
    const results = run([...repeat(false, 5), ...repeat(true, 25), ...repeat(false, 15)]);
    const finishedAt = results.findIndex((r) => r.state === 'finished');
    // Last raised frame is index 29; 400 ms later is 10 frames after it.
    expect(finishedAt).toBe(39);
    expect(results[finishedAt].frames).toEqual(Array.from({ length: 25 }, (_, i) => i + 5));
    expect(results[finishedAt + 1].state).toBe('idle');
  });

  test('a short gap inside a sign does not end it', () => {
    // hands disappear for 5 frames (200 ms) in the middle
    const results = run([...repeat(true, 10), ...repeat(false, 5), ...repeat(true, 10), ...repeat(false, 12)]);
    const finished = results.filter((r) => r.state === 'finished');
    expect(finished.length).toBe(1);
    expect(finished[0].frames.length).toBe(25);
  });

  test('a very short sign is ignored', () => {
    // 4 raised frames = 120 ms between first and last, less than min_ms
    const results = run([...repeat(true, 4), ...repeat(false, 15)]);
    expect(results.some((r) => r.state === 'finished')).toBe(false);
    expect(results[results.length - 1].state).toBe('idle');
  });

  test('a very long sign is cut at the maximum length', () => {
    const results = run(repeat(true, 200));
    const finishedAt = results.findIndex((r) => r.state === 'finished');
    expect(finishedAt).toBe(config.max_ms / FRAME_MS);
  });

  test('two signs in a row give two clips', () => {
    const oneSign = [...repeat(true, 20), ...repeat(false, 15)];
    const results = run([...oneSign, ...oneSign]);
    expect(results.filter((r) => r.state === 'finished').length).toBe(2);
  });
});

describe('isHandRaised', () => {
  // shoulders at y = 0.4, 0.2 apart; the line is one shoulder-width lower, at y = 0.6
  const pose = [[0.6, 0.4], [0.4, 0.4], [0, 0], [0, 0], [0, 0], [0, 0]];
  const handAt = (y) => [[0.5, y, 0]];

  test('a hand above the line is raised', () => {
    expect(isHandRaised({ pose, hands: [handAt(0.5)] }, 1.0, 1.0)).toBe(true);
  });

  test('a hand below the line is not raised', () => {
    expect(isHandRaised({ pose, hands: [handAt(0.7)] }, 1.0, 1.0)).toBe(false);
  });

  test('one raised hand out of two is enough', () => {
    expect(isHandRaised({ pose, hands: [handAt(0.9), handAt(0.3)] }, 1.0, 1.0)).toBe(true);
  });

  test('no hands or no pose means not raised', () => {
    expect(isHandRaised({ pose, hands: [] }, 1.0, 1.0)).toBe(false);
    expect(isHandRaised({ pose: null, hands: [handAt(0.3)] }, 1.0, 1.0)).toBe(false);
  });

  test('the line follows the picture shape', () => {
    // In a picture twice as wide, the same shoulders are 0.1 apart in x fractions.
    const widePose = [[0.55, 0.4], [0.45, 0.4], [0, 0], [0, 0], [0, 0], [0, 0]];
    expect(isHandRaised({ pose: widePose, hands: [handAt(0.55)] }, 2.0, 1.0)).toBe(true);
    expect(isHandRaised({ pose: widePose, hands: [handAt(0.65)] }, 2.0, 1.0)).toBe(false);
  });
});
