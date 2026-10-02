// Unit tests for src/confidence.js (probabilities, top 3, and the "unclear" rule).
import { describe, expect, test } from 'vitest';
import { interpret, softmax } from '../src/confidence.js';

describe('softmax', () => {
  test('probabilities add up to 1 and keep the order of the scores', () => {
    const probabilities = softmax([2, 1, 0]);
    expect(probabilities.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    expect(probabilities[0]).toBeGreaterThan(probabilities[1]);
    expect(probabilities[1]).toBeGreaterThan(probabilities[2]);
  });

  test('a higher temperature makes the model less sure', () => {
    expect(softmax([4, 0], 2)[0]).toBeLessThan(softmax([4, 0], 1)[0]);
  });

  test('very large scores do not break it', () => {
    expect(softmax([1000, 0])[0]).toBeCloseTo(1);
  });
});

describe('interpret', () => {
  const labels = ['دواء', 'صداع', 'شكراً', 'حمى', 'إشارة أخرى'];
  const other = 4;

  test('a confident answer gives the top 3 words, best first', () => {
    const result = interpret([0.7, 0.15, 0.05, 0.08, 0.02], labels, other, 0.5);
    expect(result.kind).toBe('ok');
    expect(result.candidates.map((c) => c.label)).toEqual(['دواء', 'صداع', 'حمى']);
    expect(result.candidates[0].confidence).toBeCloseTo(0.7);
  });

  test('below the threshold the answer is unclear', () => {
    const result = interpret([0.4, 0.3, 0.2, 0.05, 0.05], labels, other, 0.5);
    expect(result.kind).toBe('unclear');
  });

  test('when "other sign" is the most likely, the answer is unclear', () => {
    const result = interpret([0.3, 0.1, 0.05, 0.05, 0.5], labels, other, 0.2);
    expect(result.kind).toBe('unclear');
  });

  test('"other sign" is never offered as a candidate', () => {
    const result = interpret([0.5, 0.05, 0.03, 0.02, 0.4], labels, other, 0.3);
    expect(result.candidates.map((c) => c.label)).not.toContain('إشارة أخرى');
    expect(result.candidates.length).toBe(3);
  });
});
