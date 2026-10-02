// Unit tests for the sentence composer, using the real rules in config/templates.json.
import { readFileSync } from 'node:fs';
import { describe, expect, test } from 'vitest';
import { composeSentence } from '../src/composer.js';

const readJson = (path) => JSON.parse(readFileSync(new URL(path, import.meta.url), 'utf-8'));
const templates = readJson('../../config/templates.json');
const vocab = readJson('../../config/vocab.json');

describe('composeSentence with the real templates', () => {
  test('nothing confirmed gives an empty sentence', () => {
    expect(composeSentence([], templates)).toBe('');
  });

  test('medicine + symptom', () => {
    expect(composeSentence(['دواء', 'صداع'], templates)).toBe('أحتاج إلى دواء للصداع');
    expect(composeSentence(['مرهم', 'حساسية'], templates)).toBe('أحتاج إلى مرهم للحساسية');
  });

  test('a symptom alone', () => {
    expect(composeSentence(['حمى'], templates)).toBe('عندي حمى');
  });

  test('two symptoms', () => {
    expect(composeSentence(['حمى', 'صداع'], templates)).toBe('عندي حمى وصداع');
  });

  test('a medicine alone', () => {
    expect(composeSentence(['قطارة'], templates)).toBe('أحتاج إلى قطارة');
  });

  test('a place becomes a question', () => {
    expect(composeSentence(['صيدلية'], templates)).toBe('أين الصيدلية؟');
    expect(composeSentence(['مستشفى'], templates)).toBe('أين المستشفى؟');
  });

  test('verbs and doctor', () => {
    expect(composeSentence(['يساعد'], templates)).toBe('أحتاج إلى مساعدة');
    expect(composeSentence(['يشرب'], templates)).toBe('أريد أن أشرب');
    expect(composeSentence(['طبيب'], templates)).toBe('أحتاج إلى طبيب');
  });

  test('a full pharmacy sentence', () => {
    const words = ['السلام عليكم', 'دواء', 'صداع', 'شكراً'];
    expect(composeSentence(words, templates)).toBe('السلام عليكم، أحتاج إلى دواء للصداع، شكراً');
  });

  test('words without a rule are kept as they are', () => {
    expect(composeSentence(['شكراً'], templates)).toBe('شكراً');
    expect(composeSentence(['كلمة غير معروفة', 'شكراً'], templates)).toBe('كلمة غير معروفة، شكراً');
  });
});

describe('composeSentence basics', () => {
  test('with no rules, words are joined', () => {
    expect(composeSentence(['أ', 'ب'], { joiner: ' ' })).toBe('أ ب');
  });

  test('a missing word form falls back to the plain word', () => {
    const tiny = { rules: [{ pattern: ['<x>'], output: 'نحو {0.definite}' }], groups: { x: ['بيت'] } };
    expect(composeSentence(['بيت'], tiny)).toBe('نحو بيت');
  });
});

describe('templates only use words from the vocabulary', () => {
  const known = new Set(vocab.classes.map((c) => c.arabic));

  test('every word in a group is in config/vocab.json', () => {
    for (const words of Object.values(templates.groups)) {
      for (const word of words) expect(known.has(word), word).toBe(true);
    }
  });

  test('every exact word in a rule pattern is in config/vocab.json', () => {
    for (const rule of templates.rules) {
      for (const item of rule.pattern) {
        if (!item.startsWith('<')) expect(known.has(item), item).toBe(true);
      }
    }
  });
});
