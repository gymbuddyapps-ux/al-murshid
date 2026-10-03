// Unit tests for the sentence composer, using the real rules in config/templates.json.
import { readFileSync } from 'node:fs';
import { describe, expect, test } from 'vitest';
import { composeSentence } from '../src/composer.js';

const readJson = (path) => JSON.parse(readFileSync(new URL(path, import.meta.url), 'utf-8'));
const templates = readJson('../../config/templates.json');
const vocab = readJson('../../config/vocab.json');

describe('the demo sentences agreed with the project owner', () => {
  test('1. greeting the judges and introducing oneself', () => {
    // the greeting is one sign (KArSL makes both greetings the same way); signing it twice is fine too
    const words = ['السلام عليكم', 'بنت', 'سعيد', 'طموح', 'مهندس'];
    expect(composeSentence(words, templates)).toBe(
      'السلام عليكم وأهلاً وسهلاً بكم، أنا بنت سعيدة وطموحة، أتمنى أن أكون مهندسة',
    );
  });

  test('1b. the greeting signed twice reads the same', () => {
    expect(composeSentence(['السلام عليكم', 'السلام عليكم', 'بنت', 'سعيد'], templates)).toBe(
      'السلام عليكم وأهلاً وسهلاً بكم، أنا بنت سعيدة',
    );
  });

  test('2. explaining the project', () => {
    const words = ['يفكر', 'يبني', 'مترجم لغة الإشارة', 'ذكي', 'يساعد', 'يدعم', 'ناس', 'إعاقة سمعية'];
    expect(composeSentence(words, templates)).toBe(
      'فكرت أن أبني مترجم لغة إشارة ذكياً، يساعد ويدعم الناس من ذوي الإعاقة السمعية',
    );
  });

  test('3. thanking family and teachers', () => {
    const words = ['الحمد لله', 'يحب', 'أسرة', 'أب', 'أم', 'معلم', 'شكراً'];
    expect(composeSentence(words, templates)).toBe('الحمد لله، أنا أحب أسرتي، أبي وأمي ومعلميّ، شكراً لكم');
  });

  test('4. welcoming a friend', () => {
    const words = ['السلام عليكم', 'صديق', 'ضيف', 'تفضل'];
    expect(composeSentence(words, templates)).toBe('أهلاً وسهلاً بصديقي وضيفي، تفضل');
  });
});

describe('partial sentences still read well', () => {
  test('one or two traits', () => {
    expect(composeSentence(['سعيد'], templates)).toBe('أنا سعيدة');
    expect(composeSentence(['بنت', 'ذكي'], templates)).toBe('أنا بنت ذكية');
  });

  test('loving one person', () => {
    expect(composeSentence(['يحب', 'أم'], templates)).toBe('أنا أحب أمي');
    expect(composeSentence(['يحب', 'صديق', 'معلم'], templates)).toBe('أنا أحب صديقي ومعلمي');
  });

  test('a person alone', () => {
    expect(composeSentence(['أب'], templates)).toBe('هذا أبي');
  });

  test('help alone is a request', () => {
    expect(composeSentence(['يساعد'], templates)).toBe('أحتاج إلى مساعدة');
  });

  test('words without a rule are kept as they are', () => {
    expect(composeSentence(['الحمد لله'], templates)).toBe('الحمد لله');
    expect(composeSentence([], templates)).toBe('');
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

  test('every word with forms is in config/vocab.json', () => {
    for (const word of Object.keys(templates.forms)) expect(known.has(word), word).toBe(true);
  });
});
