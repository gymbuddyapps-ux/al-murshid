// Unit tests for the conversation history (src/chat.js).
import { describe, expect, test } from 'vitest';
import { createChat, formatTime } from '../src/chat.js';

// A tiny stand-in for localStorage.
function fakeStorage() {
  const map = new Map();
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, v),
  };
}

describe('createChat', () => {
  test('adds messages in order with who and time', () => {
    const chat = createChat();
    chat.add('signer', 'السلام عليكم', 1000);
    chat.add('other', 'وعليكم السلام', 2000);
    expect(chat.all().map((m) => m.who)).toEqual(['signer', 'other']);
    expect(chat.all()[1].text).toBe('وعليكم السلام');
  });

  test('ignores empty text', () => {
    const chat = createChat();
    expect(chat.add('other', '   ')).toBeNull();
    expect(chat.all()).toEqual([]);
  });

  test('remembers the history in storage and can clear it', () => {
    const storage = fakeStorage();
    createChat(storage).add('signer', 'شكراً لكم', 5);
    const again = createChat(storage);
    expect(again.all()).toEqual([{ who: 'signer', text: 'شكراً لكم', time: 5 }]);
    again.clear();
    expect(createChat(storage).all()).toEqual([]);
  });

  test('keeps only the last 200 messages', () => {
    const chat = createChat();
    for (let i = 0; i < 250; i++) chat.add('other', `رسالة ${i}`, i);
    expect(chat.all().length).toBe(200);
    expect(chat.all()[0].text).toBe('رسالة 50');
  });

  test('a broken storage value does not crash', () => {
    const storage = fakeStorage();
    storage.setItem('jisr-chat', '{not json');
    expect(createChat(storage).all()).toEqual([]);
  });
});

test('formatTime gives hours and minutes', () => {
  expect(formatTime(new Date(2026, 0, 1, 9, 5).getTime(), 'en-GB')).toMatch(/09:05|9:05/);
});
