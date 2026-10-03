// The conversation: what the signer said (through the app) and what the other person typed.
//
// Messages are kept in memory and, as a convenience, in the browser's local storage so the
// history survives a reload on the same device. Nothing leaves the device.

const STORAGE_KEY = 'jisr-chat';
const MAX_MESSAGES = 200;

// A message is { who: 'signer' | 'other', text, time } where time is milliseconds since 1970.
export function createChat(storage = null) {
  let messages = [];
  if (storage) {
    try {
      messages = JSON.parse(storage.getItem(STORAGE_KEY) || '[]');
    } catch {
      messages = [];
    }
  }

  const save = () => {
    if (!storage) return;
    try {
      storage.setItem(STORAGE_KEY, JSON.stringify(messages));
    } catch {
      /* storage may be unavailable (private mode); the chat still works for this visit */
    }
  };

  return {
    all: () => messages.slice(),
    add(who, text, time = Date.now()) {
      const clean = text.trim();
      if (!clean) return null;
      const message = { who, text: clean, time };
      messages = [...messages, message].slice(-MAX_MESSAGES);
      save();
      return message;
    },
    clear() {
      messages = [];
      save();
    },
  };
}

// "٠٩:٤٥" style time for a message.
export function formatTime(time, locale = 'ar-EG') {
  return new Date(time).toLocaleTimeString(locale, { hour: '2-digit', minute: '2-digit' });
}
