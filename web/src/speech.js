// Speaks Arabic aloud using the browser's built-in voices (the Web Speech API).
//
// Voices installed on the device are preferred (they work offline). If the device has
// none, the browser's own Arabic voice is used, which on some computers needs internet;
// the app then shows a small notice. If the device has no Arabic voice at all, the large
// text is the fallback.

// All Arabic voices the browser knows about, device voices first.
export function arabicVoices() {
  if (!('speechSynthesis' in window)) return [];
  const voices = window.speechSynthesis.getVoices().filter((v) => v.lang.toLowerCase().startsWith('ar'));
  return [...voices.filter((v) => v.localService), ...voices.filter((v) => !v.localService)];
}

// Voices load a little after the page opens. This waits for them (at most 2 seconds).
export function waitForVoices() {
  return new Promise((resolve) => {
    if (!('speechSynthesis' in window)) return resolve();
    if (window.speechSynthesis.getVoices().length > 0) return resolve();
    const timer = setTimeout(resolve, 2000);
    window.speechSynthesis.addEventListener(
      'voiceschanged',
      () => {
        clearTimeout(timer);
        resolve();
      },
      { once: true },
    );
  });
}

// Say the text. `rate` is the speed: 1 is normal, below 1 is slower.
// Returns 'device' (offline voice), 'browser' (may need internet), or 'none' (no speech possible).
export function speak(text, rate = 1) {
  if (!('speechSynthesis' in window) || !text) return 'none';

  window.speechSynthesis.cancel(); // stop anything still being said
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'ar-SA';
  utterance.rate = rate;

  const [voice] = arabicVoices();
  if (voice) {
    utterance.voice = voice;
    utterance.lang = voice.lang;
  }
  window.speechSynthesis.speak(utterance);
  return voice ? (voice.localService ? 'device' : 'browser') : 'none';
}
