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

// Keep the voice list fresh: browsers add voices a little after the page opens.
if ('speechSynthesis' in window) {
  window.speechSynthesis.addEventListener('voiceschanged', () => window.speechSynthesis.getVoices());
}

// How speech is available right now.
//   'device'      an Arabic voice installed on the device (works offline)
//   'browser'     the browser's own Arabic voice (may need internet)
//   'unknown'     no Arabic voice is listed; the browser may still speak with its default voice
//   'unsupported' this browser cannot speak at all
export function speechStatus() {
  if (!('speechSynthesis' in window)) return 'unsupported';
  const [voice] = arabicVoices();
  if (!voice) return 'unknown';
  return voice.localService ? 'device' : 'browser';
}

// Say the text. `rate` is the speed: 1 is normal, below 1 is slower.
// Returns the speechStatus() value that applied.
export function speak(text, rate = 1) {
  if (!('speechSynthesis' in window) || !text) return 'unsupported';

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
  return speechStatus();
}
