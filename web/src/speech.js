// Speaks an Arabic sentence aloud using the browser's built-in voice
// (the Web Speech API). Nothing is sent to the internet by this file;
// we only ever pick a voice that is installed on the device.

// Find an Arabic voice that works without internet, or null if the device has none.
export function findArabicVoice() {
  if (!('speechSynthesis' in window)) return null;
  const voices = window.speechSynthesis.getVoices().filter((v) => v.lang.toLowerCase().startsWith('ar'));
  return voices.find((v) => v.localService) || null;
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

// Say the sentence. `rate` is the speed: 1 is normal, below 1 is slower.
// Returns false if the device has no Arabic voice (the app then tells the user).
export function speak(sentence, rate = 1) {
  const voice = findArabicVoice();
  if (!voice || !sentence) return false;

  window.speechSynthesis.cancel(); // stop anything still being said
  const utterance = new SpeechSynthesisUtterance(sentence);
  utterance.voice = voice;
  utterance.lang = voice.lang;
  utterance.rate = rate;
  window.speechSynthesis.speak(utterance);
  return true;
}
