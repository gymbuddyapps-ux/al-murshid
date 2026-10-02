// Jisr: the main program of the app.
//
// The path of one sign through the app:
//   camera picture -> landmarks (landmarks.js) -> start/end of the sign (segmenter.js)
//   -> 32 x 140 numbers (features.js) -> 3 most likely words (classifier.js)
//   -> the user confirms one -> sentence (composer.js) -> speech (speech.js)
//
// Everything runs on this device. No picture or data is sent anywhere.
import './style.css';
import segmenterConfig from '../../config/segmenter.json';
import templates from '../../config/templates.json';
import { SignClassifier } from './classifier.js';
import { composeSentence } from './composer.js';
import { clipToFeatures } from './features.js';
import { LandmarkDetector } from './landmarks.js';
import { drawSkeleton } from './overlay.js';
import { Segmenter, isHandRaised } from './segmenter.js';
import { findArabicVoice, speak, waitForVoices } from './speech.js';

// Automatic tests open the app with "?e2e=1". In that mode the app does not wait
// for a tap after each sign, and it uses the CPU so results match the Python code.
const TEST_MODE = new URLSearchParams(location.search).has('e2e');

const $ = (id) => document.getElementById(id);
const el = {
  welcome: $('welcome'),
  app: $('app'),
  startButton: $('start-button'),
  startError: $('start-error'),
  video: $('video'),
  overlay: $('overlay'),
  status: $('status'),
  skeletonToggle: $('skeleton-toggle'),
  candidates: $('candidates'),
  candidatesTitle: $('candidates-title'),
  candidateList: $('candidate-list'),
  retryButton: $('retry-button'),
  chips: $('chips'),
  clearButton: $('clear-button'),
  sentence: $('sentence'),
  voiceNotice: $('voice-notice'),
  speakButton: $('speak-button'),
  replayButton: $('replay-button'),
};

// ---------- What the app remembers ----------

const words = []; // the confirmed words, in order
let lastSpoken = ''; // the last sentence that was spoken (for "replay")
let detector = null; // finds hands and body
let classifier = null; // recognizes the sign
const segmenter = new Segmenter(segmenterConfig);

// 'watching'  : looking for a sign
// 'choosing'  : candidates are shown, waiting for a tap
// 'handsDown' : after a tap, wait until the hands are lowered before watching again
let mode = 'watching';
let lastVideoTime = -1;

// Results of every classified sign, readable by the automatic tests.
window.__jisr = { results: [], ready: false };

// ---------- Small display helpers ----------

function setStatus(state, text) {
  el.status.dataset.state = state;
  el.status.textContent = text;
}

function vibrate() {
  if ('vibrate' in navigator) navigator.vibrate(40);
}

// Show the sentence strip (chips) and the composed sentence.
function renderSentence() {
  el.chips.replaceChildren(
    ...words.map((word, index) => {
      const chip = document.createElement('li');
      chip.className = 'chip';
      chip.append(word);
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.textContent = '×';
      remove.setAttribute('aria-label', `احذف ${word}`);
      remove.addEventListener('click', () => {
        words.splice(index, 1);
        renderSentence();
      });
      chip.append(remove);
      return chip;
    }),
  );

  const sentence = composeSentence(words, templates);
  const empty = words.length === 0;
  el.sentence.textContent = empty ? 'الكلمات التي تؤكدها تظهر هنا' : sentence;
  el.sentence.classList.toggle('sentence--empty', empty);
  el.clearButton.hidden = empty;
  el.speakButton.disabled = empty;
}

// Show the top 3 candidates as large buttons, or the "unclear" message.
function showCandidates(result) {
  el.candidateList.replaceChildren();

  if (result.kind === 'unclear') {
    el.candidatesTitle.textContent = 'غير واضح، أعد الإشارة';
  } else {
    el.candidatesTitle.textContent = 'اختر الكلمة الصحيحة';
    for (const candidate of result.candidates) {
      const percent = Math.round(candidate.confidence * 100);
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'candidate';
      button.innerHTML = `
        <span class="candidate__label"></span>
        <span class="candidate__confidence">${percent}٪</span>
        <span class="candidate__bar" style="width: ${Math.max(percent, 4)}%"></span>`;
      button.querySelector('.candidate__label').textContent = candidate.label;
      button.addEventListener('click', () => confirmWord(candidate.label));
      el.candidateList.append(button);
    }
  }
  el.candidates.hidden = false;
  mode = 'choosing';
  setStatus('choose', result.kind === 'unclear' ? 'غير واضح' : 'اختر الكلمة الصحيحة');
}

function hideCandidates() {
  el.candidates.hidden = true;
  segmenter.reset();
  mode = 'handsDown';
}

function confirmWord(word) {
  words.push(word);
  vibrate();
  renderSentence();
  hideCandidates();
}

// ---------- The camera loop ----------

// A finished clip: turn it into features, classify it, and show the result.
async function handleClip(frames, aspect) {
  setStatus('choose', 'جارٍ التعرّف');
  mode = 'choosing'; // stop watching while the model works

  const features = clipToFeatures(frames, aspect, segmenterConfig);
  // `features` is null when no hand was found in the clip; that counts as unclear too.
  const result = features ? await classifier.classify(features) : { kind: 'unclear', candidates: [] };
  window.__jisr.results.push({
    ...result,
    frames: frames.length,
    // when the clip started and ended, counted from the moment the camera started
    startMs: frames[0].time - window.__jisr.streamStartedAt,
    endMs: frames[frames.length - 1].time - window.__jisr.streamStartedAt,
  });

  if (TEST_MODE) {
    // A test is running: do not wait for a tap, go straight back to watching.
    segmenter.reset();
    mode = 'watching';
    return;
  }
  showCandidates(result);
}

// Runs once for every new camera picture.
function onFrame() {
  requestAnimationFrame(onFrame);
  const video = el.video;
  if (video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;
  if (mode === 'choosing') return;

  const now = performance.now();
  const frame = { ...detector.detect(video, now), time: now };
  const aspect = video.videoWidth / video.videoHeight;
  drawSkeleton(el.overlay, frame, el.skeletonToggle.checked);

  const raised = isHandRaised(frame, aspect, segmenterConfig.raise_line);

  if (mode === 'handsDown') {
    if (raised) return setStatus('ready', 'أنزل يديك');
    mode = 'watching';
  }

  const result = segmenter.push(frame, now, raised);
  if (result.state === 'finished') {
    handleClip(result.frames, aspect);
  } else if (result.state === 'signing') {
    setStatus('signing', 'جارٍ التقاط الإشارة');
  } else if (frame.pose === null) {
    setStatus('warning', 'أظهر كتفيك داخل الصورة');
  } else {
    setStatus('ready', 'جاهز، ارفع يدك لتبدأ');
  }
}

// ---------- Starting the app ----------

async function start() {
  el.startButton.disabled = true;
  el.startError.hidden = true;

  // The models start loading right away (from this site, and from the device's own
  // storage after the first visit), while the user answers the camera question.
  const modelsLoading = Promise.all([LandmarkDetector.create(!TEST_MODE), SignClassifier.create()]);
  modelsLoading.catch(() => {}); // a failure is reported below, after the camera opens

  // Tests wait for the models first, so that no part of the test video is missed.
  if (TEST_MODE) await modelsLoading.catch(() => {});

  // 1. Ask for the camera.
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
  } catch {
    el.startError.textContent =
      'لم يُسمح باستخدام الكاميرا. اسمح بها من إعدادات المتصفح ثم اضغط «افتح الكاميرا» مرة أخرى.';
    el.startError.hidden = false;
    el.startButton.disabled = false;
    return;
  }

  window.__jisr.streamStartedAt = performance.now();
  el.welcome.hidden = true;
  el.app.hidden = false;
  setStatus('loading', 'جارٍ التحميل');

  el.video.srcObject = stream;
  await el.video.play();
  // Give the camera box and the drawing layer the camera's real shape.
  el.overlay.width = el.video.videoWidth;
  el.overlay.height = el.video.videoHeight;
  el.video.parentElement.style.aspectRatio = `${el.video.videoWidth} / ${el.video.videoHeight}`;

  // 2. Wait until the models are loaded.
  try {
    [detector, classifier] = await modelsLoading;
  } catch (error) {
    console.error(error);
    setStatus('warning', 'تعذّر تحميل النموذج. أعد فتح التطبيق وأنت متصل بالإنترنت.');
    return;
  }

  // 3. Check for an Arabic voice. Without one, the large text is the way to communicate.
  await waitForVoices();
  el.voiceNotice.hidden = findArabicVoice() !== null;

  if (TEST_MODE) {
    const { runClip } = await import('./testhooks.js');
    window.__jisr.runClip = (frames, frameMs) => runClip(classifier, frames, frameMs);
  }
  window.__jisr.ready = true;
  onFrame();
}

function currentRate() {
  return Number(document.querySelector('input[name="speed"]:checked').value);
}

function say(sentence) {
  const spoken = speak(sentence, currentRate());
  el.voiceNotice.hidden = spoken;
  if (spoken) {
    lastSpoken = sentence;
    el.replayButton.disabled = false;
  }
}

el.startButton.addEventListener('click', start);
el.retryButton.addEventListener('click', hideCandidates);
el.clearButton.addEventListener('click', () => {
  words.length = 0;
  renderSentence();
});
el.speakButton.addEventListener('click', () => say(composeSentence(words, templates)));
el.replayButton.addEventListener('click', () => say(lastSpoken));

renderSentence();

// Save the app on the device so it works without internet (only in the built app).
if ('serviceWorker' in navigator && import.meta.env.PROD) {
  navigator.serviceWorker.register('/sw.js');
}

// Tests start the camera without a tap.
if (TEST_MODE) start();
