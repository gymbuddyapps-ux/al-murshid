// Jisr: the main program of the app.
//
// The path of one sign through the app:
//   camera picture -> landmarks (landmarks.js) -> start/end of the sign (segmenter.js)
//   -> 32 x 140 numbers (features.js) -> 3 most likely words (classifier.js)
//   -> the best word is added (or the user picks one) -> sentence (composer.js) -> speech (speech.js)
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
import { speak, waitForVoices } from './speech.js';

// Automatic tests open the app with "?e2e=1". In that mode the app never waits for a tap,
// never speaks, and uses the CPU so results match the Python code (unless "&gpu=1" is
// added, for tests that need real-time speed).
const QUERY = new URLSearchParams(location.search);
const TEST_MODE = QUERY.has('e2e');
const PREFER_GPU = !TEST_MODE || QUERY.has('gpu');

const $ = (id) => document.getElementById(id);
const el = {
  cameraHelp: $('camera-help'),
  startButton: $('start-button'),
  startError: $('start-error'),
  video: $('video'),
  overlay: $('overlay'),
  status: $('status'),
  candidates: $('candidates'),
  candidatesTitle: $('candidates-title'),
  candidateList: $('candidate-list'),
  undoButton: $('undo-button'),
  retryButton: $('retry-button'),
  chips: $('chips'),
  clearButton: $('clear-button'),
  sentence: $('sentence'),
  voiceNotice: $('voice-notice'),
  speakButton: $('speak-button'),
  replayButton: $('replay-button'),
  autoToggle: $('auto-toggle'),
  speakWordToggle: $('speak-word-toggle'),
  skeletonToggle: $('skeleton-toggle'),
};

// ---------- What the app remembers ----------

const words = []; // the recognized words, in order
let lastSpoken = ''; // the last text that was spoken (for "replay")
let detector = null; // finds hands and body
let classifier = null; // recognizes the sign
const segmenter = new Segmenter(segmenterConfig);

// 'watching'  : looking for a sign
// 'choosing'  : the model is working, or candidates wait for a tap (manual mode)
// 'handsDown' : after a sign, wait until the hands are lowered before watching again
let mode = 'watching';
let lastVideoTime = -1;
let hideCandidatesTimer = null;

// Results of every classified sign, readable by the automatic tests.
window.__jisr = { results: [], ready: false, framesProcessed: 0, delegate: null, trace: [] };

// ---------- Small display helpers ----------

function setStatus(state, text) {
  el.status.dataset.state = state;
  el.status.textContent = text;
}

function vibrate() {
  if ('vibrate' in navigator) navigator.vibrate(40);
}

function currentRate() {
  return Number(document.querySelector('input[name="speed"]:checked').value);
}

// Say something and tell the user if the voice may need internet or does not exist.
function say(text) {
  if (TEST_MODE) return;
  const how = speak(text, currentRate());
  if (how === 'none') {
    el.voiceNotice.textContent = 'لا يوجد صوت عربي في هذا الجهاز. اعرض الجملة على الشخص الآخر ليقرأها.';
  } else if (how === 'browser') {
    el.voiceNotice.textContent = 'يُستخدم صوت المتصفح، وقد يحتاج إلى الإنترنت.';
  }
  el.voiceNotice.hidden = how === 'device';
  if (how !== 'none') {
    lastSpoken = text;
    el.replayButton.disabled = false;
  }
}

// Show the word chips and the composed sentence.
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
  el.sentence.textContent = empty ? 'الكلمات التي تُتعرَّف تظهر هنا' : sentence;
  el.sentence.classList.toggle('sentence--empty', empty);
  el.clearButton.hidden = empty;
  el.speakButton.disabled = empty;
}

// One large button per candidate word, with its confidence.
function candidateButton(candidate, onTap) {
  const percent = Math.round(candidate.confidence * 100);
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'candidate';
  button.innerHTML = `
    <span class="candidate__label"></span>
    <span class="candidate__confidence">${percent}٪</span>
    <span class="candidate__bar" style="width: ${Math.max(percent, 4)}%"></span>`;
  button.querySelector('.candidate__label').textContent = candidate.label;
  button.addEventListener('click', onTap);
  return button;
}

function hideCandidates() {
  clearTimeout(hideCandidatesTimer);
  el.candidates.hidden = true;
  segmenter.reset();
  if (mode === 'choosing') mode = 'handsDown';
}

// Add a word to the sentence (and say it, if that setting is on).
function addWord(word) {
  words.push(word);
  vibrate();
  renderSentence();
  if (el.speakWordToggle.checked) say(word);
}

// Automatic mode: the best word is added at once. The other candidates stay on screen
// for a few seconds so the user can swap the word if the model was wrong.
function autoChoose(result) {
  addWord(result.candidates[0].label);
  el.candidatesTitle.textContent = `أُضيفت «${result.candidates[0].label}». ليست هي؟ اختر البديل:`;
  el.candidateList.replaceChildren(
    ...result.candidates.slice(1).map((candidate) =>
      candidateButton(candidate, () => {
        words[words.length - 1] = candidate.label; // swap the word that was just added
        renderSentence();
        if (el.speakWordToggle.checked) say(candidate.label);
        hideCandidates();
      }),
    ),
  );
  el.undoButton.hidden = false;
  el.retryButton.hidden = true;
  el.candidates.hidden = false;
  mode = 'handsDown'; // keep watching; no tap needed
  clearTimeout(hideCandidatesTimer);
  hideCandidatesTimer = setTimeout(hideCandidates, 5000);
}

// Manual mode: show the 3 candidates and wait for a tap.
function manualChoose(result) {
  el.candidatesTitle.textContent = 'اختر الكلمة الصحيحة';
  el.candidateList.replaceChildren(
    ...result.candidates.map((candidate) =>
      candidateButton(candidate, () => {
        addWord(candidate.label);
        hideCandidates();
      }),
    ),
  );
  el.undoButton.hidden = true;
  el.retryButton.hidden = false;
  el.candidates.hidden = false;
  mode = 'choosing';
  setStatus('choose', 'اختر الكلمة الصحيحة');
}

// ---------- The camera loop ----------

// A finished clip: turn it into features, classify it, and act on the result.
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
    segmenter.reset();
    mode = 'watching';
    return;
  }
  if (result.kind === 'unclear') {
    el.candidatesTitle.textContent = 'غير واضح، أعد الإشارة';
    el.candidateList.replaceChildren();
    el.undoButton.hidden = true;
    el.retryButton.hidden = true;
    el.candidates.hidden = false;
    mode = 'handsDown';
    clearTimeout(hideCandidatesTimer);
    hideCandidatesTimer = setTimeout(hideCandidates, 2500);
    return;
  }
  if (el.autoToggle.checked) autoChoose(result);
  else manualChoose(result);
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
  window.__jisr.framesProcessed += 1;
  const aspect = video.videoWidth / video.videoHeight;
  drawSkeleton(el.overlay, frame, el.skeletonToggle.checked);

  const raised = isHandRaised(frame, aspect, segmenterConfig.raise_line);
  if (TEST_MODE) {
    // Per-frame trace for the automatic tests (what the app saw, frame by frame).
    window.__jisr.trace.push({
      t: Math.round(now - window.__jisr.streamStartedAt),
      pose: frame.pose !== null,
      hands: frame.hands.length,
      raised,
      mode,
    });
  }

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
  el.cameraHelp.hidden = true;
  setStatus('loading', 'جارٍ فتح الكاميرا');

  // The models start loading right away (from this site, and from the device's own
  // storage after the first visit), while the camera opens.
  const modelsLoading = Promise.all([LandmarkDetector.create(PREFER_GPU), SignClassifier.create()]);
  modelsLoading.catch(() => {}); // a failure is reported below

  // Tests wait for the models first, so that no part of the test video is missed.
  if (TEST_MODE) await modelsLoading.catch(() => {});

  // 1. Open the front camera.
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
  } catch {
    el.startError.textContent =
      'لم يُسمح باستخدام الكاميرا. اسمح بها من إعدادات المتصفح ثم اضغط «افتح الكاميرا».';
    el.cameraHelp.hidden = false;
    el.startButton.disabled = false;
    setStatus('warning', 'الكاميرا مغلقة');
    return;
  }

  window.__jisr.streamStartedAt = performance.now();
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

  // 3. Voices load a moment after the page opens; wait for them once.
  await waitForVoices();

  if (TEST_MODE) {
    const { runClip } = await import('./testhooks.js');
    window.__jisr.runClip = (frames, frameMs) => {
      mode = 'choosing'; // stop the camera loop so the detector is used by the test only
      return runClip(detector, classifier, frames, frameMs);
    };
  }
  window.__jisr.ready = true;
  window.__jisr.delegate = detector.delegate;
  onFrame();
}

el.startButton.addEventListener('click', start);
el.retryButton.addEventListener('click', hideCandidates);
el.undoButton.addEventListener('click', () => {
  words.pop();
  renderSentence();
  hideCandidates();
});
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

// The camera opens as soon as the page opens.
start();
