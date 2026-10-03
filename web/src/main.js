// Jisr: the main program of the app.
//
// Two parts: "التعرّف" (recognition, shown as a conversation) and "تعلّم الإشارات" (learning).
//
// The path of one sign through the recognition part:
//   camera picture -> landmarks (landmarks.js) -> start/end of the sign (segmenter.js)
//   -> 32 x 146 numbers (features.js) -> 3 most likely words (classifier.js)
//   -> the best word is added (or the user picks one) -> sentence (composer.js)
//   -> "انطق وأرسل": spoken (speech.js) and added to the conversation (chat.js)
//
// Everything runs on this device. No picture or data is sent anywhere.
import './style.css';
import segmenterConfig from '../../config/segmenter.json';
import templates from '../../config/templates.json';
import { createChat, formatTime } from './chat.js';
import { SignClassifier } from './classifier.js';
import { composeSentence } from './composer.js';
import { clipToFeatures } from './features.js';
import { LandmarkDetector } from './landmarks.js';
import { fillLearnPage } from './learn.js';
import { drawSkeleton } from './overlay.js';
import { Segmenter, isHandRaised } from './segmenter.js';
import { arabicVoices, speak, speechStatus, waitForVoices } from './speech.js';

// Automatic tests open the app with "?e2e=1". In that mode the app never waits for a tap,
// never speaks, and uses the CPU so results match the Python code (unless "&gpu=1" is
// added, for tests that need real-time speed).
const QUERY = new URLSearchParams(location.search);
const TEST_MODE = QUERY.has('e2e');
const PREFER_GPU = !TEST_MODE || QUERY.has('gpu');

const $ = (id) => document.getElementById(id);
const el = {
  views: { recognize: $('view-recognize'), learn: $('view-learn') },
  tabs: [...document.querySelectorAll('.tab')],
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
  sentence: $('sentence'),
  speakButton: $('speak-button'),
  clearButton: $('clear-button'),
  voiceNotice: $('voice-notice'),
  replayButton: $('replay-button'),
  clearChatButton: $('clear-chat-button'),
  messages: $('messages'),
  replyForm: $('reply-form'),
  replyInput: $('reply-input'),
  autoToggle: $('auto-toggle'),
  speakWordToggle: $('speak-word-toggle'),
  skeletonToggle: $('skeleton-toggle'),
  signGrid: $('sign-grid'),
  testVoiceButton: $('test-voice-button'),
  voiceList: $('voice-list'),
};

// ---------- What the app remembers ----------

const words = []; // the words of the sentence being built right now
let lastSpoken = ''; // the last text that was spoken (for "replay")
let detector = null; // finds hands and body
let classifier = null; // recognizes the sign
const segmenter = new Segmenter(segmenterConfig);
const chat = createChat(TEST_MODE ? null : safeLocalStorage());

// 'watching'  : looking for a sign
// 'choosing'  : the model is working, or candidates wait for a tap (manual mode)
// 'handsDown' : after a sign, wait until the hands are lowered before watching again
// 'paused'    : the learning page is open
let mode = 'watching';
let lastVideoTime = -1;
let hideCandidatesTimer = null;

// Results of every classified sign, readable by the automatic tests.
window.__jisr = { results: [], ready: false, framesProcessed: 0, delegate: null, trace: [] };

function safeLocalStorage() {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

// ---------- The two parts of the site ----------

function showView(name) {
  const view = name === 'learn' ? 'learn' : 'recognize';
  for (const [key, element] of Object.entries(el.views)) element.hidden = key !== view;
  for (const tab of el.tabs) tab.setAttribute('aria-current', tab.dataset.view === view ? 'page' : 'false');
  // The camera loop rests while the learning page is open.
  if (view === 'learn' && mode !== 'choosing') mode = 'paused';
  if (view === 'recognize' && mode === 'paused') mode = 'handsDown';
}

window.addEventListener('hashchange', () => showView(location.hash.slice(1)));
showView(location.hash.slice(1));
fillLearnPage(el.signGrid);

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

// What to tell the user about the voice, for each speech status.
const VOICE_NOTICES = {
  device: '',
  browser: 'يُستخدم صوت المتصفح، وقد يحتاج إلى الإنترنت.',
  unknown:
    'لم يُعثر على صوت عربي في هذا الجهاز. إن لم تسمع شيئاً: على أندرويد افتح الإعدادات ← النظام ← اللغات والإدخال ← ' +
    'تحويل النص إلى كلام ← تثبيت بيانات الصوت ← العربية، ثم أعد فتح التطبيق. وفي كل الأحوال تظهر الجملة بخط كبير ليقرأها الشخص الآخر.',
  unsupported: 'هذا المتصفح لا يدعم النطق. اعرض الجملة على الشخص الآخر ليقرأها.',
};

function showVoiceNotice(status) {
  el.voiceNotice.textContent = VOICE_NOTICES[status];
  el.voiceNotice.hidden = status === 'device';
}

// Say something and tell the user if the voice may need internet or does not exist.
function say(text) {
  if (TEST_MODE) return;
  const how = speak(text, currentRate());
  showVoiceNotice(how);
  if (how !== 'unsupported') {
    lastSpoken = text;
    el.replayButton.disabled = false;
  }
}

// Show the word chips and the sentence being built.
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

  const empty = words.length === 0;
  el.sentence.textContent = empty ? 'الكلمات التي تُتعرَّف تظهر هنا' : composeSentence(words, templates);
  el.sentence.classList.toggle('sentence--empty', empty);
  el.clearButton.disabled = empty;
  el.speakButton.disabled = empty;
}

// Show the conversation: the signer's sentences on one side, typed replies on the other.
function renderChat() {
  const messages = chat.all();
  if (messages.length === 0) {
    el.messages.innerHTML =
      '<li class="messages__empty">لا رسائل بعد. ما تقوله بالإشارة يظهر هنا، والشخص الآخر يكتب ردّه في الأسفل.</li>';
    return;
  }
  el.messages.replaceChildren(
    ...messages.map((message) => {
      const item = document.createElement('li');
      item.className = `message message--${message.who}`;
      const text = document.createElement('p');
      text.className = 'message__text';
      text.textContent = message.text;
      const meta = document.createElement('span');
      meta.className = 'message__meta';
      meta.textContent = `${message.who === 'signer' ? 'بالإشارة' : 'ردّ'} · ${formatTime(message.time)}`;
      item.append(text, meta);
      return item;
    }),
  );
  el.messages.lastElementChild.scrollIntoView({ block: 'nearest' });
}

// "انطق وأرسل": speak the sentence, put it in the conversation, start a new sentence.
function sendSentence() {
  const sentence = composeSentence(words, templates);
  if (!sentence) return;
  say(sentence);
  chat.add('signer', sentence);
  renderChat();
  words.length = 0;
  renderSentence();
  hideCandidates();
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
  if (mode === 'choosing' || mode === 'paused') return;

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

  // 3. Voices load a moment after the page opens; wait for them once and show the voice status.
  await waitForVoices();
  showVoiceNotice(speechStatus());

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
el.speakButton.addEventListener('click', sendSentence);
el.replayButton.addEventListener('click', () => say(lastSpoken));
// "جرّب الصوت": say a test sentence and list what the browser offers.
el.testVoiceButton.addEventListener('click', () => {
  say('مرحباً، أنا جسر');
  const voices = arabicVoices();
  el.voiceList.textContent = voices.length
    ? `الأصوات العربية المتاحة: ${voices.map((v) => v.name).join('، ')}`
    : 'لا يذكر المتصفح أي صوت عربي.';
});
el.clearChatButton.addEventListener('click', () => {
  chat.clear();
  renderChat();
});
// The other person types a reply; it is shown in large text (and never spoken: the signer reads it).
el.replyForm.addEventListener('submit', (event) => {
  event.preventDefault();
  if (chat.add('other', el.replyInput.value)) {
    el.replyInput.value = '';
    renderChat();
  }
});

renderSentence();
renderChat();

// Save the app on the device so it works without internet (only in the built app).
if ('serviceWorker' in navigator && import.meta.env.PROD) {
  navigator.serviceWorker.register('/sw.js');
}

// The camera opens as soon as the page opens.
start();
