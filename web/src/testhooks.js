// Only used by the automatic tests (the app is opened with "?e2e=1").
//
// It lets a test hand the app the exact frames of one clip and get back the
// features and the prediction, using the same code as the live app:
// landmarks.js -> features.js -> classifier.js. The test then compares them
// with what the Python code produced for the same frames.
import segmenterConfig from '../../config/segmenter.json';
import { clipToFeatures } from './features.js';

const BLANK_FRAMES = 8; // blank pictures shown between clips, so one clip cannot influence the next
let clock = 0; // MediaPipe needs ever-increasing times

// Make a plain grey picture of the given size (like the pause between two signs).
function blankImage(width, height) {
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext('2d');
  context.fillStyle = '#808080';
  context.fillRect(0, 0, width, height);
  return canvas;
}

// frames: list of PNG pictures as base64 text, in order.
// frameMs: time between two frames in milliseconds (40 for the 25 fps training clips).
export async function runClip(detector, classifier, frames, frameMs) {
  const images = [];
  for (const text of frames) {
    const bytes = Uint8Array.from(atob(text), (c) => c.charCodeAt(0));
    images.push(await createImageBitmap(new Blob([bytes], { type: 'image/png' })));
  }
  const { width, height } = images[0];
  clock = Math.max(clock, performance.now()); // stay ahead of the camera loop's times

  // Forget the previous clip: a short pause with nothing to see.
  const blank = blankImage(width, height);
  for (let i = 0; i < BLANK_FRAMES; i++) {
    clock += frameMs;
    detector.detect(blank, clock);
  }

  const clip = [];
  for (let i = 0; i < images.length; i++) {
    clock += frameMs;
    clip.push({ ...detector.detect(images[i], clock), time: i * frameMs });
    images[i].close();
  }

  const features = clipToFeatures(clip, width / height, segmenterConfig);
  if (features === null) return { features: null, scores: null, result: null };
  const { scores, result } = await classifier.classifyWithScores(features);
  return { features: Array.from(features), scores, result };
}
