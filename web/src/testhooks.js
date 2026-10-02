// Only used by the automatic tests (the app is opened with "?e2e=1").
//
// It lets a test hand the app the exact frames of one clip and get back the
// features and the prediction, using the same code as the live app:
// landmarks.js -> features.js -> classifier.js. The test then compares them
// with what the Python code produced for the same frames.
import segmenterConfig from '../../config/segmenter.json';
import { clipToFeatures } from './features.js';
import { LandmarkDetector } from './landmarks.js';

// frames: list of PNG pictures as base64 text, in order.
// frameMs: time between two frames in milliseconds (40 for the 25 fps training clips).
export async function runClip(classifier, frames, frameMs) {
  // A fresh detector for every clip, so nothing is remembered from the clip before
  // (the Python code does the same).
  const detector = await LandmarkDetector.create(false);

  const clip = [];
  let aspect = 1;
  for (let i = 0; i < frames.length; i++) {
    const bytes = Uint8Array.from(atob(frames[i]), (c) => c.charCodeAt(0));
    const image = await createImageBitmap(new Blob([bytes], { type: 'image/png' }));
    aspect = image.width / image.height;
    clip.push({ ...detector.detect(image, i * frameMs), time: i * frameMs });
    image.close();
  }
  detector.close();

  const features = clipToFeatures(clip, aspect, segmenterConfig);
  if (features === null) return { features: null, result: null };
  const { scores, result } = await classifier.classifyWithScores(features);
  return { features: Array.from(features), scores, result };
}
