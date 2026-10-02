// Copy the files the app needs at run time into web/public/, so that everything is
// served from our own site (nothing is loaded from the internet while the app runs).
//
// Runs automatically before `npm run dev` and `npm run build`.
import { copyFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const web = join(dirname(fileURLToPath(import.meta.url)), '..');

// [from, to] pairs, both relative to the web/ folder
const files = [
  // MediaPipe: the programs (WASM) that run the hand and pose models in the browser
  ['node_modules/@mediapipe/tasks-vision/wasm/vision_wasm_internal.js', 'public/mediapipe/wasm/vision_wasm_internal.js'],
  ['node_modules/@mediapipe/tasks-vision/wasm/vision_wasm_internal.wasm', 'public/mediapipe/wasm/vision_wasm_internal.wasm'],
  ['node_modules/@mediapipe/tasks-vision/wasm/vision_wasm_nosimd_internal.js', 'public/mediapipe/wasm/vision_wasm_nosimd_internal.js'],
  ['node_modules/@mediapipe/tasks-vision/wasm/vision_wasm_nosimd_internal.wasm', 'public/mediapipe/wasm/vision_wasm_nosimd_internal.wasm'],
  // MediaPipe: the hand and pose models (the same files Python uses)
  ['../models/mediapipe/hand_landmarker.task', 'public/mediapipe/hand_landmarker.task'],
  ['../models/mediapipe/pose_landmarker_lite.task', 'public/mediapipe/pose_landmarker_lite.task'],
  // Our trained sign model and its label list (made by the Python training scripts)
  ['../models/jisr_model.onnx', 'public/model/jisr_model.onnx'],
  ['../models/labels.json', 'public/model/labels.json'],
];

for (const [from, to] of files) {
  const source = join(web, from);
  const target = join(web, to);
  if (!existsSync(source)) {
    console.warn(`copy-assets: missing ${from} (skipped)`);
    continue;
  }
  mkdirSync(dirname(target), { recursive: true });
  copyFileSync(source, target);
}
console.log('copy-assets: done');
