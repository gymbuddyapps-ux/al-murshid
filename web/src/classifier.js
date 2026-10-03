// Runs our trained sign model in the browser and returns the 3 most likely words.
//
// The model file (jisr_model.onnx) and labels.json are made by the Python training
// scripts. labels.json also carries the "temperature" and the "threshold" chosen
// during evaluation (see docs/REPORT.md).
import * as ort from 'onnxruntime-web/wasm';
import { interpret, softmax } from './confidence.js';
import { FEATURE_SIZE, NUM_FRAMES } from './features.js';

// Use one thread (several threads would need special server settings).
// The ONNX Runtime program (WASM) is bundled with the app by Vite.
ort.env.wasm.numThreads = 1;

export class SignClassifier {
  static async create() {
    const info = await (await fetch('model/labels.json')).json();
    const session = await ort.InferenceSession.create('model/jisr_model.onnx', {
      executionProviders: ['wasm'],
    });
    return new SignClassifier(session, info);
  }

  constructor(session, info) {
    this.session = session;
    this.info = info; // { labels, other_index, temperature, threshold }
  }

  // `features` is the Float32Array of 32 x 140 numbers from features.js.
  async classify(features) {
    return (await this.classifyWithScores(features)).result;
  }

  // Same as classify(), but also returns the model's raw scores (used by the tests).
  async classifyWithScores(features) {
    const input = new ort.Tensor('float32', features, [1, NUM_FRAMES, FEATURE_SIZE]);
    const output = await this.session.run({ [this.session.inputNames[0]]: input });
    const scores = Array.from(output[this.session.outputNames[0]].data);
    const probabilities = softmax(scores, this.info.temperature);
    const result = interpret(probabilities, this.info.labels, this.info.other_index, this.info.threshold);
    return { scores, result };
  }
}
