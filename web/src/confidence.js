// Turning the model's raw scores into the answer the app shows.

// Turn the model's raw scores into probabilities that add up to 1.
// Dividing by the temperature first makes the confidence more honest
// ("calibration"): the model stops being over-confident.
export function softmax(scores, temperature = 1) {
  const scaled = Array.from(scores, (s) => s / temperature);
  const largest = Math.max(...scaled);
  const exps = scaled.map((s) => Math.exp(s - largest));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

// From the probabilities, build the answer the app shows.
//   labels:     list of class names; the "other sign" class is at `otherIndex`
//   threshold:  below this confidence the app says "unclear"
// Returns { kind: 'ok', candidates }       -> candidates = top 3 words [{ label, confidence }]
//      or { kind: 'unclear', candidates }  -> the model thinks this is not one of our words,
//                                             or its best guess is below the threshold
export function interpret(probabilities, labels, otherIndex, threshold) {
  const order = probabilities.map((_, i) => i).sort((a, b) => probabilities[b] - probabilities[a]);
  const candidates = order
    .filter((i) => i !== otherIndex)
    .slice(0, 3)
    .map((i) => ({ label: labels[i], confidence: probabilities[i] }));

  const clear = order[0] !== otherIndex && candidates[0].confidence >= threshold;
  return { kind: clear ? 'ok' : 'unclear', candidates };
}
