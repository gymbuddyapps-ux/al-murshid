// Turn the landmarks of one clip into the model's input (32 frames x 140 numbers).
//
// This file follows docs/features.md step by step and must behave exactly like the
// Python version (jisr/features.py). A test compares the two.

export const NUM_FRAMES = 32; // every clip is resampled to this many frames
export const HAND_POINTS = 21; // landmarks per hand
export const POSE_POINTS = 6; // shoulders, elbows, wrists
export const FEATURE_SIZE = 2 * HAND_POINTS * 3 + POSE_POINTS * 2 + 2; // = 140
const MIN_FRAMES_AFTER_TRIM = 4; // trimming never leaves fewer frames than this

// MediaPipe pose indexes for: left shoulder, right shoulder, left elbow, right elbow,
// left wrist, right wrist.
export const POSE_INDEXES = [11, 12, 13, 14, 15, 16];

// Positions inside our 6-point pose list
const LEFT_SHOULDER = 0;
const RIGHT_SHOULDER = 1;
const LEFT_WRIST = 4;
const RIGHT_WRIST = 5;

// Where each part lives inside the 140-number vector
const LEFT_HAND_START = 0;
const RIGHT_HAND_START = 63;
const POSE_START = 126;
const LEFT_FLAG = 138;
const RIGHT_FLAG = 139;

// Middle value of a list (average of the two middle values for an even count).
export function median(values) {
  const ordered = [...values].sort((a, b) => a - b);
  const n = ordered.length;
  if (n % 2 === 1) return ordered[(n - 1) / 2];
  return (ordered[n / 2 - 1] + ordered[n / 2]) / 2;
}

// Step 2. `poses` has one entry per frame: a list of 6 [x, y] points, or null.
// Returns a list with no null, or null if the clip has no pose at all.
export function fillMissingPose(poses) {
  if (poses.every((p) => p === null)) return null;
  const filled = [...poses];
  // Copy forward: a missing frame takes the nearest earlier pose.
  for (let i = 1; i < filled.length; i++) {
    if (filled[i] === null) filled[i] = filled[i - 1];
  }
  // Frames at the start that are still missing take the nearest later pose.
  for (let i = filled.length - 2; i >= 0; i--) {
    if (filled[i] === null) filled[i] = filled[i + 1];
  }
  return filled;
}

// Distance between the two shoulders of one pose.
function shoulderWidth(pose) {
  return Math.sqrt(
    (pose[LEFT_SHOULDER][0] - pose[RIGHT_SHOULDER][0]) ** 2 +
      (pose[LEFT_SHOULDER][1] - pose[RIGHT_SHOULDER][1]) ** 2,
  );
}

// Step 3. Find the part of the clip that is left after cutting off the raising
// of the hand at the start and the lowering at the end.
//
// `poses` have no gaps, `times` are in milliseconds, `width` is the shoulder width,
// `trim` holds trim_speed (shoulder widths per second) and trim_max_ms.
// Returns [first, last]: the indexes of the first and last frame to keep.
export function trimRange(poses, times, width, trim) {
  const total = poses.length;

  // How fast the fastest wrist moved UP between frame t-1 and frame t.
  const speedUp = (t) => {
    const seconds = (times[t] - times[t - 1]) / 1000;
    if (seconds <= 0) return 0;
    const left = poses[t - 1][LEFT_WRIST][1] - poses[t][LEFT_WRIST][1];
    const right = poses[t - 1][RIGHT_WRIST][1] - poses[t][RIGHT_WRIST][1];
    return Math.max(left, right) / width / seconds;
  };

  // How fast the fastest wrist moved DOWN between frame t-1 and frame t.
  const speedDown = (t) => {
    const seconds = (times[t] - times[t - 1]) / 1000;
    if (seconds <= 0) return 0;
    const left = poses[t][LEFT_WRIST][1] - poses[t - 1][LEFT_WRIST][1];
    const right = poses[t][RIGHT_WRIST][1] - poses[t - 1][RIGHT_WRIST][1];
    return Math.max(left, right) / width / seconds;
  };

  let first = 0;
  while (
    first < total - 1 &&
    times[first + 1] - times[0] <= trim.trim_max_ms &&
    speedUp(first + 1) > trim.trim_speed
  ) {
    first += 1;
  }

  let last = total - 1;
  while (
    last > first &&
    times[total - 1] - times[last - 1] <= trim.trim_max_ms &&
    speedDown(last) > trim.trim_speed
  ) {
    last -= 1;
  }

  if (last - first + 1 < MIN_FRAMES_AFTER_TRIM) return [0, total - 1]; // keep everything
  return [first, last];
}

// Step 5. Decide which detected hand is the left one and which is the right one.
// `hands` is a list of 0, 1 or 2 hands (each a list of 21 [x, y, z] points).
// Returns [leftHand, rightHand]; each is a hand or null.
export function assignHands(hands, pose) {
  const dist = (hand, wristIndex) => {
    const dx = hand[0][0] - pose[wristIndex][0];
    const dy = hand[0][1] - pose[wristIndex][1];
    return Math.sqrt(dx * dx + dy * dy);
  };

  if (hands.length === 0) return [null, null];
  if (hands.length === 1) {
    if (dist(hands[0], LEFT_WRIST) <= dist(hands[0], RIGHT_WRIST)) return [hands[0], null];
    return [null, hands[0]];
  }
  const a = hands[0];
  const b = hands[1];
  if (dist(a, LEFT_WRIST) + dist(b, RIGHT_WRIST) <= dist(a, RIGHT_WRIST) + dist(b, LEFT_WRIST)) {
    return [a, b];
  }
  return [b, a];
}

// Build the feature array of one clip.
//
// `frames` has one object per video frame:
//    { pose: list of 6 [x, y] points as given by MediaPipe, or null,
//      hands: list of 0-2 hands, each a list of 21 [x, y, z] points,
//      time: time of the frame in milliseconds (only needed when trimming) }
// `aspect` is picture width / picture height.
// `trim` is the content of config/segmenter.json, or null to skip step 3.
//
// Returns a Float32Array of 32 * 140 numbers (frame after frame),
// or null if the clip must be dropped (no pose, or no hand in any frame).
export function clipToFeatures(frames, aspect, trim = null) {
  if (frames.length === 0) return null;

  // Step 1: same unit for x and y.
  let poses = frames.map((f) => (f.pose === null ? null : f.pose.map(([x, y]) => [x * aspect, y])));
  let allHands = frames.map((f) =>
    f.hands.map((hand) => hand.map(([x, y, z]) => [x * aspect, y, z * aspect])),
  );

  // Step 2: fill frames without a pose.
  poses = fillMissingPose(poses);
  if (poses === null) return null;

  // Step 3: cut off the raising and the lowering of the hand.
  if (trim !== null) {
    const widthBeforeTrim = median(poses.map(shoulderWidth));
    if (widthBeforeTrim < 1e-6) return null;
    const times = frames.map((f) => f.time);
    const [first, last] = trimRange(poses, times, widthBeforeTrim, trim);
    poses = poses.slice(first, last + 1);
    allHands = allHands.slice(first, last + 1);
  }

  if (allHands.every((hands) => hands.length === 0)) return null;

  // Step 4: one body reference for the whole clip.
  const midX = median(poses.map((p) => (p[LEFT_SHOULDER][0] + p[RIGHT_SHOULDER][0]) / 2));
  const midY = median(poses.map((p) => (p[LEFT_SHOULDER][1] + p[RIGHT_SHOULDER][1]) / 2));
  const width = median(poses.map(shoulderWidth));
  if (width < 1e-6) return null;

  // Steps 5 and 6: one 140-number vector per frame.
  const vectors = poses.map((pose, t) => {
    const vector = new Float64Array(FEATURE_SIZE);
    const [left, right] = assignHands(allHands[t], pose);
    const slots = [
      [left, LEFT_HAND_START, LEFT_FLAG],
      [right, RIGHT_HAND_START, RIGHT_FLAG],
    ];
    for (const [hand, start, flag] of slots) {
      if (hand === null) continue;
      for (let k = 0; k < HAND_POINTS; k++) {
        vector[start + 3 * k + 0] = (hand[k][0] - midX) / width;
        vector[start + 3 * k + 1] = (hand[k][1] - midY) / width;
        vector[start + 3 * k + 2] = hand[k][2] / width;
      }
      vector[flag] = 1;
    }
    for (let k = 0; k < POSE_POINTS; k++) {
      vector[POSE_START + 2 * k + 0] = (pose[k][0] - midX) / width;
      vector[POSE_START + 2 * k + 1] = (pose[k][1] - midY) / width;
    }
    return vector;
  });

  // Step 7: exactly 32 frames, written one after another into a single array.
  const picked = resample(vectors);
  const output = new Float32Array(NUM_FRAMES * FEATURE_SIZE);
  picked.forEach((vector, i) => output.set(vector, i * FEATURE_SIZE));
  return output;
}

// Step 7. Pick `numFrames` frames spread evenly over the clip (no blending).
export function resample(vectors, numFrames = NUM_FRAMES) {
  const total = vectors.length;
  const picked = [];
  for (let i = 0; i < numFrames; i++) {
    const source = total === 1 ? 0 : Math.floor((i * (total - 1)) / (numFrames - 1) + 0.5);
    picked.push(vectors[source]);
  }
  return picked;
}
