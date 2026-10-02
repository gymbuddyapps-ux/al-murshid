// Sign detection: find where one sign starts and where it ends.
//
// The idea is simple:
//   - A sign STARTS when a hand has been raised for a few frames in a row.
//   - A sign ENDS when no hand has been raised for about 0.4 seconds.
//   - Only then is the whole clip given to the model. We never classify continuously.
//
// The numbers (how many frames, how many milliseconds) are in config/segmenter.json.

// Is at least one hand raised in this frame?
//
// "Raised" means: the hand's wrist is higher than a line below the shoulders.
// The line is `raiseLine` shoulder-widths under the shoulders, so it works for
// any distance from the camera.
//
// frame:  { pose: 6 [x, y] points or null, hands: list of hands (21 [x, y, z] points each) }
// aspect: picture width / picture height
export function isHandRaised(frame, aspect, raiseLine) {
  if (frame.pose === null || frame.hands.length === 0) return false;

  const [leftShoulder, rightShoulder] = frame.pose;
  const shoulderY = (leftShoulder[1] + rightShoulder[1]) / 2;
  const shoulderWidth = Math.sqrt(
    ((leftShoulder[0] - rightShoulder[0]) * aspect) ** 2 + (leftShoulder[1] - rightShoulder[1]) ** 2,
  );
  if (shoulderWidth < 1e-6) return false;

  // In a picture, y grows downwards, so "higher" means a smaller y.
  const line = shoulderY + raiseLine * shoulderWidth;
  return frame.hands.some((hand) => hand[0][1] < line);
}

// The segmenter remembers the frames of the sign that is being made.
// Call push() once per camera frame. It returns:
//    { state: 'idle' }                      nobody is signing
//    { state: 'signing' }                   a sign is in progress
//    { state: 'finished', frames: [...] }   a sign just ended; `frames` is the clip
export class Segmenter {
  constructor(config) {
    this.config = config;
    this.reset();
  }

  reset() {
    this.signing = false;
    this.frames = []; // frames collected so far
    this.raisedInARow = 0; // how many raised frames we have seen in a row while idle
    this.startTime = 0; // when the current sign started
    this.lastRaisedTime = 0; // last moment a hand was raised
    this.lastRaisedCount = 0; // how many frames we had at that moment
  }

  push(frame, timeMs, raised) {
    const { start_frames, end_ms, min_ms, max_ms } = this.config;

    if (!this.signing) {
      if (!raised) {
        this.raisedInARow = 0;
        this.frames = [];
        return { state: 'idle' };
      }
      // A hand is up. Keep the frame; the sign starts after a few such frames in a row.
      if (this.raisedInARow === 0) this.startTime = timeMs;
      this.raisedInARow += 1;
      this.frames.push(frame);
      this.lastRaisedTime = timeMs;
      this.lastRaisedCount = this.frames.length;
      if (this.raisedInARow < start_frames) return { state: 'idle' };
      this.signing = true;
      return { state: 'signing' };
    }

    // A sign is in progress.
    this.frames.push(frame);
    if (raised) {
      this.lastRaisedTime = timeMs;
      this.lastRaisedCount = this.frames.length;
    }

    const handsDownLongEnough = timeMs - this.lastRaisedTime >= end_ms;
    const tooLong = timeMs - this.startTime >= max_ms;
    if (!handsDownLongEnough && !tooLong) return { state: 'signing' };

    // The sign is over. Keep only the frames up to the last raised hand.
    const clip = this.frames.slice(0, this.lastRaisedCount);
    const duration = this.lastRaisedTime - this.startTime;
    this.reset();
    if (duration < min_ms) return { state: 'idle' }; // too short: a twitch, not a sign
    return { state: 'finished', frames: clip };
  }
}
