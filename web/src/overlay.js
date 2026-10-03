// Draws the "skeleton" (hand and arm lines) on top of the camera picture.
// This is only for the user to see what the app sees; it does not affect recognition.

// Which hand landmarks are joined by a line (wrist to fingers, along each finger).
const HAND_LINES = [
  [0, 1], [1, 2], [2, 3], [3, 4], // thumb
  [0, 5], [5, 6], [6, 7], [7, 8], // index finger
  [5, 9], [9, 10], [10, 11], [11, 12], // middle finger
  [9, 13], [13, 14], [14, 15], [15, 16], // ring finger
  [13, 17], [17, 18], [18, 19], [19, 20], // little finger
  [0, 17],
];

// Our 9 pose points: 0 nose, 1-2 ears, 3-4 shoulders, 5-6 elbows, 7-8 wrists.
const POSE_LINES = [
  [1, 0], [0, 2], [3, 4], [3, 5], [5, 7], [4, 6], [6, 8],
];

function drawLines(context, points, lines, width, height) {
  context.beginPath();
  for (const [from, to] of lines) {
    context.moveTo(points[from][0] * width, points[from][1] * height);
    context.lineTo(points[to][0] * width, points[to][1] * height);
  }
  context.stroke();
}

// `frame` is { pose, hands } from landmarks.js. Pass `show = false` to just clear.
export function drawSkeleton(canvas, frame, show) {
  const context = canvas.getContext('2d');
  const { width, height } = canvas;
  context.clearRect(0, 0, width, height);
  if (!show || !frame) return;

  context.lineCap = 'round';
  context.lineWidth = Math.max(2, width / 160);

  if (frame.pose) {
    context.strokeStyle = '#ffffff';
    drawLines(context, frame.pose, POSE_LINES, width, height);
  }
  context.strokeStyle = '#7fd6ab';
  for (const hand of frame.hands) {
    drawLines(context, hand, HAND_LINES, width, height);
  }
}
