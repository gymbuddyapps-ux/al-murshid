// Finds the hands and the upper body in a camera picture, using MediaPipe.
//
// The model files and the WASM programs are loaded from our own site
// (/mediapipe/...), never from the internet. Settings match config/extraction.json,
// which the Python training code uses too.
import { FilesetResolver, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision';
import extraction from '../../config/extraction.json';
import { POSE_INDEXES } from './features.js';

export class LandmarkDetector {
  // Load both models. Tries the phone's graphics chip (GPU) first because it is
  // faster, and falls back to the normal processor (CPU) if that fails.
  // `preferGpu = false` is used by the automatic tests so results match Python.
  static async create(preferGpu = true) {
    const fileset = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
    const delegates = preferGpu ? ['GPU', 'CPU'] : ['CPU'];
    let lastError = null;

    for (const delegate of delegates) {
      try {
        const hand = await HandLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: '/mediapipe/hand_landmarker.task', delegate },
          runningMode: 'VIDEO',
          numHands: extraction.num_hands,
          minHandDetectionConfidence: extraction.min_hand_detection_confidence,
          minHandPresenceConfidence: extraction.min_hand_presence_confidence,
          minTrackingConfidence: extraction.min_tracking_confidence,
        });
        const pose = await PoseLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: '/mediapipe/pose_landmarker_lite.task', delegate },
          runningMode: 'VIDEO',
          numPoses: extraction.num_poses,
          minPoseDetectionConfidence: extraction.min_pose_detection_confidence,
          minPosePresenceConfidence: extraction.min_pose_presence_confidence,
          minTrackingConfidence: extraction.min_tracking_confidence,
        });
        return new LandmarkDetector(hand, pose, delegate);
      } catch (error) {
        lastError = error;
      }
    }
    throw lastError;
  }

  constructor(hand, pose, delegate) {
    this.hand = hand;
    this.pose = pose;
    this.delegate = delegate;
  }

  // Free the memory used by the two models.
  close() {
    this.hand.close();
    this.pose.close();
  }

  // Look at one picture (a <video> element or a canvas) and return a frame in the
  // form the rest of the app uses:
  //   { pose: 6 [x, y] points or null, hands: list of hands (21 [x, y, z] points each) }
  detect(image, timeMs) {
    const poseResult = this.pose.detectForVideo(image, timeMs);
    const handResult = this.hand.detectForVideo(image, timeMs);

    let pose = null;
    if (poseResult.landmarks.length > 0) {
      const body = poseResult.landmarks[0];
      pose = POSE_INDEXES.map((index) => [body[index].x, body[index].y]);
    }
    const hands = handResult.landmarks
      .slice(0, 2)
      .map((hand) => hand.map((point) => [point.x, point.y, point.z]));
    return { pose, hands };
  }
}
