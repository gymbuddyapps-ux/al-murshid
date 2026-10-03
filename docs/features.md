# Feature specification (version 2)

Version 2 (2026-10-03) added the nose and the two ears to the pose points, so that signs made next to
the face (chin, cheek, temple, forehead) can be told apart, and lowered the hand-detection thresholds
to 0.3 so hands in front of the face are kept. Version 1 had 6 pose points and 140 numbers per frame.

This is the single source of truth for how a video clip becomes numbers for the model.
The Python code (`jisr/features.py`) and the browser code (`web/src/features.js`) must both
follow it exactly. A test compares the two (Phase 5, "feature parity test").

## Words used here

- **Landmark:** a point that MediaPipe finds on the body, for example the tip of the index finger.
  Each landmark has `x`, `y` (position in the picture, from 0 to 1) and `z` (rough depth).
- **Clip:** the frames of one sign, from the moment the hand goes up until it comes down.
- **Feature vector:** the list of numbers that describes one frame.

## Inputs, per frame

From the MediaPipe **Pose Landmarker** (lite model), 9 landmarks, in this order:

| Position in our list | MediaPipe index | Body part |
|---|---|---|
| 0 | 0 | nose |
| 1 | 7 | left ear |
| 2 | 8 | right ear |
| 3 | 11 | left shoulder |
| 4 | 12 | right shoulder |
| 5 | 13 | left elbow |
| 6 | 14 | right elbow |
| 7 | 15 | left wrist |
| 8 | 16 | right wrist |

The nose and ears come from the pose model, not from a face model; no face mesh is used.

"Left" and "right" always mean the signer's own left and right.

From the MediaPipe **Hand Landmarker**: 0, 1 or 2 hands, each with 21 landmarks (`x`, `y`, `z`).

Each frame also has a **time** in milliseconds. Training clips are 25 frames per second,
so frame number `t` has time `t * 40`.

Face landmarks are never used.

## Step 1: make x and y use the same unit

MediaPipe gives `x` as a fraction of the picture width and `y` as a fraction of the height.
On a tall phone picture these are different lengths. So, with `aspect = width / height`:

```
X = x * aspect
Y = y
Z = z * aspect        (hands only; MediaPipe's hand z uses the same scale as x)
```

## Step 2: fill frames where the body was not found

If the pose is missing in a frame, copy the pose of the nearest earlier frame that has one.
If there is no earlier one, copy the nearest later one.
If **no** frame in the clip has a pose, the clip is rejected.

## Step 3: cut off the raising and the lowering of the hand

Why: the training clips (KArSL) start when the hand is already up and stop before it goes down.
In the app, a clip starts while the hand is still travelling up and ends while it travels down.
This step removes that travel, so the app's clips look like the training clips.
The same step is applied to the training clips, where it usually removes little or nothing.

The settings are in `config/segmenter.json`: `trim_speed` (in shoulder widths per second)
and `trim_max_ms`.

```
width = median over all frames of the shoulder distance

up_speed(t)   = fastest upward wrist movement between frame t-1 and frame t
              = max( left_wrist.Y[t-1] - left_wrist.Y[t],
                     right_wrist.Y[t-1] - right_wrist.Y[t] ) / width / seconds between the frames
down_speed(t) = the same with the sign reversed (movement downwards)
(if the two frames have the same time, the speed is 0)

first = 0
while first < T-1  and  time[first+1] - time[0] <= trim_max_ms  and  up_speed(first+1) > trim_speed:
    first = first + 1

last = T-1
while last > first  and  time[T-1] - time[last-1] <= trim_max_ms  and  down_speed(last) > trim_speed:
    last = last - 1

if fewer than 4 frames would be left: keep the whole clip
otherwise: keep frames first .. last
```

The wrists here are the two **pose** wrists (they are always available, even when a hand is not detected).
If, after this step, no kept frame has a hand, the clip is rejected.

## Step 4: one body reference for the whole clip

Over the kept frames of the clip:

```
mid_x = median of (left_shoulder.X + right_shoulder.X) / 2
mid_y = median of (left_shoulder.Y + right_shoulder.Y) / 2
width = median of the distance between the two shoulders (using X and Y)
```

If `width` is smaller than 0.000001 the clip is rejected.

Every point is then moved and scaled:

```
x' = (X - mid_x) / width
y' = (Y - mid_y) / width
z' = Z / width
```

After this step it does not matter where the person stands or how far from the camera they are.
The median is the middle value of a sorted list (for an even count, the average of the two middle values).

## Step 5: decide which hand is left and which is right

MediaPipe's own "Left/Right" label is not used. Instead, each detected hand is matched to the
nearest pose wrist, using the distance between the hand's landmark 0 (its wrist) and the pose wrist,
in `X`, `Y` units from step 1:

- **One hand detected:** it goes to the slot of the nearer pose wrist. A tie goes to the left slot.
- **Two hands detected** (call them A and B, in MediaPipe's order): compare
  `dist(A, left) + dist(B, right)` with `dist(A, right) + dist(B, left)`.
  Use the pairing with the smaller total. A tie uses A = left, B = right.
- More than two hands are never requested (`num_hands = 2`).

## Step 6: the feature vector of one frame (146 numbers)

| Positions | Content |
|---|---|
| 0 to 62 | left hand: 21 landmarks, each `x', y', z'` (all zeros if no left hand) |
| 63 to 125 | right hand: 21 landmarks, each `x', y', z'` (all zeros if no right hand) |
| 126 to 143 | pose: the 9 landmarks from the table above, each `x', y'` |
| 144 | 1 if a left hand was detected in this frame, otherwise 0 |
| 145 | 1 if a right hand was detected in this frame, otherwise 0 |

## Step 7: fixed length of 32 frames

A clip with `T` kept frames is turned into exactly 32 frames by picking frames (no blending):

```
for i in 0 .. 31:
    source = floor( i * (T - 1) / 31 + 0.5 )
    output[i] = frame[source]
```

If `T` is 1, the single frame is repeated 32 times.
Result: an array of shape `32 x 146`, stored as 32-bit floating point numbers.

## A clip is dropped when

- no frame has a pose (step 2), or
- no kept frame has any hand (step 3).

Dropped clips are counted per class in `reports/extraction_report.json`.

## MediaPipe settings (same in Python and in the browser)

| Setting | Value |
|---|---|
| Hand model | `hand_landmarker.task` (float16) |
| Pose model | `pose_landmarker_lite.task` (float16) |
| Running mode | VIDEO |
| `num_hands` | 2 |
| `num_poses` | 1 |
| Hand detection / presence confidence | 0.3 (version 2; was 0.5) |
| Other thresholds | see `config/extraction.json` |
