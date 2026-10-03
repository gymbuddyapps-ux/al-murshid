# Process log

One entry per phase: what was done, why, which files, key commands, real results, problems and decisions.
Numbers are copied from files under `reports/`.

## Phase 0: Vocabulary audit

**Goal:** find out which Arabic sign words really exist in the public datasets, and pick 15 to 20 of them.

**What was done**

1. Set up the project: git repository, `.gitignore` (raw data, videos and secrets are never committed), and a Python 3.13 virtual environment in `.venv/`.
2. Checked each data source named in the brief.
3. Counted the samples and signers of every class and wrote `docs/vocab_audit.md`.
4. Chose 20 classes and saved them in `config/vocab.json`.

**Results** (from `reports/audit/`)

| Source | Result |
|---|---|
| ArabicSL-Net | Not usable. The Zenodo readme says the data is private, and Kaggle answers "403 Forbidden". |
| Zenodo "Arabic words sign language video dataset" | Not usable. The archive holds 5 videos of 1 sign, not 3000 videos of 30 signs (`zenodo_8035320.json`). |
| KArSL-502 (fallback) | Usable. 502 signs, 75,515 samples, 3 signers, at least 117 samples per sign (`karsl_counts.json`). |

**Problems hit**

- Both primary datasets were unavailable, so everything depends on the fallback dataset.
- KArSL has no fruit, vegetable or shopping words.
- The official KArSL download (a university SharePoint) returned an error page to scripts. A public Kaggle copy of the same data is used instead (`yousefdotpy/karsl-502`, 25 GB of video frames at 256x256 pixels).
- The Zenodo file is a `.rar` archive and Windows could not read its Arabic file names. A portable copy of 7-Zip was placed in `data/tools/` (not installed system-wide, not committed).

**Decisions**

- **Scenario changed from grocery to pharmacy.** Reason: KArSL's health chapter has the words for it (medicine, headache, fever, pharmacy), and it keeps the idea of a deaf customer talking to a seller.
- **Numbers, letters and two-meaning labels were left out.** Numbers and letters look too similar to each other, and labels such as "مريض / مرض" do not say which meaning the sign has.
- **Signer IDs exist (3 signers)**, so Phase 3 uses signer-independent evaluation: train on one signer, tune on another, test once on the third.
- Two other Kaggle datasets were found but not used, because they are not in the brief's list of allowed sources. They are listed in `docs/vocab_audit.md`.

**Files**

- `docs/vocab_audit.md`, `config/vocab.json`
- `scripts/audit/kaggle_list_classes.py`, `scripts/audit/karsl_counts.py`, `scripts/audit/build_vocab_audit.py`
- `reports/audit/*.json`

**Key commands**

```
python scripts/audit/kaggle_list_classes.py <owner>/<dataset>
python scripts/audit/karsl_counts.py
python scripts/audit/build_vocab_audit.py
kaggle datasets download yousefdotpy/karsl-502 -p data/raw/karsl/mirror_502
```

## Phase 1: Landmark extraction

**Goal:** turn every video clip into a small table of numbers (landmarks) that the model can learn from,
with one written rule book that Python and the browser both follow.

**What was done**

1. Downloaded the two MediaPipe models (`models/mediapipe/`), the same files the browser uses:
   - https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
   - https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
2. Checked that MediaPipe finds the hand and the body in KArSL's small 256x256 frames (it does, without any upscaling).
3. Wrote the rule book `docs/features.md`, then the same rules twice: `jisr/features.py` and `web/src/features.js`.
   A shared test file (`web/tests/fixtures/feature_parity.json`, made by `scripts/make_parity_fixture.py`) holds
   invented clips and the Python results; the JavaScript tests must reproduce them.
4. `scripts/prepare_karsl.py` reads the 25 GB zip from the start and unpacks only the signs we need. It also works
   while the download is still running, and resumes later.
5. `scripts/extract_landmarks.py` runs MediaPipe on every clip in parallel (7 processes) and caches one `.npz` file
   per clip, so it can be stopped and restarted.
6. `scripts/build_features.py` applies the rule book and writes `data/features/karsl_v1.npz` plus
   `reports/extraction_report.json`.

**Findings and decisions**

- **The sample names are `<group>_<signer>_<sign>_(date)`**, so the signer is the *second* field, not the first.
  The unpacking script checks the signer against the zip's top folder (`signer_mismatches` in
  `reports/audit/karsl_downloaded_counts.json` is 0).
- **KArSL clips contain only the sign itself.** The hand is already raised in the first frame and still raised in
  the last. There are no "hands down" frames. Two consequences:
  - In the app, a clip includes the hand going up and coming down. A new **trim step** (step 3 in
    `docs/features.md`) cuts off that travel, using the speed of the wrists. The same step runs on the training
    clips, where it removes almost nothing (see the extraction report), so training and app data match.
  - There are no quiet parts to build an **idle class** from, as the brief asked. Instead an **"other sign"
    class** was built from 60 KArSL signs outside the vocabulary (`scripts/choose_other_signs.py`, fixed random
    seed): 40 signs for training, 10 for validation, 10 for the final test, 5 clips per sign and signer. The model
    can then answer "this is not one of my words", and the app says "unclear".
- **Hand assignment does not trust MediaPipe's left/right label.** Each hand is matched to the nearest pose wrist
  instead (step 5 in `docs/features.md`), which is simpler to reproduce exactly in two languages.
- The "raised hand" line for sign detection (`raise_line` in `config/segmenter.json`) was set to 1.15 shoulder
  widths below the shoulders after looking at where the wrists are in the training clips (99% of hand-visible frames
  are above 1.0) and where resting hands are (about 1.35 on the lap in the sample frame).

**Results** (from `reports/extraction_report.json`, written after the full run)

See `docs/REPORT.md`, section 2, for the per-class table: clips processed and kept, dropped clips, hand detection
rate, and the share of frames removed by the trim step.

**Problems hit**

- The 25 GB download ran at about 2.5 MB/s (close to 3 hours). The unpacking script was written to work on the
  partially downloaded file so that extraction could start on signer 1 while signers 2 and 3 were still arriving.
- MediaPipe on this 4-core laptop processes about 50 to 60 clips per minute with 7 workers.

**Files**

- `docs/features.md`, `jisr/features.py`, `web/src/features.js`, `jisr/detect.py`
- `scripts/prepare_karsl.py`, `scripts/extract_landmarks.py`, `scripts/build_features.py`, `scripts/choose_other_signs.py`
- `tests/test_features.py`, `web/tests/features.test.js`, `scripts/make_parity_fixture.py`
- `config/extraction.json`, `config/segmenter.json`

**Key commands**

```
python scripts/choose_other_signs.py
python scripts/prepare_karsl.py
python scripts/extract_landmarks.py --workers 7
python scripts/build_features.py
python scripts/make_parity_fixture.py && cd web && npm test
```

## Phase 2: Model

**Goal:** a small model that reads the 32 x 140 feature table of a clip and says which word it is.

**What was done**

1. `jisr/model.py`: a 1D convolutional network (three convolution layers over time, then average and
   maximum over time, then one linear layer). 88,437 parameters, 0.35 MB as ONNX, far under the 1 MB limit.
2. `jisr/augment.py`: left/right mirroring (hands and body sides swapped correctly), rotation up to 12 degrees,
   scale change up to 15 percent, speed change (cutting up to 15 percent at each end and stretching back),
   repeated frames, hands that disappear for a frame, and small noise.
3. `jisr/training.py`, `scripts/train.py`: AdamW, one-cycle learning rate, label smoothing, 80 epochs, seed 1,
   CPU only. The baseline is a nearest-neighbour classifier with dynamic time warping (DTW) on 16 of the 32 frames.
4. The "other sign" class replaces the idle class (see Phase 1 for the reason).

**Protocol** (`config/training.json`): development signers 1 and 2, test signer 3. Validation rounds train
on one development signer and check on the other. The final model trains on both. The test signer is used only
once, in Phase 3.

**Run 1** (`reports/run1/training_report.json`)

| Train on | Check on | CNN top-1 (words) | CNN top-3 (words) | kNN-DTW top-1 |
|---|---|---|---|---|
| signer 2 | signer 1 | 88.0% | 94.9% | 76.9% |
| signer 1 | signer 2 | 90.8% | 99.7% | 65.2% |

The CNN was kept (mean top-1 89.4% against 71.0% for the baseline). ONNX and PyTorch give the same scores
(largest difference 2.6e-6, `onnx_max_difference` in the training report).

**Problem found on validation, and the fix (run 2)**

In the validation rounds, many word clips of the unseen signer were classified as "other sign": for example
صيدلية 100% and كبسولة 100% when checking on signer 1, دواء 98% and يساعد 94% when checking on signer 2
(computed from `reports/run1`'s validation scores). The reason is class imbalance: "other sign" has 300 training
clips per signer, a word only about 50, so the model learned that "other" is the safe answer whenever a new
signer looks unfamiliar.

Fix, decided from validation data only: class-balanced loss weights (`balance_classes` in
`config/training.json`; each class is weighted by the inverse of its clip count). Everything was retrained
from scratch with the same seed. Run 2 numbers are in `reports/training_report.json` and `docs/REPORT.md`.

**Files**

- `jisr/model.py`, `jisr/augment.py`, `jisr/training.py`, `jisr/metrics.py`, `scripts/train.py`
- `tests/test_augment.py`, `config/training.json`
- `models/jisr_model.pt`, `models/jisr_model.onnx`, `reports/training_report.json`, `reports/validation_scores.npz`

**Key commands**

```
python scripts/train.py            # add --skip-baseline to skip the slow DTW baseline
```

## Phase 3: Honest evaluation

**Goal:** measure the model once on a signer it never saw, calibrate the confidence, and choose the
rejection threshold.

**Protocol.** Signer 3 was held out from everything: training, model choice, epochs, temperature and
threshold. The temperature (calibration) and the threshold were chosen on the validation outputs of the two
development rounds only (`reports/validation_scores.npz`). The test ran once per training run.

**Results on the held-out signer** (`reports/metrics.json`; run 1 in `reports/run1/metrics.json`)

| | Run 1 | Run 2 (class-balanced, shipped) | Target |
|---|---|---|---|
| Top-1 accuracy, 1010 vocabulary clips | 90.6% | **93.6%** | 90% |
| Top-3 accuracy | 99.3% | **99.6%** | 97% |
| Temperature / threshold (from validation) | 0.90 / 0.35 | 0.65 / 0.50 | |
| Vocabulary clips accepted by the app | 85.9% | 88.2% | |
| Top-1 accuracy on accepted clips | 99.1% | 99.4% | |
| Unknown signs (50 clips, 10 signs never seen) rejected | 90% | 90% | |

Mean confidence of the best guess when it is right: 89.6%; when it is wrong: 29.9% (run 2).

**Targets met, with one honest caveat.** Top-1 and top-3 rank the vocabulary words only. The per-class
acceptance numbers show what the user would experience with this signer:

- 17 of 20 words: accepted and in the top 3 in 96% to 100% of clips.
- **دواء (medicine): accepted in 59% of clips.** The rest are answered "other sign", so the app says "unclear".
- **شكراً (thanks): accepted in 10%.** The rest go to "other sign" or are confused with ألم (pain).
- **كبسولة (capsule): accepted in 0%.** Every clip of this signer is answered "other sign".

These words were not dropped, for two reasons: the brief's targets are met, and dropping classes based on
the test signer's confusion matrix would be tuning on the test data. The same three words were also among the
hardest on validation (`docs/PROCESS_LOG.md`, Phase 2), so the problem is real and not a test-set accident.
`docs/REPORT.md` shows the per-class table and the confusion matrix.

**Figures:** `docs/figures/confusion_matrix.png`, `docs/figures/confidence.png`.

**Decisions**

- The rejection threshold is the one that best balances "vocabulary clips accepted and correct" against
  "unknown signs rejected" on validation (`choose_threshold` in `jisr/metrics.py`; the full table of candidate
  thresholds is in `reports/metrics.json`).
- The shipped model is exactly the evaluated one (trained on signers 1 and 2). A model trained on all three
  signers would probably be better for new users, but its accuracy on new signers could not be measured.

**Files**

- `scripts/evaluate.py`, `jisr/metrics.py`, `reports/metrics.json`, `models/labels.json`
- `docs/figures/confusion_matrix.png`, `docs/figures/confidence.png`

**Key commands**

```
python scripts/evaluate.py
```

## Phase 4: Web app

**Goal:** a phone-friendly web app (PWA), Arabic only, right to left, that works offline after the first visit.

**What was done**

- Stack: Vite + plain JavaScript, no framework. One file per idea in `web/src/`: `landmarks.js` (MediaPipe),
  `segmenter.js` (where a sign starts and ends), `features.js` (the rule book), `classifier.js` and
  `confidence.js` (ONNX Runtime Web, softmax, top 3, "unclear"), `composer.js` (sentence templates),
  `speech.js` (Web Speech API), `overlay.js` (skeleton drawing), `main.js` (the screens and the camera loop).
- Everything is served from the app itself: MediaPipe WASM and models, ONNX Runtime WASM, the trained model,
  and the two fonts (Reem Kufi for the name and the large sentence, Tajawal for the rest). No request leaves
  the site while the app runs; the only Arabic voices used are the ones installed on the device
  (`localService` voices only).
- Service worker (`web/public/sw.js`): at build time Vite writes the list of all built files into it, and it
  saves them on the first visit. Manifest with icons drawn by `scripts/make_icons.py`.
- Screens: welcome (tips, privacy sentence, camera permission), camera (door-shaped arch, status pill,
  optional skeleton), candidates (3 large buttons with percentages, "أعد الإشارة"), sentence strip (chips,
  clear all), the sentence in very large text, "انطق", "أعد النطق" and three speeds.
- Design: colors from the Omani flag in calm tones on a lime-white background, an eight-pointed-star pattern
  inspired by carved doors, an original bridge logo. No emblem or flag.

**Decisions and problems**

- The models load when the user taps "افتح الكاميرا" (during the permission dialog), not at page open:
  loading them at page open made Lighthouse's performance score drop to 70; now it is 99.
- Edge/Chrome preview servers add a `Vary` header that made module scripts miss the service-worker cache
  offline. Fixed with `ignoreVary: true` in the cache lookup (found by the offline browser test).
- ONNX Runtime's WASM file is bundled once by Vite (the earlier extra copy doubled the first download).
- Lighthouse 13 has no PWA category any more; offline use is tested by `web/e2e/offline.spec.js` instead.

**Results** (`reports/lighthouse.report.json`): performance 99, accessibility 100, best practices 100, SEO 100.

**Files:** `web/` (see `docs/ARCHITECTURE.md`), `config/templates.json`, `config/segmenter.json`.

**Key commands**

```
cd web && npm install && npm run dev        # development
cd web && npm run build && npm run preview  # production build at http://localhost:4173
```
