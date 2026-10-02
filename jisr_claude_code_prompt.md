# Project brief for Claude Code: "Jisr" (جسر), Arabic sign language to Arabic text and speech

Read this whole brief first. Then create a task list and work through the phases in order. Work autonomously. Stop and ask me only at the points marked **STOP**.

## 1. What we are building

A mobile-friendly web app (PWA). A deaf or mute person signs in front of the phone camera. The app recognizes Arabic sign language words, shows them as Arabic text, and speaks the sentence aloud in Arabic so a hearing person (a shopkeeper, for example) can understand.

- Direction is one way only: **sign to text and speech**. No speech-to-text.
- Primary scenario: buying from a grocery or vegetable seller (for example "I want a kilo of tomatoes, how much, thank you"). The real vocabulary depends on what the public datasets contain (see Phase 0).
- This is for a student's competition in Oman. The student is a beginner programmer who will present it, and I will explain it to her afterwards. **Simple, readable code and clear documentation matter as much as the app.**
- I want a working end-to-end demo within 1 to 2 days. Get a thin working path first (data to model to browser to speech), then improve accuracy.
- **Accuracy is the top priority**, but it must be measured honestly (see Phase 3). A smaller vocabulary with real, verified accuracy beats a bigger one with inflated numbers.

## 2. Hard rules

1. **No new data collection.** I will not film anything, and nobody will record custom gestures. Use only public datasets listed below. Do not invent, mime, or guess signs.
2. **No manual training by me.** You write and run the training scripts. I only supply credentials and run the final app.
3. **Honesty about numbers.** Every metric in any document must come from a script output saved under `reports/` (for example `metrics.json`). Never write a number from memory or estimation. Never report test results after tuning on the test data.
4. **Do not guess what a dataset label means.** If a class name is ambiguous, cryptic, or I cannot verify it from the dataset's own documentation, exclude that class.
5. **Privacy:** all inference runs on the device. No video or landmarks leave the phone. No analytics, no external calls at runtime except loading the app itself. Say this in the UI.
6. **Secrets:** Kaggle credentials come from `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables or `~/.kaggle/kaggle.json`. Never print them, log them, or commit them. If they are missing, **STOP** and tell me exactly how to create a Kaggle API token and where to put it.
7. **Ask before:** deploying anywhere public, installing system-wide packages, or spending money. Prepare deployment but do not publish without my yes.
8. Use git from the start, with a commit at the end of every phase and a `.gitignore` that excludes raw data, videos, and secrets.

## 3. Data sources (public, free)

Primary:

- **ArabicSL-Net**: 307 Arabic words, about 30,000 videos recorded with mobile cameras in four settings (bank, cafe, hospital, train station). License CC BY 4.0. Zenodo record 10.5281/zenodo.7771372 holds only a readme. The videos are on Kaggle at `deepologylab/arabicsl-net`. That exact URL returned 404 for me when I checked, so search Kaggle for the dataset by name or authors (Deepology Lab, Zagazig University) and use the `kaggle` CLI.
- **Arabic words sign language video dataset**: 3000 videos, 5 volunteers, 30 signs, 20 repetitions each, smartphone recordings. License CC BY 4.0. Zenodo record 10.5281/zenodo.8035320 (`Dataset.rar`). The listed file size looks suspiciously small for 3000 videos, so verify the actual download.

Fallback only if both lack usable classes: **KArSL** (https://hamzah-luqman.github.io/KArSL/, 502 isolated signs, 3 professional signers, citation required). Its chapters are letters, numbers, health, common verbs, family, characteristics, directions, social relationships, in-house, religion, and jobs. There are no fruits or vegetables in it.

Check disk space and total download size before downloading. Download only the class folders you need if the tooling allows it. Cap videos per class (for example 150 to 200) to keep extraction time reasonable.

If a download is blocked by the sandbox network, **STOP** and tell me which file to download manually and where to put it.

## 4. Phases

### Phase 0: Vocabulary audit (gate)

1. Download or list the class names of both datasets.
2. Write `docs/vocab_audit.md`: a table of every class with its Arabic label, source dataset, number of videos, and number of distinct signers if the metadata allows.
3. Choose **15 to 20 classes**, in this order of preference, using only what exists:
   - Produce: طماطم، بطاطس، خيار، موز، تفاح، برتقال، بصل، جزر، ليمون، عنب (and similar)
   - Shopping words: أريد، كيلو، السعر، كم، شكراً، نعم، لا، من فضلك
   - If produce is mostly missing, adapt the scenario to the available shopping, cafe, or food words (for example drinks, food, price, thank you). Record the change and the reason in `docs/vocab_audit.md`.
4. Selection criteria: label verifiable, enough videos per class (at least about 40), multiple signers where possible, classes not nearly identical to each other.
5. Record the chosen vocabulary in `config/vocab.json` (Arabic label, id, source dataset).
6. **STOP only if** fewer than 10 usable classes exist across all sources. Otherwise continue automatically and I will review the audit later.

Important: check whether the datasets provide **signer IDs**. This decides how we split data in Phase 3. Document what you find.

### Phase 1: Landmark extraction

- Use MediaPipe **Hand Landmarker** and **Pose Landmarker** (Tasks API, same model files in Python and in the browser). Do not use face landmarks.
- Per frame build one feature vector: both hands (21 points x 3 coordinates each) plus a small upper-body pose subset (shoulders, elbows, wrists). Normalize relative to the shoulder midpoint and shoulder width so position and distance from the camera do not matter. If a hand is missing, fill zeros and add a presence flag.
- Resample every clip to a fixed number of frames (start with 32).
- Save features to `data/features/*.npz` with labels and signer or source metadata. Run extraction in parallel and cache results so it can resume.
- **Put the feature code in one shared, well-commented specification** (`docs/features.md`) so the JavaScript version matches the Python version exactly. Later you will verify this match with a test (Phase 4).
- Drop clips where hands were never detected, and report how many were dropped per class.

### Phase 2: Model

- Small sequence model (PyTorch): start with a 1D-CNN or a small GRU over the landmark sequences. Target size under about 1 MB. Compare against a simple baseline (for example k-nearest-neighbors with dynamic time warping) and keep the better one based on **validation** data.
- Augmentation: left/right mirroring (swap hands correctly), small rotation, scale, jitter, speed change, random dropped frames.
- Include an **idle / no-sign** class built from the quiet parts of clips (hands down, start and end of videos), so the model does not invent words when nobody is signing.
- Train on CPU. Fix random seeds. Save the training config and the label map.

### Phase 3: Honest evaluation (this is the accuracy gate)

- **If signer IDs exist:** use leave-one-signer-out evaluation. Tune only on validation signers. Evaluate once on held-out signers that were never used for tuning.
- **If signer IDs do not exist:** say so prominently in the report. Use the Zenodo 5-volunteer dataset for a signer-independent test wherever possible, and clearly label any remaining numbers as "same-signer, optimistic".
- Report in `reports/`: top-1 accuracy, top-3 accuracy, per-class precision and recall, a confusion matrix image (`docs/figures/`), and the confidence distribution for correct versus wrong predictions.
- Calibrate confidence (temperature scaling) and choose a **rejection threshold**: below it, the app says "unclear, repeat the sign" instead of guessing. Report accuracy on accepted predictions and the acceptance rate.
- **Targets:** held-out-signer top-1 at least 90 percent and top-3 at least 97 percent on the chosen vocabulary.
- If targets are not met: first try better augmentation, more data per class, and feature fixes. Then use the confusion matrix to drop the most confusable classes and re-evaluate. Do not tune on the test signers. If the targets still cannot be met, do not hide it. Report the real numbers and the smallest vocabulary that does meet them.

### Phase 4: Web app (PWA)

Stack: Vite plus plain TypeScript or JavaScript, no heavy framework (so a beginner can read it). Export the model to ONNX and run it with onnxruntime-web (or TensorFlow.js if clearly simpler). Bundle the MediaPipe model files and WASM **locally** so the app works offline after the first load. Add a web app manifest and a service worker.

Language: **Arabic only, right-to-left.** Work on desktop Chrome and Android Chrome. The camera needs HTTPS (localhost is fine for testing).

Screens and behavior:

1. **Welcome and guidance screen:** camera permission request, plus simple tips: keep both hands and face in view, stay at arm's length, use good lighting. Only check what can be checked reliably (are hands and shoulders visible). Do not claim to analyze the background.
2. **Camera screen:** live video with an optional skeleton overlay, and a clear status (ready, signing, processing).
3. **Sign detection:** start a segment when a hand is raised and visible for a few frames, end it when hands drop or disappear for about 0.4 seconds, then classify the segment. Do not classify continuously.
4. **Confirmation:** show the **top 3 candidates** as large buttons with confidence. The user taps to confirm one, or taps "أعد الإشارة". If the top candidate is below the rejection threshold, show "غير واضح، أعد الإشارة".
5. **Sentence strip:** confirmed words appear as chips, each removable, plus a clear-all button.
6. **Sentence composer:** simple rule-based templates (stored as JSON in `config/templates.json`, with unit tests) that turn word sequences into correct Arabic, for example "أريد + كيلو + طماطم" into "أريد كيلو طماطم". Adapt templates to the real vocabulary. Fallback: join words with spaces.
7. **Speech:** a large "انطق" button using the Web Speech API (`speechSynthesis`), preferring an Arabic voice. If no Arabic voice is available, tell the user clearly and show the sentence in very large text so the other person can read it. Always show the text in large size regardless. Add a replay button and a speed control.

Accessibility: very large buttons, high contrast, never rely on sound alone, short vibration on confirm where supported.

Performance: target at least 15 frames per second on a mid-range Android phone. Use the lite MediaPipe models and prefer GPU delegates when available.

**Design (Omani-inspired, professional but warm):**

- Colors derived from the Omani flag: deep red, white, and green, in calm tones, with a light theme as the base.
- Subtle geometric patterns inspired by Omani silverwork and carved doors in backgrounds and borders. A very light khanjar-inspired line motif is allowed as a small accent.
- Do **not** use the national emblem, the coat of arms, or the flag itself.
- Arabic-friendly font bundled locally (for example Cairo, Tajawal, or Noto Naskh Arabic).
- Name: **جسر (Jisr)**. Logo: an original simple bridge-arch SVG.
- The design must not feel like a charity or medical app. Aim for a dignified, modern tool.

### Phase 5: Testing

- Unit tests: feature normalization, segment detection, sentence composer.
- **Feature parity test:** run the same video through the Python extractor and the browser extractor and assert the feature vectors match within a small tolerance. Fix any mismatch, because this is the most common silent cause of lost accuracy.
- **End-to-end test:** with Playwright and Chrome's fake video capture (`--use-file-for-fake-video-capture`), feed held-out test videos through the real browser pipeline. Browser predictions must agree with Python predictions on at least 95 percent of clips. Report the actual numbers.
- Manual checklist for desktop and Android, saved in `docs/manual_test_checklist.md`.
- A Lighthouse PWA check.

### Phase 6: Documentation (needed for the explanation slides)

I will turn these documents into slides to teach the student, so write them for a **beginner programmer**: plain language, define every technical term the first time (landmarks, sequence model, confidence, overfitting, confusion matrix), and keep each section short.

- `docs/PROCESS_LOG.md`: **append after every phase**. For each phase write what was done, why, which files, the key commands, the real results (numbers copied from `reports/`), problems hit, and decisions made.
- `docs/ARCHITECTURE.md`: a Mermaid diagram of the full pipeline (video, MediaPipe landmarks, sequence model, top 3, confirmation, sentence, speech) and one short paragraph per component.
- `docs/REPORT.md`: final results with tables and figures, the vocabulary with its source datasets, and the evaluation protocol described honestly.
- `docs/LIMITATIONS.md`: short vocabulary only, isolated words and not continuous signing, signers and regional variants differ across the Arab world, accuracy on new signers can be lower than on the training data, not a replacement for a human interpreter, and the recommendation to test with a deaf community member or sign language interpreter before real use.
- `CREDITS.md`: dataset citations required by CC BY 4.0, MediaPipe, and font licenses.
- `README.md`: how to install, run, test, and deploy, and how to reproduce training from scratch.

## 5. Definition of done

- [ ] `docs/vocab_audit.md` exists and `config/vocab.json` is consistent with the datasets
- [ ] Held-out-signer metrics saved in `reports/`, targets met or the shortfall reported honestly
- [ ] Rejection threshold chosen and used in the app
- [ ] Feature parity test and browser end-to-end test pass (at least 95 percent agreement)
- [ ] The app runs offline after first load, in Arabic RTL, on desktop and Android Chrome
- [ ] Top-3 confirmation, sentence strip, composer, and speech all work, with a large-text fallback if no Arabic voice exists
- [ ] All documents in Phase 6 written from real outputs
- [ ] Everything committed, with no secrets or raw data in git
- [ ] Deployment prepared (Vercel, Netlify, or GitHub Pages), not published

## 6. Final message to me

When finished, reply briefly with: the chosen vocabulary, the real held-out accuracy numbers, anything that fell short, how to run the app, and the list of documents I should bring back to make the slides. No step-by-step recap.
