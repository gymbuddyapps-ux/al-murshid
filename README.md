# جسر (Jisr)

A small web app that turns isolated Arabic sign language words into Arabic text and speech,
so a deaf or mute person can talk to a pharmacist. Everything runs inside the phone's browser;
no video or data leaves the device.

- Scenario: at the pharmacy (greeting, place, medicine, symptom, thanks).
- Vocabulary: 20 words from the KArSL dataset, see `config/vocab.json` and `docs/vocab_audit.md`.
- Honest accuracy numbers, measured on a signer the model never saw: `docs/REPORT.md`.
- What it cannot do: `docs/LIMITATIONS.md`.

## Folder map

| Folder | What is in it |
|---|---|
| `web/` | The app (Vite + plain JavaScript). `web/src/` is the code to read. |
| `jisr/` | Python code shared by the scripts: features, model, augmentation, metrics. |
| `scripts/` | One script per step of the pipeline, in the order listed below. |
| `config/` | Settings that both Python and the browser read: vocabulary, feature trimming, templates. |
| `models/` | MediaPipe model files and the trained sign model (`jisr_model.onnx`, `labels.json`). |
| `reports/` | Every number in the documents comes from a file here. |
| `docs/` | The documents, figures and the feature specification. |
| `tests/`, `web/tests/`, `web/e2e/` | Python unit tests, JavaScript unit tests, browser tests. |

## Run the app

Requirements: Node.js 20 or newer.

```bash
cd web
npm install
npm run dev          # http://localhost:5173  (camera works on localhost)
```

For a production build and a local preview of it (this is what the browser tests use):

```bash
npm run build
npm run preview      # http://localhost:4173
```

On an Android phone the camera needs HTTPS. The easiest way to test locally is Chrome's port
forwarding: connect the phone by USB, open `chrome://inspect` on the computer, add
`4173 -> localhost:4173`, then open `http://localhost:4173` on the phone.

## Run the tests

```bash
# Python (features, augmentation, metrics)
.venv\Scripts\python -m pytest tests

# JavaScript (feature parity with Python, composer, sign detection, confidence)
cd web && npm test

# Browser tests: offline use, feature parity on real frames, fake-camera end to end.
# The last two need data/e2e, made by: python scripts/make_e2e_set.py
cd web && npm run test:e2e
```

Results of the browser tests are written to `reports/parity_report.json` and `reports/e2e_report.json`.
The manual checklist for desktop and Android is `docs/manual_test_checklist.md`.

## Reproduce the training from scratch

Requirements: Python 3.13, a Kaggle account, about 30 GB of free disk space, and a few hours.

1. Create the environment:
   ```bash
   py -3.13 -m venv .venv
   .venv\Scripts\python -m pip install --only-binary :all: mediapipe opencv-python numpy scikit-learn matplotlib onnx onnxruntime pytest pandas openpyxl kaggle
   .venv\Scripts\python -m pip install --only-binary :all: torch --index-url https://download.pytorch.org/whl/cpu
   ```
2. Put your Kaggle API token where the Kaggle CLI reads it (`~/.kaggle/kaggle.json`, or
   `~/.kaggle/access_token` for the newer token format). Never copy it into this folder.
3. Download the MediaPipe models (already in `models/mediapipe/`; if missing, the URLs are in
   `docs/PROCESS_LOG.md`, Phase 1).
4. Run the pipeline, one script per step:
   ```bash
   .venv\Scripts\kaggle datasets download yousefdotpy/karsl-502 -p data/raw/karsl/mirror_502   # 25 GB
   .venv\Scripts\python scripts/prepare_karsl.py        # unpack only the signs we use
   .venv\Scripts\python scripts/extract_landmarks.py    # MediaPipe on every clip (slow, resumable)
   .venv\Scripts\python scripts/build_features.py       # landmarks -> 32 x 140 features
   .venv\Scripts\python scripts/train.py                # validation rounds, baseline, final model
   .venv\Scripts\python scripts/evaluate.py             # one honest test on the held-out signer
   .venv\Scripts\python scripts/make_e2e_set.py         # data for the browser tests
   ```
   The vocabulary audit of Phase 0 is reproduced with the scripts in `scripts/audit/`.
5. `cd web && npm run build` copies the new model into the app.

All random seeds are fixed (`config/training.json`), so a rerun gives the same model.

## Deploy (prepared, not published)

The app is a static site. `vercel.json` and `netlify.toml` at the repository root are ready:

- **Vercel:** import the repository; the config builds `web/` and publishes `web/dist`.
- **Netlify:** import the repository; the config uses `web/` as the base folder.
- **GitHub Pages:** run `cd web && npm run build` and publish the `web/dist` folder (the app
  expects to live at the root of the site).

Camera access needs HTTPS, which all three provide. Nothing has been published; do that only when
you are ready to share the address.

## Documents

- `docs/PROCESS_LOG.md`: what was done in every phase, with the real numbers and problems.
- `docs/ARCHITECTURE.md`: the pipeline diagram and one paragraph per component.
- `docs/REPORT.md`: results, vocabulary, evaluation protocol.
- `docs/LIMITATIONS.md`: what the demo cannot do.
- `docs/features.md`: the exact feature rules shared by Python and the browser.
- `docs/vocab_audit.md`: every class in every dataset, and why the pharmacy scenario was chosen.
- `CREDITS.md`: dataset citation, libraries and font licenses.
