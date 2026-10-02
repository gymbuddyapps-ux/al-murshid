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
