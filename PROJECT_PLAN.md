# GTSRB Traffic Sign Recognition — Capstone Project Plan

> **Audience**: this document is the planning brief. Claude Code will execute against it. No implementation code is included here intentionally — implementation decisions stay with the executor.

---

## 1. Project Overview

Reproduce a CNN-with-Spatial-Transformer-Network (STN) traffic sign classifier inspired by [hello2all/GTSRB_Keras_STN](https://github.com/hello2all/GTSRB_Keras_STN), but rebuild cleanly on a modern Python/TensorFlow stack. Train on the German GTSRB dataset, then evaluate domain transfer using a small set of locally-collected Taiwan traffic sign photos. Compare three methods under identical conditions inside a modular monolith.

**The original repo is reference-only** — clone it for architectural inspiration, do not try to run it.

### Why this structure

This project's deliverables map to a teacher's rubric with three sections. The repository structure exists to make each rubric section trivially producible from artifacts the pipeline emits, rather than requiring separate manual writing work.

| Rubric section | Points | Produced from |
|---|---|---|
| 系統設計 (System design) | 5 | Architecture description + plain-CNN ablation justifying STN |
| 方法比較 (Method comparison) | 5 | `analysis/comparison/` cross-method plots and tables |
| 技術討論 (Technical discussion) | 10 | `bottleneck.md` running log + `analysis/taiwan/` domain-shift results |

### Method roster

Three methods, each implemented as a swappable module behind a common interface:

1. **Random Forest on HOG features** — classical ML baseline, cited in the original paper at ~96% on GTSRB. Demonstrates feature engineering vs feature learning.
2. **Plain CNN** — deep learning without STN or learned color transform. Serves as the ablation that quantifies what STN actually contributes.
3. **STN + CNN (hero model)** — full architecture with Spatial Transformer Network and learned color transformation via 1×1 convolutions.

---

## 2. Environment

- **OS**: WSL2 Ubuntu (on Windows host)
- **Python version manager**: asdf
- **Python version**: 3.11.14
- **Package manager**: uv

### Setup reference (user-provided)

```
# initializing
asdf list python

# choose the version
asdf local python 3.11.14

# if version not installed
asdf list-all python
asdf install python 3.11.14

# project create — clone the empty repo
git clone git@github.com:matthewhoung/signbench.git && cd signbench
uv init --python 3.11.14

# activate venv
source .venv/bin/activate

# add packages
uv add <package>

# run
uv run python <script>
```

### Dependencies to install via `uv add`

Listed by purpose, not as install commands:

- **Deep learning**: tensorflow (2.16+ or 2.17+, includes Keras)
- **Classical ML**: scikit-learn
- **Feature extraction**: scikit-image (for HOG)
- **Image I/O**: opencv-python, pillow
- **Numerics**: numpy, pandas
- **Plotting**: matplotlib, seaborn
- **Dev/CLI**: tqdm (progress bars), click or typer (CLI ergonomics — optional)

---

## 3. Folder Structure

```
signbench/
├── .python-version              # asdf-pinned to 3.11.14
├── pyproject.toml               # uv-managed
├── uv.lock
├── Makefile                     # pipeline orchestration
├── README.md                    # short — points reader to this plan
├── PROJECT_PLAN.md              # this file
├── bottleneck.md               # running difficulty log (start day 1)
│
├── data/
│   ├── gtsrb/                   # from traffic-signs-data.zip
│   │   ├── train.p
│   │   ├── valid.p
│   │   └── test.p
│   └── taiwan/
│       ├── images/              # 5–10 user-collected photos
│       │   ├── 01_stop.jpg
│       │   ├── 02_speed30.jpg
│       │   └── ...
│       └── labels.csv           # filename, gtsrb_class_id_or_OOD, notes
│
├── reference/                   # READ-ONLY — original repo for reference
│   └── GTSRB_Keras_STN/         # cloned but never modified or run
│
├── src/
│   ├── __init__.py
│   ├── common/                  # shared infrastructure (built once, reused)
│   │   ├── __init__.py
│   │   ├── interfaces.py        # ModelStrategy ABC
│   │   ├── data_loader.py       # loads GTSRB pickles
│   │   ├── preprocess.py        # resize, normalize, augmentation (off by default)
│   │   ├── taiwan_loader.py     # loads Taiwan images + labels
│   │   ├── evaluate.py          # metrics, plots, confusion matrix
│   │   └── class_names.py       # GTSRB class id → human-readable name
│   │
│   ├── methods/                 # each method = swappable module
│   │   ├── __init__.py
│   │   ├── registry.py          # name → ModelStrategy lookup
│   │   ├── rf_hog/
│   │   │   ├── __init__.py
│   │   │   ├── model.py         # ModelStrategy implementation
│   │   │   └── config.py        # hyperparameters
│   │   ├── plain_cnn/
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   └── config.py
│   │   └── stn_cnn/
│   │       ├── __init__.py
│   │       ├── stn_layer.py     # custom Keras layer
│   │       ├── color_transform.py  # 1×1 conv color module
│   │       ├── model.py         # ModelStrategy implementation, wires everything
│   │       └── config.py
│   │
│   └── runners/                 # CLI entry points called by Makefile
│       ├── __init__.py
│       ├── train.py             # train <method>
│       ├── eval_gtsrb.py        # evaluate <method> on GTSRB test set
│       ├── eval_taiwan.py       # evaluate <method> on Taiwan images
│       └── report.py            # aggregate cross-method comparison
│
├── models/                      # saved weights / serialized estimators
│   ├── rf_hog.pkl
│   ├── plain_cnn.keras
│   └── stn_cnn.keras
│
└── analysis/                    # all outputs — single source of truth for the report
    ├── rf_hog/
    │   ├── metrics.json         # accuracy, precision, recall, f1
    │   ├── confusion_matrix.png
    │   ├── training_curves.png  # for tree-based: feature importance plot
    │   └── classification_report.txt
    ├── plain_cnn/
    │   └── (same shape)
    ├── stn_cnn/
    │   └── (same shape)
    ├── taiwan/                  # cross-method Taiwan evaluation
    │   ├── predictions.csv      # filename, method, predicted_class, confidence, ground_truth, correct
    │   ├── visualization_grid.png  # all images, all methods, red/green-coded
    │   └── summary.txt          # one-line-per-method bottom line
    └── comparison/              # cross-method aggregate
        ├── accuracy_comparison.png
        ├── loss_comparison.png
        ├── final_table.md       # markdown table ready to drop into the report
        └── per_class_accuracy.png
```

---

## 4. Module Contracts

### `src/common/interfaces.py` — `ModelStrategy` ABC

The contract every method implements. Methods:

- `name() -> str` — identifier for filenames and reports (e.g. `"rf_hog"`)
- `fit(X_train, y_train, X_val, y_val) -> history_dict` — trains the model, returns a dict of training-curve data (epoch-keyed for CNNs, single-point for RF)
- `predict(X) -> (predictions, confidences)` — returns class ids and per-prediction confidence scores
- `save(path: str) -> None` — serialize to disk
- `load(path: str) -> None` — restore from disk

### `src/common/data_loader.py`

- `load_gtsrb(data_dir) -> ((X_train, y_train), (X_val, y_val), (X_test, y_test))`
- Returns numpy arrays. Images already 32×32×3 from the pickle.

### `src/common/preprocess.py`

- `normalize(X) -> X_normalized` — to [0, 1] floats; identical pipeline for all methods
- `resize_to_32(image) -> image_32` — for Taiwan photos that arrive at arbitrary size
- Augmentation functions exist but are off by default (the original paper deliberately avoids augmentation; keep that choice unless experimenting)

### `src/common/taiwan_loader.py`

- `load_taiwan(data_dir) -> (X, y_or_OOD, filenames)`
- Reads `labels.csv`, loads each image, applies the same preprocessing as training
- Returns `y` as ints for known classes, sentinel value (e.g. `-1`) for OOD

### `src/common/evaluate.py`

- `compute_metrics(y_true, y_pred) -> dict` — accuracy, precision, recall, F1
- `plot_training_curves(history, out_path)` — accuracy and loss over epochs
- `plot_confusion_matrix(y_true, y_pred, out_path)`
- `plot_taiwan_grid(images, predictions, ground_truths, filenames, methods, out_path)` — the multi-method visualization grid
- `classification_report_text(y_true, y_pred) -> str`

### `src/common/class_names.py`

- `GTSRB_CLASSES: dict[int, str]` — maps class ids 0–42 to human-readable names. Necessary for visualization grids and the final report's per-class table.

### `src/methods/registry.py`

- `get_method(name: str) -> ModelStrategy` — lookup table; runners use this so they don't hardcode methods

### `src/methods/rf_hog/`

- HOG feature extraction via scikit-image's `hog()` function
- sklearn `RandomForestClassifier` wrapping the HOG features
- Config: HOG cell size, block size, orientations; RF n_estimators, max_depth
- Expected accuracy on GTSRB: ~96% (per cited Zaklouta et al.)

### `src/methods/plain_cnn/`

- Simple CNN: 3 conv blocks (Conv → BN → ReLU → MaxPool) → flatten → dense → softmax
- No STN, no learned color transform
- This is deliberately a "vanilla" deep learning baseline — its job is to underperform STN+CNN, not to be optimal

### `src/methods/stn_cnn/`

- `stn_layer.py`: custom Keras layer implementing the spatial transformer. Three components — localization network, grid generator, bilinear sampler. **Use the reference repo as architectural source of truth**, but rewrite in modern TF/Keras (no `keras.engine.topology`, no TF 1.x sessions).
- `color_transform.py`: RGB → 10ch (1×1 conv) → 3ch (1×1 conv) with VLReLU activation, per the original paper
- `model.py`: assembles input → STN → color transform → deep CNN classifier
- Expected accuracy on GTSRB: ~99%+ (per original paper)

### `src/runners/train.py`

CLI signature: `python -m src.runners.train --method <rf_hog|plain_cnn|stn_cnn>`

Flow:
1. Resolve method via registry
2. Load GTSRB data
3. Call `method.fit(...)`
4. Save model to `models/<method>.{pkl|keras}`
5. Plot training curves to `analysis/<method>/training_curves.png`

### `src/runners/eval_gtsrb.py`

CLI signature: `python -m src.runners.eval_gtsrb --method <name>`

Flow:
1. Load saved model
2. Load GTSRB test set
3. Predict
4. Compute metrics → `analysis/<method>/metrics.json`
5. Confusion matrix → `analysis/<method>/confusion_matrix.png`
6. Classification report → `analysis/<method>/classification_report.txt`

### `src/runners/eval_taiwan.py`

CLI signature: `python -m src.runners.eval_taiwan` (runs all methods together)

Flow:
1. Load Taiwan images + labels
2. For each method: load model, predict, record (filename, method, prediction, confidence, ground_truth, correct?)
3. Append all rows to `analysis/taiwan/predictions.csv`
4. Generate `analysis/taiwan/visualization_grid.png` — images on rows, methods on columns, predictions colored red/green
5. Write `analysis/taiwan/summary.txt`:
   - per-method: "X/N universal-class signs correct, Y/M OOD signs predicted as <something>"

### `src/runners/report.py`

CLI signature: `python -m src.runners.report`

Flow:
1. Read all `analysis/<method>/metrics.json`
2. Read all `analysis/<method>/training_curves` data (if available)
3. Produce:
   - `analysis/comparison/accuracy_comparison.png` (bar chart, all methods)
   - `analysis/comparison/loss_comparison.png` (curves for CNN methods only)
   - `analysis/comparison/per_class_accuracy.png` (heatmap)
   - `analysis/comparison/final_table.md` — markdown table ready to paste into the report

---

## 5. Makefile Targets

The Makefile is the single entry point. Names only — implementation belongs in the executor's hands.

| Target | Effect |
|---|---|
| `make setup` | Verify Python version, run `uv sync` |
| `make clone-reference` | `git clone` the original repo into `reference/` |
| `make data` | Download `traffic-signs-data.zip` if not present, extract to `data/gtsrb/` |
| `make train-rf` | Train Random Forest + HOG |
| `make train-cnn` | Train Plain CNN |
| `make train-stn` | Train STN + CNN |
| `make train-all` | All three of the above |
| `make eval-gtsrb` | Evaluate all three on GTSRB test set |
| `make eval-taiwan` | Evaluate all three on Taiwan images |
| `make report` | Generate cross-method comparison artifacts |
| `make all` | `data` → `train-all` → `eval-gtsrb` → `eval-taiwan` → `report` |
| `make clean` | Remove `analysis/` and `models/` |
| `make clean-all` | Also removes `data/gtsrb/` |

---

## 6. Implementation Phases

Phased so each phase has a verifiable endpoint. Claude Code should pause at each phase boundary and confirm acceptance criteria before moving on.

### Phase 1 — Project skeleton

1. Initialize uv project with Python 3.11.14
2. Add dependencies (Section 2)
3. Create folder structure per Section 3 (empty `__init__.py` files for now)
4. Create empty `Makefile` with target stubs that print "TODO"
5. Create `bottleneck.md` with a header — start logging immediately
6. Create minimal `README.md` pointing to this plan

**Acceptance**: `uv run python -c "import src"` succeeds. `make setup` runs without error.

### Phase 2 — Shared infrastructure

Implement the `src/common/` modules. No method-specific code yet.

- `interfaces.py` — `ModelStrategy` ABC
- `data_loader.py` — GTSRB pickle loader
- `preprocess.py` — normalize + resize
- `evaluate.py` — metrics + plots
- `class_names.py` — class id mappings

**Acceptance**: A throwaway smoke test loads `data/gtsrb/train.p` and prints shape/class distribution successfully.

### Phase 3 — Reference + data

1. `make clone-reference` clones the original repo into `reference/`
2. `make data` downloads and extracts `traffic-signs-data.zip` into `data/gtsrb/`
3. Smoke test: `data_loader.load_gtsrb()` returns expected shapes (34799 train, 4410 val, 12630 test)

**Acceptance**: Data loader returns the documented shapes. Reference repo cloned read-only.

### Phase 4 — First method: RF + HOG

Easiest method, fastest signal, validates the pipeline end-to-end before deep learning effort.

1. Implement `methods/rf_hog/`
2. Implement `runners/train.py` enough to train RF+HOG
3. Implement `runners/eval_gtsrb.py` enough to evaluate RF+HOG
4. `make train-rf` and `make eval-gtsrb` produce `analysis/rf_hog/` artifacts

**Acceptance**: Test accuracy ≥ 90% (looser than cited 96% to account for hyperparameter differences). `analysis/rf_hog/metrics.json` exists with non-empty content.

### Phase 5 — Second method: Plain CNN

1. Implement `methods/plain_cnn/`
2. Train and evaluate

**Acceptance**: Test accuracy ≥ 95%. Training curves saved.

### Phase 6 — Third method: STN + CNN

The most complex piece. Use the reference repo as the architectural source of truth, but rewrite cleanly for modern TF/Keras.

1. Implement `methods/stn_cnn/color_transform.py`
2. Implement `methods/stn_cnn/stn_layer.py`
3. Implement `methods/stn_cnn/model.py`
4. Train and evaluate

**Acceptance**: Test accuracy ≥ 98%. Visual sanity check: STN output for a sample image looks like a rectified version of the input.

### Phase 7 — Taiwan data collection + evaluation

This phase has a human-in-the-loop dependency — user collects the photos.

User tasks (cannot be delegated):
1. Take 5–10 Taiwan traffic sign photos. Mix universal signs (stop, no entry, speed limits) with Taiwan-unique signs (中文 text, local-only signs).
2. Crop each photo tightly around the sign — square aspect, sign fills most of the frame.
3. Save to `data/taiwan/images/` with descriptive filenames.
4. Create `data/taiwan/labels.csv` mapping each filename to either a GTSRB class id (for universal signs) or `OOD` (for Taiwan-unique).

Code tasks:
5. Implement `common/taiwan_loader.py`
6. Implement `runners/eval_taiwan.py`
7. `make eval-taiwan` produces full `analysis/taiwan/` artifacts

**Acceptance**: `analysis/taiwan/predictions.csv` has one row per (image × method) combination. `visualization_grid.png` is human-readable. `summary.txt` produces quotable one-liners.

### Phase 8 — Cross-method report

1. Implement `runners/report.py`
2. `make report` produces all `analysis/comparison/` artifacts

**Acceptance**: `analysis/comparison/final_table.md` renders a complete comparison table including all three methods and Taiwan results.

### Phase 9 — Writing (human)

Use the analysis artifacts to write the three rubric sections. The artifacts and the `bottleneck.md` log should make this near-mechanical.

---

## 7. The `bottleneck.md` Habit

**Start writing in this file on Phase 1, day 1.** Never let a debugging session end without logging it.

Suggested format (use whatever works — the point is volume, not polish):

```
## YYYY-MM-DD — short title of what broke

What happened:
- ...

What I tried:
- ...

What worked:
- ...

Quick reflection (optional, can be one line):
- ...
```

Categories worth flagging:
- Dependency / version conflicts
- Model not converging or NaN losses
- Data shape / preprocessing mismatches
- STN layer implementation gotchas (especially around bilinear sampling)
- Taiwan signs misclassifying in surprising ways
- RF being weirdly competitive — or not — with CNN
- HOG parameter tuning surprises

By submission day this file is the raw material for the entire 技術討論 (10pt) section.

---

## 8. Rubric Mapping

How each rubric requirement maps to deliverables produced by the pipeline:

### 系統設計 (5pt)

- **模型架構** ← `methods/stn_cnn/` module structure + visualization of the network. The plain CNN ablation gives the architecture description teeth ("here's what each component contributes").
- **損失函數說明** ← brief writeup describing categorical cross-entropy + the choice of optimizer. Reference values pulled from `analysis/stn_cnn/training_curves.png`.

### 方法比較 (5pt)

- **加入其他 machine learning 方法比較** ← `analysis/comparison/final_table.md` — three methods compared on identical splits.
- **數據分析 (accuracy/loss)** ← `analysis/comparison/accuracy_comparison.png` and `loss_comparison.png`.

### 技術討論 (10pt)

- **遇到的困難** ← entries from `bottleneck.md`.
- **問題分析與解法** ← annotated `bottleneck.md` entries (the "what worked" parts).
- **技術反思** ← combination of `analysis/taiwan/summary.txt` (domain-shift findings) and personal reflection on the modular-monolith architecture decision and what it enabled.

---

## 9. Out of Scope (Deliberate)

Listed so Claude Code doesn't go off-roading:

- Data augmentation (the original paper deliberately omits it; this project preserves that choice)
- Hyperparameter sweeping (mentioned by the original paper as future work — same here)
- Multiple STN layers (also future work in the original paper)
- Training on Taiwan images (the whole point is to test domain transfer, not fix it)
- Microservices, containers, deployment (this is a capstone, not a thesis)
- Fancy CLI frameworks beyond what's needed (no Hydra, no MLflow, no W&B unless the user explicitly asks)

---

## 10. Handoff Notes for Claude Code

- This plan is the source of truth. If something here conflicts with reflexes the agent has from training data (e.g. "use Hydra", "containerize it"), this plan wins.
- The `reference/` repo is read-only. Never modify it, never try to run it.
- Pause at each phase boundary and verify acceptance criteria before proceeding.
- When stuck, the user (Matthew) provides the human-in-the-loop input — especially for Phase 7 (Taiwan data collection).
- The `bottleneck.md` file is the user's responsibility to populate, but the agent can suggest entries when it encounters something noteworthy during implementation.