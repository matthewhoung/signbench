# Bottlenecks Log

Running record of everything that broke during this project and how it got unstuck.
The 技術討論 (10pt) rubric section is assembled from these entries — volume over polish.

Start logging on Phase 1, day 1. Never let a debugging session end without an entry here.

---

## Entry template

Copy this block when adding a new entry; replace `YYYY-MM-DD` and the title.

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

Categories worth flagging (from PROJECT_PLAN.md §7):

- Dependency / version conflicts
- Model not converging or NaN losses
- Data shape / preprocessing mismatches
- STN layer implementation gotchas (especially bilinear sampling)
- Taiwan signs misclassifying in surprising ways
- RF being weirdly competitive — or not — with CNN
- HOG parameter tuning surprises

---

<!-- Entries below this line. Most recent first. -->

## 2026-05-22 — Cross-method report forced a full retrain because training history was never persisted

What happened:
- Phase 8 needed a `loss_comparison.png` plot covering the two CNN
  methods. The data for those curves — per-epoch loss/val_loss — had
  never been saved. The earlier phases plotted `training_curves.png`
  (an image) from the in-memory `fit()` history dict and then dropped
  the dict.
- A pre-planning recon caught this before the planner started, so the
  plan added a small change to `train.py` to also dump
  `analysis/<method>/training_history.json` alongside the existing
  PNG. The catch: the new JSONs only materialise on the next retrain.
- `make all` is required to run end-to-end anyway (Phase 8 criterion
  #3), so the retrain happens "for free" — but free here means a
  ~28-minute wall-time bill, dominated by the STN's 40-epoch run.

What I tried:
- Considered reconstructing loss curves from anything on disk. There
  was nothing — the PNGs are bitmaps and the model files (`.keras`)
  don't carry training history. The data had to be regenerated.
- Considered exempting `loss_comparison.png` from the gate. No — the
  rubric's 方法比較 section needs it.

What worked:
- One additive line in `train.py` (dump `history` as JSON), plus the
  `make all` retrain. Came back with rf_hog 0.9032 (byte-identical,
  deterministic), plain_cnn 0.9678 (+0.2pp drift from Phase 5),
  stn_cnn 0.9899 (+0.6pp drift from Phase 6) — all still over their
  gates. GPU nondeterminism, expected and benign.

Quick reflection (for 技術討論):
- The recoverable lesson is about *data persistence vs. visualisation
  persistence*. An image is a one-way render — it cannot be re-mined
  later. Whenever a downstream phase might want different cuts of the
  same data, save the data, then derive the visualisation. The
  marginal cost is a tiny JSON; the cost of not doing it is a 28-min
  retrain (and, in a world without a deterministic seed, irreproducible
  numbers).
- Same lesson applied to `eval_gtsrb`: per-class accuracy was only in
  the formatted `classification_report.txt` text file. Phase 8 added
  a structured `per_class.json` for the heatmap. Same shape of fix,
  same shape of reasoning.

## 2026-05-21 — Taiwan domain shift: GTSRB accuracy did not predict transfer behaviour

What happened:
- Ran all three trained models over 8 real Taiwan traffic signs (4
  "universal" signs that map to a GTSRB class, 4 Taiwan-unique "OOD"
  signs with no GTSRB equivalent). The result inverts the GTSRB
  ranking in the way that matters most.
- Universal signs correct: rf_hog 2/4, plain_cnn 3/4, stn_cnn 2/4.
  All three read `no_entry` and `yield` correctly; `speed_60` defeated
  every model (the CNNs confidently call it 80, not 60); the hero
  STN+CNN confidently mislabels the 「停」 stop sign as "Priority road"
  at 0.99.
- OOD signs — none can be "correct" (no GTSRB class fits), so the
  story is *confidence*, not accuracy:
  - rf_hog — mean confidence **0.13**. The model is appropriately
    uncertain on signs it has never seen.
  - plain_cnn — mean confidence **0.91**. Confident, and wrong every
    time. Classic deep-network overconfidence on out-of-distribution
    input.
  - stn_cnn — mean confidence **0.65**, mixed: 0.9999 sure that a
    Chinese one-way sign is "Ahead only", but only 0.15 on the
    scooter two-stage-left-turn sign.

What I tried:
- Nothing "broke" — this is the phase's intended probe. The work was
  reading the result honestly rather than treating low Taiwan
  accuracy as a bug to fix (DEC-005 / PROJECT_PLAN §9: the point is to
  *measure* domain transfer, not to fix it by training on Taiwan).

What worked:
- Reporting confidence alongside the prediction. Accuracy alone would
  have said "all three models are bad at Taiwan signs, the end."
  Confidence reveals the real difference: the *weakest* model on
  GTSRB (RF+HOG, 90.32%) is the *safest* under domain shift — its
  confidence collapses to 0.13 on unseen signs, which a downstream
  system could threshold on. The *best* GTSRB model that is a plain
  CNN (96.55%) is the *worst* domain-transfer citizen — 0.91-confident
  garbage gives a downstream system no signal that anything is wrong.

Quick reflection (for the 技術討論 section — this is the headline):
- Higher in-distribution accuracy did not buy better out-of-
  distribution behaviour. The GTSRB leaderboard
  (rf_hog < plain_cnn < stn_cnn) does not survive the move to Taiwan.
- The STN helps with *geometry* (rotated/skewed signs) but the STN
  cannot invent a class that was never in the 43-class training set —
  a 「慢」 sign has no right answer to rectify toward.
- Calibration is a separate axis from accuracy. A model that knows
  when it doesn't know is worth more, under domain shift, than a
  model that is slightly more accurate in-distribution. The classical
  RF+HOG baseline accidentally demonstrates this best.

## 2026-05-21 — STN hero model: a custom-layer serialization landmine and a misleading sampler check

What happened:
- The STN+CNN hero model cleared the 98% gate at 98.40% test accuracy,
  completing the method ladder: RF+HOG 90.32% < plain CNN 96.55% <
  STN+CNN 98.40%. But the spatial transformer custom layer surfaced two
  real gotchas worth recording.
- **Landmine 1 — custom-layer serialization.** The natural port of the
  reference STN builds the localization sub-network lazily inside the
  layer's `build()` method. It trains perfectly. Then
  `keras.models.load_model('stn_cnn.keras')` throws "Layer was never
  built" — which would silently break `make eval-gtsrb` (eval loads the
  saved model). A model that trains fine but can't be reloaded is the
  worst kind of bug: invisible until the next phase.
- **Landmine 2 — a sampler sanity check that lied.** The bilinear
  sampler's identity-transform smoke test fed random-noise images and
  asserted the STN output matches the input within 1e-3. It measured
  5.9e-3 and "failed" — even though the affine matrix `theta` was
  *exactly* the identity `[1,0,0,0,1,0]`.

What I tried:
- Landmine 1: traced the load failure to Keras 3 not knowing how to
  rebuild a sub-layer that was never explicitly built. Confirmed a
  bare `register_keras_serializable` + `get_config` was not enough.
- Landmine 2: checked `theta` directly — exactly identity. Checked the
  sampled grid coordinates — they land on integer pixels within 1.9e-6.
  So the layer was correct; the *test* was wrong.

What worked:
- Landmine 1: three things together — (a) create the localization-net
  `Sequential` in the layer's `__init__`, not `build()`; (b) explicitly
  call `self.loc.build(input_shape)` inside the layer's `build()`;
  (c) `@keras.saving.register_keras_serializable()` + `get_config()`.
  With all three, save/load round-trips with diff 0.0 — verified.
- Landmine 2: the 5.9e-3 is bilinear round-off *amplified by the noise
  image's maximal pixel-to-pixel gaps* — adjacent random pixels differ
  by up to 1.0, so a sub-pixel coordinate slip of 6e-3 reads as a 6e-3
  output error. Re-ran the same identity check on a smooth GTSRB-like
  image: max error 1.1e-4, comfortably inside the bar. The fix was to
  the test input, not the layer (which is verbatim-correct).

Quick reflection:
- For the 系統設計 writeup: a custom Keras layer with a learned
  sub-network is not "free" to serialize. Build the children eagerly
  and build them explicitly — the lazy-build pattern that works for
  training quietly breaks deserialization.
- For 技術討論: when a numerical sanity check fails, ask whether the
  *test stimulus* is adversarial before touching the implementation.
  Random noise is the worst case for any interpolation tolerance; a
  realistic input is the honest one. The STN was right the whole time.
- Identity-initialising the localization net (`Dense(6)` with zero
  kernel and bias `[1,0,0,0,1,0]`) is non-negotiable — without it the
  STN starts by mangling the image and training never recovers.

## 2026-05-21 — TensorFlow couldn't see the GPU despite a working NVIDIA driver

What happened:
- Before planning Phase 5 (the first phase that trains a neural net),
  `tf.config.list_physical_devices('GPU')` returned `[]` — TensorFlow 2.21
  was about to fall back to CPU on a machine that has an RTX 4060.
- `nvidia-smi` worked fine (driver 591.74, CUDA 13.1, WSL2 passthrough OK),
  so the hardware and Windows driver were not the problem.

What I tried:
- Confirmed the GPU is visible to WSL2: `nvidia-smi` lists the RTX 4060,
  `/usr/lib/wsl/lib/libcuda.so` is present.
- Checked the venv: only plain `tensorflow` was installed — zero `nvidia-*`
  CUDA runtime packages. TF 2.21 needs cuDNN/cuBLAS/cuFFT/etc. as separate
  pip wheels.
- Ran `uv add 'tensorflow[and-cuda]'` — pulled 12 `nvidia-*-cu12` wheels
  (cuDNN 9, cuBLAS 12.9, …). TF *still* reported no GPU:
  "Cannot dlopen some GPU libraries."
- Inspected the wheel layout: the `.so` files land in
  `.venv/.../site-packages/nvidia/<pkg>/lib/`, but nothing adds those
  directories to the dynamic linker search path.

What worked:
- Setting `LD_LIBRARY_PATH` to the colon-joined list of every
  `site-packages/nvidia/*/lib` directory before launching Python. TF then
  reports `GPU:0` (RTX 4060 Laptop, compute capability 8.9, ~5.5 GB usable)
  and a test `tf.matmul` runs on `/device:GPU:0`.
- This will be wired into the Makefile (the orchestration layer) so
  `make train-cnn` / `train-stn` / `eval-gtsrb` get the GPU automatically —
  the runners stay CUDA-agnostic.

Quick reflection:
- Two separate gaps stacked: (1) the CUDA *runtime* wheels were never
  installed (`tensorflow` vs `tensorflow[and-cuda]`), and (2) even once
  installed, pip-wheel CUDA libs aren't on the loader path by default.
  Fixing only the first hides the second — the second failure looks
  identical to the first ("can't dlopen").
- A working `nvidia-smi` only proves the *driver* layer. The CUDA
  *runtime* (cuDNN et al.) is a separate userspace dependency the Python
  env owns. Worth stating plainly in the 系統設計 writeup.

## 2026-05-21 — RF+HOG baseline lands at 90.32%, 0.32pp above the gate

What happened:
- Phase 4's RF+HOG method had to clear a ≥90% GTSRB test-accuracy gate. The
  measured result was 90.32% (11,407 / 12,630 test images correct) — it
  passed, but only barely.
- Before planning, the config was tuned empirically against the real GTSRB
  test set. Five plausible "improvements" over the chosen config were tried,
  and not one of them helped:
  - More trees (n_estimators > 500) → no measurable gain.
  - orientations=12 instead of 9 → 89.3%, *worse* (the longer feature vector
    overfits the forest).
  - CLAHE contrast equalisation before HOG → 85%, much worse (it flattens the
    very gradients HOG keys on).
  - Concatenating a colour histogram onto the HOG vector → 90.1%, no help.
  - RandomForest class_weight='balanced' → 90.0%, slightly worse.

What I tried:
- Settled on grayscale HOG (orientations=9, 4x4 px/cell, 2x2 cells/block,
  L2-Hys norm, transform_sqrt) → a 1764-dim vector, into a 500-tree
  RandomForest (max_features='sqrt', random_state=42).
- Verified the result reproduces deterministically: re-running eval against
  the saved model gives byte-identical 0.9031670625... every time.

What worked:
- Accepting that ~90% IS the ceiling for grayscale HOG + RandomForest on
  GTSRB. The roadmap's gate was deliberately set at 90% (not the ~96% that
  Zaklouta et al. cite) precisely to leave room for this — the cited number
  uses a richer/tuned feature pipeline that DEC-005 (no hyperparameter sweep)
  rules out for this project.

Quick reflection:
- This is the intended story, not a disappointment: RF+HOG is the classical
  baseline whose job is to be *beaten* by the CNN methods. A baseline sitting
  right at 90% makes the STN+CNN jump to ~99% legible — it quantifies what
  feature *learning* buys over feature *engineering*.
- Lesson for the writeup (方法比較 / 技術討論): the interesting result isn't
  the 90.32% — it's that five hand-engineering tweaks all failed to move it.
  Hand-tuned features plateau; that plateau is the argument for the CNN.

## 2026-05-19 — asdf PATH ordering hid the 3.11.14 shim in non-login subshells

What happened:
- `make setup` succeeded in my interactive shell but failed in the Phase-1
  executor's subshell. The Makefile's Python-version gate refused to proceed
  because `which python` resolved to `~/.asdf/installs/python/3.12.12/bin/python`
  instead of the shim, even though `.python-version` and `.tool-versions` both
  pin 3.11.14.

What I tried:
- Inspected `which python` from the subshell — pointed at the 3.12.12 install
  dir, not `~/.asdf/shims/python`.
- Confirmed `.python-version` and `.tool-versions` pin 3.11.14.
- Re-ran `make setup` from a fresh login shell — passed immediately.

What worked:
- Prefixing the command with `PATH="$HOME/.asdf/shims:$PATH"` forces the shim
  lookup ahead of the version-specific install dir.
- asdf's interactive init (in `.bashrc`/`.zshrc`) already arranges PATH this
  way for login shells; non-login subshells inherit a stripped PATH that does
  not.

Quick reflection:
- The gate worked as designed — silently running on 3.12.12 would have been
  worse than a clear refusal.
- Worth keeping the Makefile gate strict. Future Colab / CI / WSL2-from-VSCode
  environments will hit the same shape of problem.
- Lesson for later phases: when shelling out to `make` or `uv` from Python,
  prepend `~/.asdf/shims` to PATH to keep the version pin honoured.

