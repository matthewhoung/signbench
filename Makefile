# signbench Makefile — single user entry point per PROJECT_PLAN.md §5 / REQ-makefile-orchestration.
# Phase 1: `setup` is functional; all other targets are stubs printing "TODO: <target>".
# Subsequent phases (3–8) replace stubs incrementally.

.PHONY: setup clone-reference data train-rf train-cnn train-stn train-all eval-gtsrb eval-taiwan report all clean clean-all
.DEFAULT_GOAL := setup

# --- Phase 3 provisioning variables -----------------------------------------
# clone-reference: read-only architectural source of truth (DEC-004).
REFERENCE_REPO := https://github.com/hello2all/GTSRB_Keras_STN.git
REFERENCE_DIR  := reference/GTSRB_Keras_STN

# data: GTSRB pickles. GTSRB_ZIP is overridable — e.g.
#   make data GTSRB_ZIP=/mnt/c/Users/Matt/Downloads/traffic-signs-data.zip
GTSRB_DIR := data/gtsrb
GTSRB_ZIP ?= traffic-signs-data.zip
GTSRB_URL := https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip
# ----------------------------------------------------------------------------

# --- GPU: TensorFlow's pip-installed CUDA wheels are not on the dynamic-linker
# path by default. Compute the nvidia/*/lib dirs and prepend them so every
# recipe's child python process sees libcudnn/libcublas/etc. This MUST be set
# BEFORE python starts — it cannot be fixed after `import tensorflow`. Declared
# with `export` at file scope so it lands in EVERY recipe's environment in one
# place (no per-recipe prefix to keep in sync). Harmless for CPU-only recipes
# (rf_hog / data / setup ignore it).
NVIDIA_LIBS := $(shell ls -d .venv/lib/python3.11/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH := $(NVIDIA_LIBS)$(LD_LIBRARY_PATH)
# ----------------------------------------------------------------------------

setup:
	@PINNED=$$(cat .python-version); \
	ACTUAL=$$(python --version 2>&1 | awk '{print $$2}'); \
	if [ "$$PINNED" != "$$ACTUAL" ]; then \
		echo "Python version mismatch: .python-version=$$PINNED but python --version=$$ACTUAL"; \
		echo "Run: asdf install python $$PINNED && asdf local python $$PINNED"; \
		exit 1; \
	fi
	@echo "Python $$(cat .python-version) verified."
	uv sync

clone-reference:
	@if [ -d "$(REFERENCE_DIR)/.git" ]; then \
		echo "Reference repo already present at $(REFERENCE_DIR)/ — skipping (DEC-004: read-only)."; \
	else \
		echo "Cloning $(REFERENCE_REPO) (shallow, read-only)..."; \
		git clone --depth 1 "$(REFERENCE_REPO)" "$(REFERENCE_DIR)"; \
	fi
	@test -f "$(REFERENCE_DIR)/README.md" && \
		echo "clone-reference OK: $(REFERENCE_DIR)/ present." || \
		{ echo "clone-reference FAILED: expected files missing."; exit 1; }

data:
	@if [ -f "$(GTSRB_DIR)/train.p" ] && [ -f "$(GTSRB_DIR)/valid.p" ] && [ -f "$(GTSRB_DIR)/test.p" ]; then \
		echo "GTSRB pickles already present in $(GTSRB_DIR)/ — skipping."; \
	else \
		if [ ! -f "$(GTSRB_ZIP)" ]; then \
			echo "Local zip $(GTSRB_ZIP) not found — downloading from Udacity CloudFront..."; \
			curl -L -o "$(GTSRB_ZIP)" "$(GTSRB_URL)"; \
		fi; \
		echo "Extracting $(GTSRB_ZIP) into $(GTSRB_DIR)/ ..."; \
		mkdir -p "$(GTSRB_DIR)"; \
		unzip -o "$(GTSRB_ZIP)" -d "$(GTSRB_DIR)"; \
	fi
	@test -f "$(GTSRB_DIR)/train.p" && test -f "$(GTSRB_DIR)/valid.p" && test -f "$(GTSRB_DIR)/test.p" && \
		echo "data OK: train.p valid.p test.p present in $(GTSRB_DIR)/." || \
		{ echo "data FAILED: one or more pickles missing."; exit 1; }

train-rf:
	uv run python -m src.runners.train --method rf_hog

train-cnn:
	uv run python -m src.runners.train --method plain_cnn

train-stn:
	uv run python -m src.runners.train --method stn_cnn

# Phase 6: all three trainers now exist, so train-all chains them
# (PROJECT_PLAN.md §5 — "all three of the above"). Editing a stub recipe is
# not adding a target — train-all is already in the .PHONY 13-target list.
train-all: train-rf train-cnn train-stn

# Phase 5: rf_hog + plain_cnn both exist, so this recipe runs both sequentially.
# Phase 6 appends a third (stn_cnn) line. Editing a recipe is not adding a
# target — DEC-005-compliant.
eval-gtsrb:
	uv run python -m src.runners.eval_gtsrb --method rf_hog
	uv run python -m src.runners.eval_gtsrb --method plain_cnn
	uv run python -m src.runners.eval_gtsrb --method stn_cnn

eval-taiwan:
	uv run python -m src.runners.eval_taiwan

# Phase 8: report.py exists, so this recipe runs the cross-method aggregation.
# Editing a stub recipe is not adding a target — report is already in the
# .PHONY 13-target list (DEC-005).
report:
	uv run python -m src.runners.report

# Phase 8: full pipeline. `all` chains the five sub-targets in dependency order
# (PROJECT_PLAN.md §5: data -> train-all -> eval-gtsrb -> eval-taiwan -> report),
# mirroring how train-all chains its three trainers as prerequisites. The
# file-scope `export LD_LIBRARY_PATH` already covers every recipe's child
# process, so the CNN/STN retrains inside train-all get the GPU. Editing a stub
# recipe is not adding a target — all is already in the .PHONY 13-target list.
all: data train-all eval-gtsrb eval-taiwan report

clean:
	@echo "TODO: clean"

clean-all:
	@echo "TODO: clean-all"
