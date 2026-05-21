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
	@echo "TODO: train-cnn"

train-stn:
	@echo "TODO: train-stn"

train-all:
	@echo "TODO: train-all"

# Phase 4: only rf_hog exists, so --method rf_hog is hardcoded. Phases 5-6 will
# EDIT this recipe to also run plain_cnn/stn_cnn (editing a recipe is not adding
# a target — DEC-005-compliant).
eval-gtsrb:
	uv run python -m src.runners.eval_gtsrb --method rf_hog

eval-taiwan:
	@echo "TODO: eval-taiwan"

report:
	@echo "TODO: report"

all:
	@echo "TODO: all"

clean:
	@echo "TODO: clean"

clean-all:
	@echo "TODO: clean-all"
