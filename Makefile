# signbench Makefile — single user entry point per PROJECT_PLAN.md §5 / REQ-makefile-orchestration.
# Phase 1: `setup` is functional; all other targets are stubs printing "TODO: <target>".
# Subsequent phases (3–8) replace stubs incrementally.

.PHONY: setup clone-reference data train-rf train-cnn train-stn train-all eval-gtsrb eval-taiwan report all clean clean-all
.DEFAULT_GOAL := setup

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
	@echo "TODO: clone-reference"

data:
	@echo "TODO: data"

train-rf:
	@echo "TODO: train-rf"

train-cnn:
	@echo "TODO: train-cnn"

train-stn:
	@echo "TODO: train-stn"

train-all:
	@echo "TODO: train-all"

eval-gtsrb:
	@echo "TODO: eval-gtsrb"

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
