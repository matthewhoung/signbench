# report/Makefile — packaging for the final report.
#
# Decoupled from the project's main Makefile (which is locked at 13 targets
# per DEC-005) — this one only deals with bundling figures/data into the
# report directory for upload.
#
# Two bundle targets, two purposes (kept separate so they don't clobber each other):
#   make bundle        # ./analysis/             ORIGINAL filenames — for LaTeX/Prism compile
#   make agent-bundle  # ./analysis_for_agent/   <method>_-PREFIXED filenames — for agents
#                      # that flatten paths and would otherwise overwrite same-name files
#
# Both targets rsync from ../analysis/ with --delete (strict mirror, idempotent).
# `agent-bundle` runs an extra rename pass after the rsync so the renames survive
# re-bundling. `comparison/` and `taiwan/` filenames are already globally unique,
# so only files under rf_hog/, plain_cnn/, stn_cnn/ get the prefix.
#
# Usage:
#   make               # = make bundle
#   make bundle        # mirror ../analysis/ -> ./analysis/                (LaTeX-ready)
#   make agent-bundle  # mirror ../analysis/ -> ./analysis_for_agent/      (rename-prefixed)
#   make clean         # remove both bundles + LaTeX build artefacts
#   make pdf           # OPTIONAL: compile main.tex locally (pdflatex + bibtex, 4-pass)

.PHONY: bundle agent-bundle clean pdf
.DEFAULT_GOAL := bundle

SRC          := ../analysis
DST          := ./analysis
AGENT_DST    := ./analysis_for_agent
PREFIX_DIRS  := rf_hog plain_cnn stn_cnn

bundle:
	@if [ ! -d "$(SRC)" ]; then \
		echo "ERROR: $(SRC) not found — run \`make all\` from the project root first."; \
		exit 1; \
	fi
	@mkdir -p $(DST)
	@rsync -a --delete --exclude='.gitkeep' $(SRC)/ $(DST)/
	@echo "Bundled $$(find $(DST) -type f | wc -l) files from $(SRC) into $(DST)"
	@du -sh $(DST)

agent-bundle:
	@if [ ! -d "$(SRC)" ]; then \
		echo "ERROR: $(SRC) not found — run \`make all\` from the project root first."; \
		exit 1; \
	fi
	@mkdir -p $(AGENT_DST)
	@rsync -a --delete --exclude='.gitkeep' $(SRC)/ $(AGENT_DST)/
	@echo "Mirrored $(SRC) -> $(AGENT_DST); applying <method>_ prefix to per-method files..."
	@for d in $(PREFIX_DIRS); do \
		for f in $(AGENT_DST)/$$d/*; do \
			[ -e "$$f" ] || continue; \
			base=$$(basename "$$f"); \
			case "$$base" in \
				$${d}_*) : ;; \
				*) mv "$$f" "$(AGENT_DST)/$$d/$${d}_$${base}" ;; \
			esac; \
		done; \
	done
	@echo "Agent bundle ready: $$(find $(AGENT_DST) -type f | wc -l) files with globally-unique basenames"
	@du -sh $(AGENT_DST)

clean:
	@rm -rf $(DST) $(AGENT_DST)
	@rm -f *.aux *.log *.out *.toc *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz *.nav *.snm *.vrb
	@echo "Cleaned $(DST), $(AGENT_DST), and LaTeX build artefacts"

pdf:
	@command -v pdflatex >/dev/null 2>&1 || { echo "pdflatex not installed — use Prism / Overleaf, or install texlive."; exit 1; }
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex
	@echo "Built main.pdf"
