# signbench

GTSRB traffic-sign classifier — capstone project comparing three methods (RF+HOG, Plain CNN, STN+CNN) under identical conditions, with a domain-transfer evaluation on Taiwan traffic-sign photos.

**The full specification lives in [PROJECT_PLAN.md](./PROJECT_PLAN.md).** Read that first.

## Quick start

```bash
make setup        # verifies Python 3.11.14, runs `uv sync`
make help         # not yet implemented — see PROJECT_PLAN.md §5 for the target list
```

## Layout

- `src/` — the modular monolith (common/, methods/, runners/) per `PROJECT_PLAN.md` §3
- `data/` — GTSRB pickles (Phase 3) and Taiwan photos (Phase 7); both gitignored
- `models/` — saved weights/estimators; gitignored
- `analysis/` — every output artifact the final report quotes from; gitignored
- `reference/` — read-only clone of [hello2all/GTSRB_Keras_STN](https://github.com/hello2all/GTSRB_Keras_STN) (architectural source of truth)
- `bottleneck.md` — running difficulty log; raw material for the 技術討論 (10pt) rubric section
