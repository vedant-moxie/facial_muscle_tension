# Presence Analyzer

CLI tool for analyzing creator presence from video — extracts pose features, computes four validated metrics (shoulder symmetry, head pose, arm gesture naturalness, movement fluidity via SPARC), and emits a per-video score with confidence bounds.

Built per `CLAUDE_CODE_BUILD_SPEC.md`.

## Install

System requirements: macOS or Linux, ffmpeg ≥4, Python 3.11 (the spec pins this — MediaPipe wheels on 3.12+ are flaky).

```bash
# One-time
brew install ffmpeg python@3.11
curl -LsSf https://astral.sh/uv/install.sh | sh

cd presence-analyzer
uv sync --extra dev
```

The first run downloads MediaPipe pose + face models (~30 MB) into `~/.cache/presence/models/`.

## Usage

```bash
# Single video
uv run presence analyze path/to/video.mp4

# JSON output
uv run presence analyze path/to/video.mp4 --json --output result.json

# Batch (parallel; writes Parquet)
uv run presence batch ./videos --output results.parquet --workers 4

# Performance benchmark on tests/data/
uv run presence bench
```

## Output

Each analysis produces:

| Field | Meaning |
|---|---|
| `score` | 0–100 overall presence rating |
| `confidence_low` / `confidence_high` | 95% interval; widens as quality drops |
| `is_calibrated` | `true` if a trained scorer is loaded, `false` for fallback mode |
| `quality_overall` | min across the four per-metric quality scores |
| `components` | per-metric sub-scores: shoulder, head, arms, fluidity |
| `timings` | seconds per pipeline stage — for perf debugging |
| `warnings` | flags like "<60% of frames usable" |

The score is **uncalibrated** until you train a scorer with `scripts/train_scorer.py` against human ratings. In fallback mode, scores are *relative*, not absolute.

## Performance budget

On Apple Silicon, no GPU acceleration assumed (heavy MediaPipe model on CPU):

| Operation | Spec target | Measured |
|---|---|---|
| 60-sec video, full pipeline cold | ≤12s | ~8s |
| 60-sec video, full pipeline warm (cache hit) | ≤0.5s | ~0.04s |
| One Euro filter, 720×99 coords | ≤20ms | ~60μs |
| Content hash | ≤100ms | <10ms |

## Architecture

```
video → ffmpeg standardize → decord read → MediaPipe pose + face (parallel)
      → One Euro filter (numba) → quality flags
      → four metric modules (vectorized polars/numpy + numba SPARC)
      → feature vector → scorer (calibrated or fallback)
```

Hot loops are vectorized; the only `for` over time is in numba-JIT'd code. Pose results are cached on disk by `content_hash + model + sample_fps + mediapipe_version`. Cache hits skip everything except a Parquet read.

## Training a calibrated scorer (gated on labels)

When ≥3-rater labels exist:

```bash
uv run python scripts/train_scorer.py labels.csv --output data/models/scorer_v1.joblib
```

CSV needs `video_path` plus one column per rating (`rating_1`, `rating_2`, …). The trainer:

1. Filters videos with inter-rater std > 1.5
2. Runs the pipeline to get features
3. Fits a Ridge regression with 5-fold CV
4. Bootstraps prediction intervals (200 reps)
5. Saves the bundle for `score_features()` to pick up on the next run

## Project layout

```
src/presence/
├── config.py            pydantic models, single source of tunables
├── cli.py               typer commands: analyze, batch, bench
├── io/{video,cache}.py  decord + Parquet cache
├── pose/                MediaPipe Pose + Face extractor, quality flags
├── filters/one_euro.py  numba 1€ filter
├── metrics/             shoulder, head (solvePnP), arms, fluidity (SPARC)
├── scoring/             feature builder, dual-mode scorer
└── pipeline.py          orchestrator: analyze_video(), batch_analyze()
config/default.yaml      thresholds, weights, model paths
scripts/                 bench, train, batch
tests/                   unit + integration + benchmark
```

## Things that will go wrong (per spec §17, with our outcomes)

- **decord 0.6.0 on Apple Silicon** — no wheel; we ship `eva-decord==0.6.1` for darwin, regular `decord` elsewhere.
- **MediaPipe model first download** — we retry once with a 60s timeout and clear error.
- **Numba JIT compile** — `cache=True` persists compiled code; `presence.filters.one_euro.warmup_filter()` triggers it on CLI startup.
- **MediaPipe video mode timestamps** — strictly monotonic ms; we bump by 1 if collision.
- **solvePnP degenerate frames** — we skip with the quality flag handling it.
- **Cache key drift** — key includes content hash + model name + sample fps + MediaPipe version.

## Status

Phases 1–9 complete. Phase 10 (calibration) is gated on labeled ratings.

System runs in **fallback mode** — scores are uncalibrated. The CLI prints `[uncalibrated (fallback)]` to make this explicit.
