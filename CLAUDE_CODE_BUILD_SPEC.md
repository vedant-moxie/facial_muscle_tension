# Presence Analyzer — Claude Code Build Specification

> **For Claude Code:** This is an executable build manual. Work through phases sequentially. Each phase has explicit deliverables, file paths, and acceptance criteria. **Hit the performance budget at every phase** — if a phase ships slower than spec, fix it before moving on. Do not skip benchmarks.

---

## 0. Project goal and non-negotiable constraints

**Goal.** Build a CLI tool that ingests creator videos (mp4/mov/webm), extracts pose features, computes four validated presence metrics, and produces a per-video score with confidence bounds. Must scale to batches of hundreds of videos overnight on a laptop.

**Performance budget (hard targets, measured on M2 Pro / equivalent CPU laptop, no GPU):**

| Operation | Target | Hard limit |
|---|---|---|
| Pose extraction, 60-sec video | ≤ 8 sec | ≤ 15 sec |
| Full pipeline, 60-sec video | ≤ 12 sec | ≤ 20 sec |
| Batch of 100 videos (parallel) | ≤ 10 min | ≤ 20 min |
| Cold start (model load) | ≤ 3 sec | ≤ 5 sec |
| Warm start (cached pose) | ≤ 0.5 sec | ≤ 1 sec |

**Correctness budget:** test-retest reliability ≥ 0.98 on identical inputs. Metric scores deterministic given a fixed seed.

**Non-negotiables:**
1. All hot loops vectorized with numpy. Zero per-frame Python loops in metric computation.
2. Pose results cached on disk keyed by video content hash. Re-runs hit cache.
3. Feature store is Parquet, not CSV or JSON.
4. Every video produces a structured result object with explicit data-quality flags.
5. Single-source-of-truth config via pydantic. No magic numbers in code.

---

## 1. Environment setup

Use **uv** for package management. It is 10-100× faster than pip for installs and resolves; non-negotiable for iteration speed.

```bash
# Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project
uv init presence-analyzer --python 3.11
cd presence-analyzer

# Add core dependencies
uv add mediapipe==0.10.18
uv add opencv-python-headless==4.10.0.84
uv add decord==0.6.0
uv add numpy==1.26.4
uv add scipy==1.13.1
uv add numba==0.60.0
uv add polars==1.12.0
uv add pyarrow==17.0.0
uv add pydantic==2.9.2
uv add pydantic-settings==2.6.1
uv add typer==0.13.0
uv add rich==13.9.4
uv add joblib==1.4.2
uv add tqdm==4.66.6
uv add ffmpeg-python==0.2.0
uv add scikit-learn==1.5.2

# Dev dependencies
uv add --dev pytest==8.3.3 pytest-benchmark==4.0.0 ruff==0.7.4 mypy==1.13.0
```

**Why these choices:**
- `decord` over `cv2.VideoCapture` — 3-5× faster random-access video decoding; critical for any non-sequential frame sampling
- `opencv-python-headless` over full `opencv-python` — smaller, no Qt deps, faster import
- `polars` over `pandas` — 5-10× faster for the feature store operations we'll do
- `numba` — JIT compilation for the One Euro filter and SPARC computation hot paths
- `pyarrow` — required by polars for Parquet I/O
- `ffmpeg-python` — wraps system ffmpeg for fast video standardization

**System dependency check.** ffmpeg must be installed and on PATH:
```bash
ffmpeg -version || (echo "Install ffmpeg first" && exit 1)
```

---

## 2. Repository structure

Create exactly this structure. Claude Code: create directories and empty files now, then fill them per phase.

```
presence-analyzer/
├── pyproject.toml                  # managed by uv
├── README.md
├── config/
│   └── default.yaml                # default scoring weights, thresholds
├── src/
│   └── presence/
│       ├── __init__.py
│       ├── config.py               # pydantic config models
│       ├── cli.py                  # typer CLI entry point
│       ├── io/
│       │   ├── __init__.py
│       │   ├── video.py            # decord-based video reader, hashing
│       │   └── cache.py            # pose result cache (Parquet)
│       ├── pose/
│       │   ├── __init__.py
│       │   ├── extractor.py        # MediaPipe pose extraction
│       │   └── quality.py          # frame-level quality flags
│       ├── filters/
│       │   ├── __init__.py
│       │   └── one_euro.py         # vectorized One Euro filter (numba)
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── shoulder.py         # shoulder symmetry
│       │   ├── head.py             # head pose (face landmarks + PnP)
│       │   ├── arms.py             # elbow angle variance + gesture stats
│       │   └── fluidity.py         # SPARC smoothness (numba)
│       ├── scoring/
│       │   ├── __init__.py
│       │   ├── aggregate.py        # per-video aggregation
│       │   └── model.py            # regression model load + score
│       └── pipeline.py             # orchestrator
├── tests/
│   ├── test_filters.py
│   ├── test_metrics.py
│   ├── test_pipeline.py
│   └── data/                       # small test videos
├── scripts/
│   ├── bench_pipeline.py           # performance benchmarks
│   ├── train_scorer.py             # train scoring regression
│   └── batch_analyze.py            # parallel batch runner
└── data/
    ├── cache/                      # pose extraction cache
    ├── features/                   # per-video Parquet feature files
    └── models/                     # trained scoring models
```

---

## 3. Phase 1 — Configuration and CLI skeleton (30 min)

### Files to create

**`config/default.yaml`:**
```yaml
pose:
  model: heavy                      # lite | full | heavy
  min_pose_confidence: 0.5
  min_visibility: 0.5
  sample_fps: 12                    # frames sampled per second from source
quality:
  min_usable_frame_fraction: 0.60
  max_camera_roll_deg: 15
filters:
  one_euro_min_cutoff: 1.0
  one_euro_beta: 0.007
metrics:
  rolling_window_seconds: 5
scoring:
  model_path: data/models/scorer_v1.joblib
  fallback_weights:                 # used if no model present
    shoulder: 0.20
    head: 0.20
    arms: 0.30
    fluidity: 0.30
cache:
  dir: data/cache
  enabled: true
runtime:
  num_workers: 4                    # batch parallelism
  use_gpu: auto                     # auto | true | false
```

**`src/presence/config.py`** — pydantic models with strict validation. All numbers come from here, nowhere else.

**`src/presence/cli.py`** — typer app with three commands:
- `analyze <video>` — single video, prints rich-formatted result
- `batch <directory>` — parallel batch, writes Parquet result table
- `bench` — runs performance benchmarks, prints results

### Acceptance criteria
- `uv run presence analyze --help` works
- `uv run presence bench` runs (will be empty, fine for now)
- Config loads from YAML, validates, errors clearly on bad input

---

## 4. Phase 2 — Video I/O with caching (1-2 hours)

### Performance principle
**Read videos with decord, not OpenCV.** decord provides batched random-access decoding via FFI, with internal threading. For our pipeline this is 3-5× faster.

### `src/presence/io/video.py`

Implement:
- `compute_content_hash(path: Path) -> str` — SHA-256 of first MB + last MB + filesize. Fast (no full read) and deterministic.
- `standardize_video(input: Path, output: Path, target_fps: int) -> VideoMetadata` — uses ffmpeg-python to re-encode to constant frame rate. Skip if already at target fps. Output stored in `data/cache/standardized/<hash>.mp4`.
- `VideoReader` class wrapping `decord.VideoReader`:
  - `get_frame_indices(target_fps: int) -> np.ndarray` — returns indices to sample
  - `read_batch(indices: np.ndarray) -> np.ndarray` — returns (N, H, W, 3) uint8 array
  - Use `decord.bridge.set_bridge('native')` for raw numpy output (faster than torch bridge)
  - Reads in batches of 64 frames maximum — beyond that, decord allocations dominate

### `src/presence/io/cache.py`

Implement:
- `PoseCache` class with `get(hash) -> Optional[pl.DataFrame]` and `put(hash, df)`.
- Pose results stored as Parquet at `data/cache/poses/<hash>.parquet`.
- Schema: `frame_idx (int32), timestamp (float32), landmark_idx (int8), x (float32), y (float32), z (float32), visibility (float32)`.
- Polars LazyFrame for reads when only some columns needed.

### Performance test
Create `tests/test_io.py`:
```python
def test_video_read_speed(benchmark, sample_video):
    reader = VideoReader(sample_video)
    indices = reader.get_frame_indices(target_fps=12)
    result = benchmark(reader.read_batch, indices)
    # 60-sec video at 12fps = 720 frames; must read in under 2 sec
    assert benchmark.stats.stats.mean < 2.0
```

### Acceptance criteria
- 60-sec 1080p test video reads at 12fps in ≤ 2 sec
- Cache hit returns DataFrame in ≤ 50 ms
- Content hash computed in ≤ 100 ms for any file size

---

## 5. Phase 3 — Pose extraction (2-3 hours)

### Performance principle
**Two optimization tiers:**
1. Detect GPU availability (CUDA via MediaPipe, or Apple Silicon via Metal). Use GPU delegate when available.
2. Run inference in a thread pool with sequential frame submission — MediaPipe is single-threaded internally but Python+OS overhead per call can be hidden by pipelining the next frame's preparation while the previous frame infers.

### `src/presence/pose/extractor.py`

Implement `PoseExtractor`:

```python
class PoseExtractor:
    def __init__(self, config: PoseConfig):
        # Lazy load — model only loads on first call
        self._landmarker = None
        self._face_landmarker = None  # for head pose

    def extract(self, video_path: Path) -> pl.DataFrame:
        # 1. Check cache by content hash; return if hit
        # 2. Standardize video to target fps
        # 3. Read all sampled frames into memory (batch)
        # 4. Run pose detection per frame (with optional face landmarks)
        # 5. Vectorize results into a DataFrame
        # 6. Write to cache
        # 7. Return
```

**Critical performance notes:**
- Use `PoseLandmarker.create_from_options(...)` with `running_mode=VIDEO`, not `IMAGE`. VIDEO mode uses temporal tracking internally and is 2-3× faster for sequential frames.
- Pass timestamps in milliseconds, monotonically increasing.
- For each frame, call `landmarker.detect_for_video(mp_image, timestamp_ms)`.
- Build the result DataFrame in one shot at the end with `pl.from_dict(...)` and numpy arrays — never append to a DataFrame in a loop.
- Pre-allocate numpy arrays of shape `(num_frames * 33, ...)` and fill by index. Convert to Polars at the end.

### Face landmarker for head pose
Run MediaPipe Face Landmarker in parallel with Pose for frames where head pose is needed. Same caching pattern. Store 6 face landmarks needed for solvePnP (nose tip, chin, left eye corner, right eye corner, left mouth corner, right mouth corner).

### `src/presence/pose/quality.py`

Per-frame quality flags computed from raw landmarks:
- `shoulder_visible` (both shoulders > min_visibility)
- `face_visible` (nose + eyes > min_visibility)
- `arms_visible` (per arm: shoulder + elbow + wrist > min_visibility)
- `camera_roll_deg` — angle of eye line from horizontal
- `frame_usable` (composite)

Computed vectorized over the full landmark DataFrame at once with polars expressions.

### Benchmarks to hit
- Cold start (first call, model loads): ≤ 3 sec
- 60-sec video extraction, CPU only, heavy model: ≤ 8 sec
- Same with lite model: ≤ 4 sec
- Cache hit: ≤ 100 ms (just Parquet read)

### Acceptance criteria
- Identical input produces identical landmarks (deterministic)
- Cache invalidates if config (e.g., model choice) changes — include model name in cache key
- Pose DataFrame has correct schema, no nulls except where visibility < threshold

---

## 6. Phase 4 — Filters (1 hour)

### `src/presence/filters/one_euro.py`

One Euro filter vectorized over all landmarks simultaneously using numba.

```python
@numba.njit(cache=True, fastmath=True)
def one_euro_filter(
    values: np.ndarray,        # shape (T, N) — T frames, N landmark coords
    timestamps: np.ndarray,    # shape (T,)
    min_cutoff: float,
    beta: float,
) -> np.ndarray:
    # Standard 1€ filter, but vectorized across N
    # See Casiez et al. 2012
    ...
```

**Why numba:** the One Euro filter has per-timestep state dependency, so it doesn't vectorize cleanly along the time axis. But it does vectorize across landmarks. Numba compiles the inner loop to native code, giving ~30× speedup over pure Python.

### Acceptance criteria
- 720 frames × 99 coords (33 landmarks × xyz) filtered in ≤ 20 ms
- Output dimensions match input
- Compile happens once (cache=True saves to disk on first call)

---

## 7. Phase 5 — Metrics (2-3 hours)

All four metric modules take the filtered pose DataFrame and produce a fixed-schema metric output: a struct with the raw measurement, robust aggregates, and quality flags.

### `src/presence/metrics/shoulder.py`

```python
def compute_shoulder_metric(df: pl.DataFrame) -> ShoulderMetric:
    # 1. Filter to usable frames (shoulders visible, low camera roll)
    # 2. De-roll: if camera roll detected per frame, rotate shoulder positions back
    # 3. Compute shoulder line angle (atan2 of dy/dx between shoulders)
    # 4. Compute |angle| per frame
    # 5. Aggregate: median, IQR, n_frames_used
    # All in polars expressions, no Python loops
```

### `src/presence/metrics/head.py`

Head pose via PnP using face landmarks:
- Use `cv2.solvePnP` with a fixed 3D face model and the 6 detected 2D points
- Decompose rotation matrix to pitch, yaw, roll
- Estimate per-creator baseline pitch from the median of the video (or from a separate baseline call)
- Output: pitch deviation from baseline, in degrees
- Run solvePnP vectorized across frames via batched calls — solvePnP itself is per-frame but the loop is in compiled C++ via opencv

### `src/presence/metrics/arms.py`

Rolling-window gesture analysis:
- Compute elbow angle per frame per arm (vectorized — uses np.arctan2 on vector differences)
- Rolling window (5 sec): variance, gesture frequency (count of velocity peaks via scipy.signal.find_peaks), gesture amplitude (max wrist excursion)
- Aggregate across windows: median variance, median gesture frequency, median amplitude
- All in polars `rolling_var`, `rolling_apply` where needed

### `src/presence/metrics/fluidity.py`

SPARC (spectral arc length) — the rigorous smoothness measure.

```python
@numba.njit(cache=True, fastmath=True)
def sparc(
    movement: np.ndarray,      # shape (T,) — speed profile
    fs: float,                 # sampling rate
    padlevel: int = 4,
    fc: float = 10.0,
    amp_th: float = 0.05,
) -> float:
    # Pad to power of 2
    # FFT
    # Normalize amplitude spectrum
    # Compute arc length of normalized spectrum below cutoff
    # Return negative arc length (more negative = less smooth)
```

Reference: Balasubramanian et al. 2015, the SPARC implementation is ~30 lines. Apply per landmark (nose, left wrist, right wrist), then average. Compute speed profile from filtered positions, sampling rate from frame timestamps.

### Vectorization audit
For every metric module, write a one-line comment stating which numpy/polars operation does the work. If the answer is "a for loop," rewrite it.

### Acceptance criteria
- All four metrics computed on 720-frame video in ≤ 100 ms total (excluding pose extraction)
- Each metric exposes `quality_score` field (0-1) based on fraction of usable frames
- Metric outputs serializable to JSON

---

## 8. Phase 6 — Aggregation and scoring (1-2 hours)

### `src/presence/scoring/aggregate.py`

Builds the final per-video feature vector consumed by the scoring model:

```python
class VideoFeatures(BaseModel):
    shoulder_median_deg: float
    shoulder_iqr_deg: float
    head_pitch_dev_median: float
    head_pitch_dev_iqr: float
    elbow_var_median: float
    gesture_freq_per_min: float
    gesture_amplitude_median: float
    sparc_nose: float
    sparc_wrist_mean: float
    quality_shoulder: float
    quality_head: float
    quality_arms: float
    quality_fluidity: float
    duration_sec: float
    fps_effective: float
```

### `src/presence/scoring/model.py`

Two-mode scoring:

**Mode A (calibrated):** if `data/models/scorer_v1.joblib` exists, load and use it. This is a sklearn Pipeline (StandardScaler + Ridge or RandomForest). Output: score + 95% confidence interval (from bootstrap, computed at training time and stored alongside the model).

**Mode B (fallback):** linearly combine normalized metrics with `fallback_weights` from config. Mark output as `is_calibrated=False`. This exists so the pipeline produces something useful before you've collected labels.

```python
class PresenceScore(BaseModel):
    score: float                  # 0-100
    confidence_low: float
    confidence_high: float
    is_calibrated: bool
    quality_overall: float        # min of metric qualities
    components: dict[str, float]  # per-metric sub-scores for transparency
```

### Acceptance criteria
- Score is deterministic for fixed input
- Confidence interval narrower when quality is high, wider when low
- Components dict makes it possible to see *why* a video scored as it did

---

## 9. Phase 7 — Pipeline orchestration (1 hour)

### `src/presence/pipeline.py`

The single entry point that wires everything:

```python
def analyze(video_path: Path, config: Config) -> AnalysisResult:
    # 1. Validate input exists, is a video
    # 2. Compute content hash
    # 3. Pose extraction (cached)
    # 4. Apply One Euro filter to landmarks
    # 5. Compute quality flags
    # 6. Compute four metrics
    # 7. Build feature vector
    # 8. Score
    # 9. Return AnalysisResult with everything: features, score, quality, timings
```

Every stage records its duration. The result includes a `timings: dict[str, float]` field for performance debugging.

### Acceptance criteria
- End-to-end on 60-sec test video: ≤ 12 sec cold, ≤ 0.5 sec warm
- Timings dict populated and sums to ≤ total runtime
- Memory peak ≤ 1.5 GB on 60-sec 1080p video

---

## 10. Phase 8 — Batch runner (1 hour)

### `scripts/batch_analyze.py`

Parallel batch processor using `joblib.Parallel` with `loky` backend (process-based, not threads — MediaPipe and ffmpeg release the GIL but the per-worker model load benefits from process isolation).

```python
from joblib import Parallel, delayed

def batch_analyze(video_dir: Path, output: Path, n_workers: int):
    videos = list(video_dir.rglob("*.mp4")) + list(video_dir.rglob("*.mov"))

    results = Parallel(
        n_jobs=n_workers,
        backend="loky",
        verbose=10,
    )(
        delayed(analyze_one)(v, config) for v in videos
    )

    # Each result -> row in a Polars DataFrame
    # Write to Parquet
```

**Key optimization:** each worker has its own MediaPipe model instance, but the cache directory is shared. A re-run of the same batch hits the cache and completes in seconds.

### Memory budget per worker
1.5 GB per worker. `n_workers = min(cpu_count // 2, available_memory_gb // 2)`. Default 4; settable in config.

### Acceptance criteria
- 100-video batch on 8-core laptop with 4 workers: ≤ 10 min cold, ≤ 1 min warm
- Cache hit rate logged
- Failures don't abort the batch — they go to an `errors.parquet` file

---

## 11. Phase 9 — Benchmarks (30 min)

### `scripts/bench_pipeline.py`

Runs a fixed suite of test videos and emits a benchmark report. Run this before declaring any phase complete.

```python
def main():
    videos = sorted(Path("tests/data").glob("*.mp4"))

    for v in videos:
        # Cold run (clear cache for this hash)
        cache.invalidate(v)
        t0 = time.perf_counter()
        result = analyze(v, config)
        cold_time = time.perf_counter() - t0

        # Warm run
        t0 = time.perf_counter()
        result = analyze(v, config)
        warm_time = time.perf_counter() - t0

        print_row(v.name, v.duration, cold_time, warm_time, result.timings)

    # Aggregate: are we within budget?
```

Emits a table with cold time, warm time, breakdown by stage, and pass/fail vs budget.

### Acceptance criteria
- Bench completes in under 5 minutes on standard test corpus
- All videos within hard-limit budgets
- Output saved to `data/bench/<git-sha>-<timestamp>.json` for tracking regressions

---

## 12. Phase 10 — Training the scoring model (separate, when labels exist)

### `scripts/train_scorer.py`

This phase is **gated on having labeled data.** Do not skip ahead. Until you have labels, the system runs in fallback mode and that's fine.

When labels are available (CSV with `video_path, rating_1, rating_2, rating_3`):

```python
def train():
    # 1. Load labels, compute mean rating and inter-rater reliability
    # 2. Reject videos where rater disagreement > threshold
    # 3. Run pipeline on each video to get features
    # 4. Train Ridge regression with cross-validation (leave-one-creator-out)
    # 5. Bootstrap to get prediction intervals
    # 6. Save model + bootstrap distributions
    # 7. Report: Spearman correlation on held-out, feature importance, CV scores
```

### Acceptance criteria for production deployment
- Inter-rater Krippendorff's α ≥ 0.6
- Cross-validated Spearman correlation ≥ 0.5 against held-out human ratings
- Discriminant check: correlation with video resolution ≤ 0.2
- Model file ≤ 5 MB

---

## 13. Performance optimization checklist

Apply these throughout development. Run the checklist at the end of every phase.

### Hot path discipline
- [ ] No Python `for` loop iterates over frames in any metric code
- [ ] No `pandas.DataFrame.append` or `pl.DataFrame.vstack` inside loops
- [ ] No JSON parsing inside the analyze hot path
- [ ] No model instantiation inside `analyze()` — done once at startup
- [ ] Numpy arrays passed by reference, not copied, when possible

### I/O discipline
- [ ] Video read with `decord`, not `cv2.VideoCapture`
- [ ] Pose cache hit verified by content hash before any work
- [ ] Parquet for all persistent storage; never CSV in the hot path
- [ ] Standardized videos cached on disk, not regenerated

### Memory discipline
- [ ] Frame batches capped at 64 frames in memory at a time
- [ ] Pose results streamed to Parquet, not held entirely in RAM for long videos
- [ ] No accidental float64 where float32 will do — pose landmarks are 0-1 range

### Compilation discipline
- [ ] One Euro filter is `@numba.njit(cache=True)`
- [ ] SPARC computation is `@numba.njit(cache=True)`
- [ ] First call warmup happens in a `warmup()` function called from CLI entry, not on first user call

### Parallelism discipline
- [ ] Batch runner uses `loky` backend (not `threading`)
- [ ] Per-worker memory budget respected
- [ ] Cache directory is shared across workers (filesystem locking via atomic Parquet writes — write to `.tmp` then `os.rename`)

---

## 14. Testing strategy

### Unit tests
- `test_filters.py` — One Euro filter on known signals (step input, sine input, white noise). Verify it matches reference implementation outputs to 1e-5.
- `test_metrics.py` — synthetic pose sequences with known properties. A perfectly still skeleton should produce SPARC near baseline; a jittery one should produce more negative SPARC.

### Integration tests
- `test_pipeline.py` — small test video, runs end-to-end. Verifies score is deterministic, quality flags fire correctly, cache works.

### Performance tests
- `pytest --benchmark-only` runs all `benchmark`-marked tests. Fails CI if any benchmark exceeds its budget.

### Regression tests
- Golden output: pickled result for a fixed test video. Pipeline output must match within tolerance on every run. If you intentionally change scoring, update the golden output deliberately.

---

## 15. CLI reference (target UX)

```bash
# Single video
$ uv run presence analyze creator_video.mp4
┌─ Presence Analysis ─────────────────────────────────┐
│ Video: creator_video.mp4 (45.3s)                    │
│ Overall Score: 73 ± 6  [calibrated]                 │
│                                                     │
│ ├─ Shoulder Symmetry:     82  (high quality)        │
│ ├─ Head Position:         71  (high quality)        │
│ ├─ Arm Naturalness:       68  (medium quality)      │
│ └─ Movement Fluidity:     74  (high quality)        │
│                                                     │
│ Timing: 7.2s total                                  │
│   pose extraction: 5.1s | metrics: 0.08s | score: 0.01s │
└─────────────────────────────────────────────────────┘

# Batch
$ uv run presence batch ./creators/ --output results.parquet
Processing 47 videos with 4 workers...
[####################] 47/47 (cache hits: 12, errors: 1)
Wrote results.parquet (46 rows)
Wrote errors.parquet (1 row)

# Benchmark
$ uv run presence bench
Test video        | Duration | Cold   | Warm   | Budget | Status
short_30s.mp4     |  30.0s   |  4.1s  |  0.3s  |  ≤8s   | ✓
medium_60s.mp4    |  60.0s   |  7.8s  |  0.4s  |  ≤12s  | ✓
long_120s.mp4     | 120.0s   | 14.2s  |  0.5s  |  ≤24s  | ✓
All benchmarks passed.
```

---

## 16. Build order for Claude Code

Execute in this order. Run benchmarks at the end of each phase before moving on.

1. **Phase 1** — config + CLI skeleton; verify CLI runs
2. **Phase 2** — video I/O + cache; benchmark hits target
3. **Phase 3** — pose extraction; benchmark hits target
4. **Phase 4** — One Euro filter; unit tests pass
5. **Phase 5** — four metric modules; unit tests on synthetic data pass
6. **Phase 6** — scoring (fallback mode only at this stage)
7. **Phase 7** — pipeline orchestrator; integration test passes; bench hits budget
8. **Phase 8** — batch runner; bench hits batch budget
9. **Phase 9** — full benchmark script; locked-in golden outputs
10. **Phase 10** — (gated on labels) train calibrated scorer; replace fallback

After Phase 9, the system is shippable in fallback mode. Phase 10 makes it credible per the validation criteria — until labels exist, document that scores are *relative* and *uncalibrated*.

---

## 17. Things that will go wrong (preemptive notes)

**MediaPipe install on Apple Silicon.** Use Python 3.11 specifically. 3.12 and 3.13 have intermittent wheel issues. The version pin above is tested.

**decord on Apple Silicon.** Newer decord wheels exist but the 0.6.0 pin is the most stable. If you hit issues, fall back to `pyav` — slower but reliable.

**MediaPipe heavy model first download.** It pulls ~30 MB on first run. The model loader silently retries on network failure but can hang. Add an explicit timeout and a clear error message.

**ffmpeg version mismatches.** Standardization fails silently if ffmpeg is older than v4.4. Check version at startup, error out if too old.

**Polars schema strictness.** Polars will refuse to concat DataFrames with mismatched schemas (unlike pandas, which silently coerces). This is a feature, not a bug — but it bites the first time. Always cast columns to canonical types before any concat.

**Numba JIT compilation time.** First call to a `@njit` function takes 1-3 seconds to compile. Use `cache=True` so subsequent runs skip this. Trigger warmup explicitly on CLI startup so user-facing latency is consistent.

**MediaPipe video mode timestamps.** Must be strictly monotonically increasing in milliseconds. If you skip frames, the timestamp must still increase. Off-by-one here causes the tracker to reset and degrades temporal stability — your fluidity scores will silently get worse.

**solvePnP failure modes.** Returns success=False when the points are degenerate. Always check the success flag, fall back to NaN, and let the quality flags handle it.

**Caching gotcha.** Cache key MUST include: video content hash, pose model name, sample fps, MediaPipe library version. Otherwise a config change silently uses stale poses.

---

## 18. Done definition

The system is complete when:

1. All 10 phases shipped, all acceptance criteria met
2. `uv run presence bench` passes with all videos under hard-limit budgets
3. 100-video batch completes overnight in under 20 min
4. Test suite passes including benchmarks
5. README documents: install, single-video usage, batch usage, interpreting scores, calibration status
6. If labels exist: Spearman ≥ 0.5 on held-out, model file committed, README states "calibrated against N human ratings"
7. If labels don't exist: README clearly states "uncalibrated, scores are relative not absolute"

This is the shippable bar. Anything below it should be marked as "in development."
