# Effortlessness Analyzer

A full-stack web app that watches a video of someone speaking and scores
**how physically effortless or strained their face looks**, with a
per-method breakdown, a timeline, and concrete insights.

Under the hood it pulls FACS Action Units out of every frame with
[py-feat](https://py-feat.org), tracks landmarks / head-pose / blinks with
mediapipe, learns a per-person baseline from the first 30 s, and runs five
independent scoring methods that are then blended into a single weighted
ensemble score.

---

## What it actually computes

A presenter's face leaks effort through five different channels. Each is
implemented as its own scoring method, and each returns a 0–1 score. The
final number is a weighted average:

| # | Method               | What it captures                                                         | Tier | Weight |
|---|----------------------|--------------------------------------------------------------------------|:----:|:------:|
| 1 | **Tension AUs**      | Mean activation of AU4/7/10/43 above baseline → brow & lid strain        |  4   |  0.15  |
| 2 | **Micro-expressions**| 40–200 ms bursts in AU4/7/17/23 that "leak" through a held expression    |  3   |  0.20  |
| 3 | **Asymmetry**        | Left/right activation mismatch — genuine emotion sits in 0.08–0.30       |  2   |  0.25  |
| 4 | **Coherence**        | Lag between upper-face and lower-face AU envelopes (smile-before-eyes)   |  2   |  0.25  |
| 5 | **Drift + blink**    | Blink rate vs. 12–20/min norm + AU variance in "quiet" frames            |  3   |  0.15  |

The label scale: `≥0.80 Highly effortless · ≥0.65 Effortless · ≥0.50 Moderate
effort · ≥0.35 Noticeable strain · else High strain`.

---

## Architecture

```
                ┌────────────────────┐
   video  ───►  │   FastAPI upload   │   POST /api/analyse
                └─────────┬──────────┘
                          │ background task
                          ▼
   ┌──────────────────────────────────────────────────────────┐
   │  pipeline/extractor.py        decode @ 12 fps, downscale  │
   │  pipeline/au_detector.py      py-feat → AU intensities    │
   │  pipeline/face_tracker.py     mediapipe → pose, EAR, lm   │
   │  pipeline/baseline.py         first-30s per-person stats  │
   │  pipeline/methods/*.py        5 scoring methods           │
   │  pipeline/aggregator.py       weighted ensemble + timeline│
   │  pipeline/insights.py         natural-language findings   │
   └──────────────────────────────────────────────────────────┘
                          │ JSON result
                          ▼
                ┌────────────────────┐
   browser ◄──  │  React dashboard   │   GET /api/result/{id}
                └────────────────────┘
```

### Project layout

```
facial_muscle_tension/
├── backend/
│   ├── main.py                    # FastAPI app + background job orchestration
│   ├── models.py                  # Pydantic response shapes
│   ├── requirements.txt
│   ├── .env.example
│   └── pipeline/
│       ├── extractor.py           # ffmpeg/OpenCV frame decode + downscale
│       ├── au_detector.py         # py-feat AU detection wrapper
│       ├── face_tracker.py        # mediapipe landmarks, head pose, EAR, blinks
│       ├── baseline.py            # per-person calibration (first 30 s)
│       ├── aggregator.py          # weighted ensemble + timeline
│       ├── insights.py            # human-readable insights
│       └── methods/
│           ├── tension_aus.py     # Method 1
│           ├── micro_expr.py      # Method 2
│           ├── asymmetry.py       # Method 3
│           ├── coherence.py       # Method 4
│           └── drift_blink.py     # Method 5
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── .env.example
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── api.js
│       ├── index.css
│       └── components/
│           ├── Upload.jsx
│           ├── Dashboard.jsx
│           ├── Timeline.jsx
│           ├── AUBarChart.jsx
│           ├── MethodScorecard.jsx
│           ├── InsightList.jsx
│           └── StatGrid.jsx
└── scripts/
    ├── setup.sh                   # one-shot setup
    └── dev.sh                     # run backend + frontend together
```

---

## Prerequisites

- **Python 3.10 or 3.11** — **strict.** py-feat, mediapipe, and numexpr do not
  yet have wheels for 3.12+, and they cannot build from source cleanly. On
  macOS: `brew install python@3.11`. On other systems: `pyenv install 3.11.9`.
- **Node 18+** and **npm**
- **ffmpeg** on `$PATH` (`brew install ffmpeg` on macOS)
- ~2 GB free disk for model weights (downloaded on first run)
- Optional: an Nvidia GPU + CUDA-enabled torch dramatically speeds AU detection

---

## Quick start

```bash
git clone <this repo>
cd facial_muscle_tension

./scripts/setup.sh          # creates backend/.venv, installs Python + npm deps
./scripts/dev.sh            # starts backend + frontend (logs to backend.dev.log / frontend.dev.log)
```

Then open **http://localhost:5173** and drop in a video.

### Step-by-step (if you'd rather drive it manually)

```bash
# Backend
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --port 8000

# Frontend (in another terminal)
cd frontend
npm install
cp .env.example .env
npm run dev
```

First request will be slow — py-feat downloads ~500 MB of model weights to
`~/.cache/feat/` and `~/.cache/torch/hub/`. Subsequent runs reuse those.

---

## How to use it

1. Open `http://localhost:5173`.
2. Drag in any mp4/mov/webm/mkv. For best results: clear face, ≥ 30 s long
   so baseline calibration has enough material.
3. Watch the progress bar (extracting frames → AUs → tracking → scoring →
   aggregating → insights). On a Mac M-series, expect ~0.5 s per second of
   12-fps video on CPU.
4. The dashboard shows:
   - **Overall score** + label and a per-method breakdown
   - **Video player** synced to the timeline — click micro-expression markers
     or the table to jump
   - **Effortlessness timeline** (per-second) with tension overlay and
     micro-expression dots
   - **AU bar chart** of mean activations (tension AUs red, positive-affect
     green, neutral blue)
   - **Key insights** (Good / Watch / Info)
   - **Stats grid** (frames, micro-events, blink rate, asymmetry)
   - **Micro-expression event table** with jump-to-time links

---

## API

### `POST /api/analyse`

Multipart form-data with a single `file` field (the video). Returns:

```json
{ "job_id": "f5d2b7c4-…" }
```

### `GET /api/result/{job_id}`

Returns the live job status. Statuses: `queued` · `processing` · `done` ·
`error`. While processing, `stage` and `progress` (0–1) update.

When `status: "done"` the `result` field looks like:

```json
{
  "job_id": "…",
  "filename": "demo.mp4",
  "duration_secs": 92.5,
  "frames_analysed": 1110,
  "fps": 12,
  "overall_score": 0.73,
  "overall_label": "Effortless",
  "method_scores": {
    "tension_aus":       { "method": "tension_aus", "tier": 4, "weight": 0.15, "score": 0.81, … },
    "micro_expressions": { … },
    "asymmetry":         { … },
    "coherence":         { … },
    "drift_blink":       { … }
  },
  "timeline":               [{ "second": 0, "effortless": 0.82, "tension": 0.31 }, …],
  "au_means":               { "AU01": 0.41, "AU04": 1.12, … },
  "au_stds":                { "AU01": 0.22, … },
  "micro_expression_events":[{ "timestamp": 14.6, "timestamp_str": "0:14", "duration_ms": 167, "intensity": 1.4, "dominant_au": "AU04" }, …],
  "blink_rate_per_min":     16.4,
  "asymmetry_mean":         0.18,
  "insights":               [{ "type": "good", "text": "…" }, …],
  "baseline":               { "calibration_frames": 360, "blink_rate_per_min": 15.2 },
  "warnings":               []
}
```

### `GET /api/health`

Liveness probe — returns target FPS and number of in-memory jobs.

### `DELETE /api/result/{job_id}`

Forget a finished job.

---

## Tuning knobs

All configurable via `backend/.env`:

| Var | Default | Purpose |
| --- | --- | --- |
| `TARGET_FPS`   | `12`                          | Frames per second to decode. 12 is fast + accurate; bump to 24 for max fidelity. |
| `UPLOAD_DIR`   | `/tmp/effortlessness_uploads` | Where uploads land. Deleted after processing unless `KEEP_UPLOADS=1`. |
| `KEEP_UPLOADS` | `0`                           | Set to `1` to retain uploaded videos (useful for debugging). |
| `MAX_UPLOAD_MB`| `500`                         | Reject larger uploads. |
| `CORS_ORIGINS` | `*`                           | Comma-separated origins to allow. |
| `LOG_LEVEL`    | `INFO`                        | `DEBUG` to see py-feat / mediapipe chatter. |

---

## Implementation notes

These are the things that bit during the build — kept here so they don't
have to be re-discovered.

1. **py-feat is slow on CPU.** Detection dominates total runtime. We:
   - decode at 12 fps by default (good enough for AU dynamics);
   - downscale frames so the long edge ≤ 720 px;
   - run py-feat in batches of 8 frames written to a temp dir.
2. **py-feat column names drift between releases.** `au_detector.py` runs a
   canonicaliser that turns `AU1` → `AU01` and strips bilateral `_l`/`_r`
   suffixes. `CANONICAL_AUS` is the set we rely on downstream.
3. **Multiple faces in one frame** — py-feat returns one row per detected
   face. We keep the largest face (by bounding-box area).
4. **Head-pose filtering.** `face_tracker.py` exposes pitch/yaw/roll via
   `cv2.solvePnP` — downstream methods can choose to ignore frames where
   `|yaw| > 30°`, but with py-feat's retinaface front-end this is rarely
   needed in practice.
5. **Bilateral AUs.** Only some py-feat backbones provide bilateral
   variants. `asymmetry.py` has a bilateral branch and a proxy branch and
   picks automatically.
6. **Short videos.** Anything < 30 s falls back to using the first 25 % of
   the video for baseline calibration; a `warning` is surfaced to the UI.
7. **Memory.** Frames are dropped from RAM as soon as AU + tracking are
   done, before scoring starts.
8. **Background tasks vs. workers.** The current backend uses FastAPI's
   `BackgroundTasks` (one in-process threadpool). Fine for one user; for
   concurrent users move the pipeline to Celery / RQ and back the job store
   with Redis.
9. **No data leaves the machine.** Uploads land on local disk, are deleted
   after processing, and only JSON metrics ever cross the wire.

---

## Roadmap (Phase 2 ideas baked into the architecture)

- **Audio coherence**: cross-correlate librosa pitch/energy with the
  effortlessness timeline (mismatches between "voice intensity" and
  "facial calm" are a great deception probe).
- **OpenFace 2.0 swap**: drop py-feat for OpenFace where available — better
  bilateral AUs and Gaussian-process AU intensity.
- **PDF export** with weasyprint.
- **Batch / leaderboard mode** for comparing presenters.
- **Live mode** via WebRTC for real-time coaching.

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'numpy'` while building numexpr,
  or any `× Getting requirements to build wheel did not run successfully.`
  during `pip install`** — you're on Python 3.12+ (probably 3.14). py-feat
  and its deps don't have wheels for that yet, so pip falls back to a
  source build that fails. Fix:

  ```bash
  brew install python@3.11
  rm -rf backend/.venv
  ./scripts/setup.sh        # finds python3.11 automatically
  ```

  Or point setup at an explicit interpreter:

  ```bash
  PYTHON=/opt/homebrew/opt/python@3.11/bin/python3.11 ./scripts/setup.sh
  ```

- **`ModuleNotFoundError: feat`** — the package is published as `py-feat`
  on PyPI but imports as `feat`. `pip install py-feat` (already in
  `requirements.txt`).
- **py-feat hangs on first run** — it's downloading model weights.
  Watch `~/.cache/feat/` to confirm progress.
- **`cv2.error: !_src.empty()`** — your video has no decodable frames.
  Re-encode with `ffmpeg -i bad.mov -c:v libx264 -pix_fmt yuv420p ok.mp4`.
- **Tailwind classes look unstyled** — make sure you ran `npm install` in
  `frontend/`; Tailwind is built at dev-server start.
- **CORS errors in the browser** — set `CORS_ORIGINS` in `backend/.env` to
  your dev URL, e.g. `http://localhost:5173`.
- **Slow** — drop `TARGET_FPS` to 8, or run on a GPU. py-feat reads `CUDA`
  if torch sees it.

---

## License & data

Local-only by design. Model weights belong to py-feat / mediapipe upstream.
Do not run this on anyone's face without their consent — the tool produces
plausible-sounding insights but is a *signal*, not a verdict.
