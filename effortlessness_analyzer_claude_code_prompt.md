# Claude Code Prompt — Effortlessness Analyzer

Paste this entire prompt into Claude Code to build the product.

---

## The Task

Build a full-stack web application called **Effortlessness Analyzer**. A user uploads a video of a person speaking or presenting. The system extracts frames, detects facial Action Units (AUs) using the FACS framework, runs five analysis methods, and produces a scored dashboard showing how physically effortless or strained the person appears, with per-method breakdowns, a timeline, and key insights.

---

## Tech Stack

- **Backend**: Python 3.11, FastAPI, uvicorn
- **AU Detection**: `py-feat` (Facial Action Coding System library)
- **Face Tracking**: `mediapipe` (landmarks, head pose, blink detection)
- **Video**: `opencv-python`, `ffmpeg-python`
- **Signal Processing**: `scipy`, `numpy`, `pandas`
- **Frontend**: React 18 + Vite, Tailwind CSS, Recharts
- **File handling**: Python `aiofiles`, multipart form uploads

---

## Project Structure

```
effortlessness-analyzer/
├── backend/
│   ├── main.py                  # FastAPI app, upload endpoint
│   ├── pipeline/
│   │   ├── extractor.py         # Frame extraction with OpenCV/ffmpeg
│   │   ├── au_detector.py       # py-feat AU detection wrapper
│   │   ├── face_tracker.py      # mediapipe landmarks, head pose
│   │   ├── methods/
│   │   │   ├── tension_aus.py   # Method 1: Tension AU scoring
│   │   │   ├── micro_expr.py    # Method 2: Micro-expression bursts
│   │   │   ├── asymmetry.py     # Method 3: Asymmetry index
│   │   │   ├── coherence.py     # Method 4: Temporal coherence
│   │   │   └── drift_blink.py   # Method 5: Resting state drift + blink
│   │   ├── baseline.py          # Calibration from first 30s of video
│   │   ├── aggregator.py        # Weighted ensemble scorer
│   │   └── insights.py          # Natural language insight generator
│   ├── models.py                # Pydantic response models
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── Upload.jsx
│   │   │   ├── Dashboard.jsx
│   │   │   ├── Timeline.jsx
│   │   │   ├── AUBarChart.jsx
│   │   │   ├── MethodScorecard.jsx
│   │   │   ├── InsightList.jsx
│   │   │   └── StatGrid.jsx
│   │   └── api.js
│   ├── package.json
│   └── vite.config.js
└── README.md
```

---

## Backend — Detailed Implementation

### `requirements.txt`
```
fastapi
uvicorn[standard]
python-multipart
aiofiles
feat                  # py-feat: pip install feat
mediapipe
opencv-python-headless
ffmpeg-python
numpy
pandas
scipy
pydantic
```

---

### `main.py` — FastAPI Application

```python
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid, os, aiofiles
from pipeline.extractor import extract_frames
from pipeline.au_detector import detect_aus
from pipeline.face_tracker import track_face
from pipeline.baseline import calibrate_baseline
from pipeline.methods.tension_aus import score_tension_aus
from pipeline.methods.micro_expr import detect_micro_expressions
from pipeline.methods.asymmetry import compute_asymmetry
from pipeline.methods.coherence import compute_coherence
from pipeline.methods.drift_blink import analyse_drift_blink
from pipeline.aggregator import aggregate_scores
from pipeline.insights import generate_insights
from models import AnalysisResult

app = FastAPI(title="Effortlessness Analyzer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

results_store: dict = {}

@app.post("/api/analyse")
async def analyse_video(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    video_path = f"{UPLOAD_DIR}/{job_id}_{file.filename}"
    
    async with aiofiles.open(video_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    results_store[job_id] = {"status": "processing"}
    
    # Run full pipeline synchronously (use BackgroundTasks for production)
    try:
        result = run_pipeline(video_path, job_id)
        results_store[job_id] = {"status": "done", "result": result}
    except Exception as e:
        results_store[job_id] = {"status": "error", "message": str(e)}
    
    return {"job_id": job_id}

@app.get("/api/result/{job_id}")
def get_result(job_id: str):
    return results_store.get(job_id, {"status": "not_found"})

def run_pipeline(video_path: str, job_id: str) -> dict:
    """Full analysis pipeline."""
    fps = 24
    
    # 1. Extract frames at target FPS
    frames, timestamps = extract_frames(video_path, target_fps=fps)
    
    # 2. Run AU detection on all frames
    au_df = detect_aus(frames, fps=fps)  # DataFrame: rows=frames, cols=AUs
    
    # 3. Run face tracking (landmarks, head pose, blink detection)
    tracking_df = track_face(frames, fps=fps)
    
    # 4. Calibrate baseline from first 30 seconds
    baseline = calibrate_baseline(au_df, tracking_df, fps=fps, duration_secs=30)
    
    # 5. Run all five methods
    m1 = score_tension_aus(au_df, baseline)
    m2 = detect_micro_expressions(au_df, fps=fps)
    m3 = compute_asymmetry(au_df, baseline)
    m4 = compute_coherence(au_df, fps=fps)
    m5 = analyse_drift_blink(au_df, tracking_df, fps=fps, baseline=baseline)
    
    # 6. Aggregate
    ensemble = aggregate_scores(m1, m2, m3, m4, m5, timestamps)
    
    # 7. Generate insights
    insights = generate_insights(m1, m2, m3, m4, m5, ensemble, au_df)
    
    return {
        "duration_secs": len(frames) / fps,
        "frames_analysed": len(frames),
        "timestamps": timestamps,
        "overall_score": ensemble["overall_score"],
        "overall_label": ensemble["label"],
        "timeline": ensemble["timeline"],
        "method_scores": {
            "tension_aus": m1["summary"],
            "micro_expressions": m2["summary"],
            "asymmetry": m3["summary"],
            "coherence": m4["summary"],
            "drift_blink": m5["summary"]
        },
        "au_means": au_df.mean().to_dict(),
        "au_stds": au_df.std().to_dict(),
        "micro_expression_events": m2["events"],
        "blink_rate_per_min": m5["blink_rate_per_min"],
        "asymmetry_mean": m3["mean"],
        "insights": insights
    }
```

---

### `pipeline/extractor.py`

Extract frames from video at a controlled FPS. Use ffmpeg for format compatibility, OpenCV for frame reading.

```python
import cv2, numpy as np
from typing import Tuple

def extract_frames(video_path: str, target_fps: int = 24) -> Tuple[list, list]:
    """
    Extract frames from video at target_fps.
    Returns: (list of numpy arrays, list of timestamps in seconds)
    """
    cap = cv2.VideoCapture(video_path)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(source_fps / target_fps))
    
    frames, timestamps = [], []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamps.append(round(frame_idx / source_fps, 3))
        frame_idx += 1
    
    cap.release()
    return frames, timestamps
```

---

### `pipeline/au_detector.py`

Use `py-feat` to detect all 44 Action Units per frame. py-feat returns AU intensities (0–5 scale) and probabilities.

```python
import pandas as pd
import numpy as np
from feat import Detector

_detector = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = Detector(
            face_model="retinaface",
            landmark_model="mobilefacenet",
            au_model="xgb",          # XGBoost-based AU detector
            emotion_model="resmasknet",
            facepose_model="img2pose"
        )
    return _detector

def detect_aus(frames: list, fps: int = 24) -> pd.DataFrame:
    """
    Run py-feat AU detection on all frames.
    Returns DataFrame with AU columns (AU01–AU28) and frame index.
    """
    detector = get_detector()
    
    AU_COLUMNS = [f"AU{str(i).zfill(2)}" for i in 
                  [1,2,4,5,6,7,9,10,11,12,13,14,15,17,18,20,
                   23,24,25,26,28,43]]
    
    all_aus = []
    
    # Process in batches of 32 for efficiency
    batch_size = 32
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        try:
            result = detector.detect_image(batch)
            # Extract AU columns from result
            au_batch = result[AU_COLUMNS].fillna(0)
            all_aus.append(au_batch)
        except Exception:
            # If detection fails for a batch, append zeros
            empty = pd.DataFrame(np.zeros((len(batch), len(AU_COLUMNS))), 
                                 columns=AU_COLUMNS)
            all_aus.append(empty)
    
    df = pd.concat(all_aus, ignore_index=True)
    df["frame_idx"] = range(len(df))
    df["timestamp"] = df["frame_idx"] / fps
    return df
```

---

### `pipeline/face_tracker.py`

Use mediapipe for blink detection (eye aspect ratio), head pose estimation, and landmark coordinates for asymmetry computation.

```python
import mediapipe as mp
import numpy as np
import pandas as pd

mp_face_mesh = mp.solutions.face_mesh

# Mediapipe landmark indices for eye aspect ratio
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

def eye_aspect_ratio(landmarks, indices):
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in indices])
    # Vertical distances
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    # Horizontal distance
    h = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h + 1e-6)

def track_face(frames: list, fps: int = 24) -> pd.DataFrame:
    """
    Track face per frame: head pose, eye aspect ratio, blink events.
    Returns DataFrame with columns: timestamp, pitch, yaw, roll, left_ear, right_ear, blink
    """
    records = []
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        for i, frame in enumerate(frames):
            result = face_mesh.process(frame)
            rec = {
                "frame_idx": i,
                "timestamp": i / fps,
                "pitch": 0, "yaw": 0, "roll": 0,
                "left_ear": 0.3, "right_ear": 0.3,
                "blink": False
            }
            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0].landmark
                left_ear = eye_aspect_ratio(lm, LEFT_EYE[:6])
                right_ear = eye_aspect_ratio(lm, RIGHT_EYE[:6])
                rec["left_ear"] = left_ear
                rec["right_ear"] = right_ear
                rec["blink"] = (left_ear < 0.2 and right_ear < 0.2)
            records.append(rec)
    
    return pd.DataFrame(records)
```

---

### `pipeline/baseline.py`

Calibrate per-person baseline from the first 30 seconds of video to enable personalized deviation scoring.

```python
import pandas as pd
import numpy as np

def calibrate_baseline(au_df: pd.DataFrame, tracking_df: pd.DataFrame, 
                        fps: int = 24, duration_secs: int = 30) -> dict:
    """
    Compute baseline statistics from first N seconds of video.
    Returns dict with per-AU mean and std, blink rate, asymmetry baseline.
    """
    calibration_frames = min(duration_secs * fps, len(au_df))
    au_cal = au_df.iloc[:calibration_frames]
    tr_cal = tracking_df.iloc[:calibration_frames]
    
    au_cols = [c for c in au_df.columns if c.startswith("AU")]
    
    baseline = {
        "au_mean": au_cal[au_cols].mean().to_dict(),
        "au_std": au_cal[au_cols].std().fillna(0.1).to_dict(),
        "blink_rate_per_min": (tr_cal["blink"].sum() / duration_secs) * 60,
        "left_ear_mean": tr_cal["left_ear"].mean(),
        "right_ear_mean": tr_cal["right_ear"].mean(),
        "calibration_frames": calibration_frames
    }
    
    return baseline
```

---

### `pipeline/methods/tension_aus.py` — Method 1

Score effortlessness from weighted AU4, AU7, AU43, AU10 activations normalized against baseline.

```python
import pandas as pd
import numpy as np

# Tension AUs with weights (higher = more diagnostic for effort)
TENSION_AUS = {
    "AU04": 0.35,   # Brow lowerer — most involuntary
    "AU07": 0.30,   # Lid tightener
    "AU10": 0.20,   # Upper lip raiser
    "AU43": 0.15,   # Eyes closed (sustained)
}
MAX_ACTIVATION = 5.0  # py-feat intensity scale

def score_tension_aus(au_df: pd.DataFrame, baseline: dict) -> dict:
    """
    Compute per-frame effortlessness score from tension AUs.
    
    Formula: effortless = 1 - (sum(w_k * AU_k) / max_activation)
    Adjusted by baseline deviation.
    """
    frames = []
    
    for _, row in au_df.iterrows():
        tension = 0.0
        for au, weight in TENSION_AUS.items():
            au_val = row.get(au, 0)
            # Normalize by baseline: how much above baseline?
            baseline_mean = baseline["au_mean"].get(au, 0)
            deviation = max(0, au_val - baseline_mean)
            tension += weight * deviation
        
        effortless = 1.0 - min(1.0, tension / MAX_ACTIVATION)
        frames.append({
            "timestamp": row["timestamp"],
            "effortless": effortless,
            "raw_tension": tension,
            "AU04": row.get("AU04", 0),
            "AU07": row.get("AU07", 0),
            "AU10": row.get("AU10", 0),
            "AU43": row.get("AU43", 0),
        })
    
    df = pd.DataFrame(frames)
    
    return {
        "frame_scores": df.to_dict(orient="records"),
        "summary": {
            "method": "tension_aus",
            "tier": 4,
            "weight": 0.15,
            "score": float(df["effortless"].mean()),
            "std": float(df["effortless"].std()),
            "worst_moments": df.nsmallest(3, "effortless")[["timestamp","effortless"]].to_dict(orient="records")
        }
    }
```

---

### `pipeline/methods/micro_expr.py` — Method 2

Detect micro-expression bursts: AU4+AU7 spikes lasting 40–200ms that break through suppression.

```python
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

MICRO_EXPR_AUS = ["AU04", "AU07", "AU17", "AU23"]  # effort/suppression cluster
MIN_DURATION_MS = 40
MAX_DURATION_MS = 200

def detect_micro_expressions(au_df: pd.DataFrame, fps: int = 24) -> dict:
    """
    Detect micro-expression bursts: transient spikes in tension AUs.
    A burst is a local maximum in the combined tension signal lasting 40–200ms.
    """
    ms_per_frame = 1000 / fps
    
    au_cols = [c for c in MICRO_EXPR_AUS if c in au_df.columns]
    if not au_cols:
        return {"events": [], "summary": {"method": "micro_expressions", "tier": 3, "weight": 0.20, "score": 0.8}}
    
    # Combined signal: sum of tension AUs
    signal = au_df[au_cols].sum(axis=1).values
    
    # Rolling baseline (1-second window)
    window = fps
    rolling_mean = pd.Series(signal).rolling(window, center=True, min_periods=1).mean().values
    
    # Find peaks above baseline by threshold
    threshold = rolling_mean + np.std(signal) * 1.5
    deviation = signal - rolling_mean
    
    peaks, props = find_peaks(deviation, 
                               height=np.std(signal) * 1.5,
                               distance=int(fps * 0.2))  # min 200ms between events
    
    events = []
    min_frames = int(MIN_DURATION_MS / ms_per_frame)
    max_frames = int(MAX_DURATION_MS / ms_per_frame)
    
    for peak in peaks:
        # Estimate duration of the spike
        left = peak
        right = peak
        while left > 0 and deviation[left] > 0:
            left -= 1
        while right < len(deviation)-1 and deviation[right] > 0:
            right += 1
        
        duration_frames = right - left
        duration_ms = duration_frames * ms_per_frame
        
        if MIN_DURATION_MS <= duration_ms <= MAX_DURATION_MS:
            timestamp = au_df.iloc[peak]["timestamp"]
            events.append({
                "timestamp": float(timestamp),
                "timestamp_str": f"{int(timestamp//60)}:{int(timestamp%60):02d}",
                "duration_ms": round(duration_ms, 1),
                "intensity": float(deviation[peak]),
                "dominant_au": au_cols[np.argmax([au_df.iloc[peak].get(au, 0) for au in au_cols])]
            })
    
    # Score: fewer events = more effortless
    # Normalize: 0 events = 1.0, 10+ events = 0.0
    event_count = len(events)
    score = max(0.0, 1.0 - event_count / 10.0)
    
    return {
        "events": events,
        "event_count": event_count,
        "summary": {
            "method": "micro_expressions",
            "tier": 3,
            "weight": 0.20,
            "score": round(score, 3),
            "event_count": event_count
        }
    }
```

---

### `pipeline/methods/asymmetry.py` — Method 3

Compute facial asymmetry index. Genuine emotion = slight asymmetry (0.1–0.3). Fake/performed = over-symmetric (< 0.08).

```python
import pandas as pd
import numpy as np

# AU pairs that exist bilaterally (left/right variants)
# py-feat provides _r and _l suffixes for bilateral AUs in some models
# If unavailable, approximate using landmark x-coordinates from mediapipe

def compute_asymmetry(au_df: pd.DataFrame, baseline: dict) -> dict:
    """
    Compute asymmetry index per frame.
    
    If py-feat provides bilateral AUs (AU_r, AU_l), use those directly.
    Otherwise, estimate from overall AU variance (proxy method).
    
    Asymmetry index = |AU_left - AU_right| / (AU_left + AU_right + eps)
    Genuine range: 0.10–0.30
    Over-symmetric (performed): < 0.08
    Pathological: > 0.50
    """
    # Check if bilateral AUs are available
    bilateral_cols = [c for c in au_df.columns if c.endswith("_r") or c.endswith("_l")]
    
    records = []
    
    if bilateral_cols:
        # Direct bilateral asymmetry
        au_bases = set(c[:-2] for c in bilateral_cols)
        
        for _, row in au_df.iterrows():
            asym_vals = []
            for au in au_bases:
                left = row.get(f"{au}_l", 0)
                right = row.get(f"{au}_r", 0)
                total = abs(left) + abs(right) + 1e-6
                asym = abs(left - right) / total
                asym_vals.append(asym)
            
            mean_asym = np.mean(asym_vals) if asym_vals else 0.15
            
            # Scoring: penalize both over-symmetric (fake) and over-asymmetric (distressed)
            if mean_asym < 0.08:
                score = mean_asym / 0.08  # Under-symmetric (performed)
            elif mean_asym <= 0.30:
                score = 1.0               # Genuine range
            else:
                score = max(0, 1.0 - (mean_asym - 0.30) / 0.20)  # Too asymmetric
            
            records.append({"timestamp": row["timestamp"], "asymmetry": mean_asym, "score": score})
    else:
        # Proxy: use inter-frame AU variance as asymmetry estimate
        au_cols = [c for c in au_df.columns if c.startswith("AU")]
        for _, row in au_df.iterrows():
            # Frame-level AU variance across all AUs as proxy for natural expression variation
            au_vals = row[au_cols].values.astype(float)
            variance = np.std(au_vals)
            # Map variance to asymmetry-like score
            # Very low variance = over-controlled, very high = strain
            score = 1.0 - abs(variance - 0.5) / 0.5 if variance <= 1.0 else 0.5
            records.append({"timestamp": row["timestamp"], "asymmetry": variance, "score": min(1.0, max(0.0, score))})
    
    df = pd.DataFrame(records)
    
    return {
        "frame_scores": df.to_dict(orient="records"),
        "mean": float(df["asymmetry"].mean()),
        "summary": {
            "method": "asymmetry",
            "tier": 2,
            "weight": 0.25,
            "score": float(df["score"].mean()),
            "mean_asymmetry": float(df["asymmetry"].mean()),
            "in_genuine_range": bool(0.08 <= df["asymmetry"].mean() <= 0.30)
        }
    }
```

---

### `pipeline/methods/coherence.py` — Method 4

Compute upper/lower face temporal coherence. Genuine emotion: upper and lower face activate within ~80ms. Performed: lower face leads by 200–500ms (smile before eyes).

```python
import pandas as pd
import numpy as np
from scipy.signal import correlate

# Upper face AUs (eyes/brows)
UPPER_FACE = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07"]
# Lower face AUs (mouth)
LOWER_FACE = ["AU10", "AU12", "AU13", "AU14", "AU15", "AU17", "AU20", "AU23", "AU25", "AU26"]

GENUINE_LAG_MAX_MS = 100   # Upper and lower in sync within 100ms
PERFORMED_LAG_MS = 200     # Lower face leading by 200ms+ = performed

def compute_coherence(au_df: pd.DataFrame, fps: int = 24) -> dict:
    """
    Cross-correlate upper face signal vs lower face signal.
    Find the lag at maximum correlation.
    
    Genuine: peak correlation at |lag| < 100ms
    Performed: peak at lag 200–500ms (lower face leading)
    """
    ms_per_frame = 1000 / fps
    
    upper_cols = [c for c in UPPER_FACE if c in au_df.columns]
    lower_cols = [c for c in LOWER_FACE if c in au_df.columns]
    
    if not upper_cols or not lower_cols:
        return {"summary": {"method": "coherence", "tier": 2, "weight": 0.25, "score": 0.7}}
    
    upper_signal = au_df[upper_cols].sum(axis=1).values
    lower_signal = au_df[lower_cols].sum(axis=1).values
    
    # Normalize signals
    upper_norm = (upper_signal - upper_signal.mean()) / (upper_signal.std() + 1e-6)
    lower_norm = (lower_signal - lower_signal.mean()) / (lower_signal.std() + 1e-6)
    
    # Cross-correlation
    corr = correlate(upper_norm, lower_norm, mode="full")
    lags = np.arange(-(len(upper_norm)-1), len(upper_norm))
    
    peak_lag_frames = lags[np.argmax(corr)]
    peak_lag_ms = peak_lag_frames * ms_per_frame
    max_corr = np.max(corr) / len(upper_norm)
    
    # Scoring:
    # lag near 0 (±100ms) = genuine = high score
    # lag 200-500ms (lower leads) = performed = lower score
    abs_lag = abs(peak_lag_ms)
    if abs_lag <= GENUINE_LAG_MAX_MS:
        score = 1.0
    elif abs_lag <= PERFORMED_LAG_MS:
        score = 1.0 - (abs_lag - GENUINE_LAG_MAX_MS) / (PERFORMED_LAG_MS - GENUINE_LAG_MAX_MS)
    else:
        score = max(0.0, 0.3 - (abs_lag - PERFORMED_LAG_MS) / 1000)
    
    return {
        "peak_lag_ms": round(peak_lag_ms, 1),
        "max_correlation": round(float(max_corr), 3),
        "summary": {
            "method": "coherence",
            "tier": 2,
            "weight": 0.25,
            "score": round(score, 3),
            "peak_lag_ms": round(peak_lag_ms, 1),
            "interpretation": (
                "genuine sync" if abs_lag <= GENUINE_LAG_MAX_MS
                else "possible performed expression" if abs_lag <= PERFORMED_LAG_MS
                else "strong mismatch"
            )
        }
    }
```

---

### `pipeline/methods/drift_blink.py` — Method 5

Analyse resting state between expressions and blink rate/pattern.

```python
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

GENUINE_BLINK_MIN = 12   # blinks/min
GENUINE_BLINK_MAX = 20
EFFORT_BLINK_THRESHOLD = 10  # below = cognitive load
PERFORMED_BLINK_THRESHOLD = 9  # very low = suppression

def analyse_drift_blink(au_df: pd.DataFrame, tracking_df: pd.DataFrame, 
                         fps: int = 24, baseline: dict = None) -> dict:
    """
    Analyse blink rate, blink regularity, and AU drift between expressions.
    """
    duration_secs = len(tracking_df) / fps
    
    # --- Blink analysis ---
    blink_frames = tracking_df[tracking_df["blink"] == True]["frame_idx"].values
    
    # Filter: blinks must be at least 100ms apart (de-duplicate)
    min_gap = int(0.1 * fps)
    filtered_blinks = []
    last_blink = -min_gap
    for b in blink_frames:
        if b - last_blink >= min_gap:
            filtered_blinks.append(b)
            last_blink = b
    
    blink_count = len(filtered_blinks)
    blink_rate_per_min = (blink_count / duration_secs) * 60 if duration_secs > 0 else 0
    
    # Blink interval regularity (CV = std/mean, lower = more robotic)
    if len(filtered_blinks) > 2:
        intervals = np.diff(filtered_blinks) / fps  # intervals in seconds
        blink_cv = np.std(intervals) / (np.mean(intervals) + 1e-6)
    else:
        blink_cv = 0.5  # default to medium
    
    # --- Drift analysis: AU variance in quiet periods ---
    au_cols = [c for c in au_df.columns if c.startswith("AU")]
    au_signal = au_df[au_cols].sum(axis=1).values
    
    # Find "quiet" periods: below median activation
    median_act = np.median(au_signal)
    quiet_mask = au_signal < median_act
    
    quiet_variance = float(au_df.loc[quiet_mask, au_cols].values.std()) if quiet_mask.any() else 0.1
    
    # --- Score blink rate ---
    if GENUINE_BLINK_MIN <= blink_rate_per_min <= GENUINE_BLINK_MAX:
        blink_score = 1.0
    elif blink_rate_per_min < PERFORMED_BLINK_THRESHOLD:
        blink_score = blink_rate_per_min / PERFORMED_BLINK_THRESHOLD
    elif blink_rate_per_min < GENUINE_BLINK_MIN:
        blink_score = 0.5 + 0.5 * (blink_rate_per_min - PERFORMED_BLINK_THRESHOLD) / (GENUINE_BLINK_MIN - PERFORMED_BLINK_THRESHOLD)
    else:
        blink_score = max(0.5, 1.0 - (blink_rate_per_min - GENUINE_BLINK_MAX) / 20)
    
    # Penalize very regular blinking (robotic suppression)
    regularity_penalty = max(0, 0.3 - blink_cv) * 0.5
    blink_score = max(0.0, blink_score - regularity_penalty)
    
    # --- Score drift: some variance in quiet periods is healthy ---
    # Near-zero variance = over-controlled; moderate = genuine
    drift_score = min(1.0, quiet_variance / 0.3) if quiet_variance < 0.3 else max(0.5, 1.0 - (quiet_variance - 0.3) / 0.5)
    
    final_score = 0.6 * blink_score + 0.4 * drift_score
    
    return {
        "blink_rate_per_min": round(blink_rate_per_min, 1),
        "blink_count": blink_count,
        "blink_cv": round(float(blink_cv), 3),
        "quiet_variance": round(quiet_variance, 3),
        "summary": {
            "method": "drift_blink",
            "tier": 3,
            "weight": 0.15,
            "score": round(final_score, 3),
            "blink_rate_per_min": round(blink_rate_per_min, 1),
            "blink_regularity": "regular (possible suppression)" if blink_cv < 0.3 else "natural",
            "drift_state": "over-controlled" if quiet_variance < 0.05 else "natural"
        }
    }
```

---

### `pipeline/aggregator.py`

Weighted ensemble combining all five methods. Tier 1 methods get higher weight.

```python
import numpy as np
import pandas as pd

WEIGHTS = {
    "tension_aus":        0.15,   # Tier 4
    "micro_expressions":  0.20,   # Tier 3
    "asymmetry":          0.25,   # Tier 2
    "coherence":          0.25,   # Tier 2
    "drift_blink":        0.15,   # Tier 3
}
# Weights sum to 1.0

def label_score(score: float) -> str:
    if score >= 0.80:
        return "Highly effortless"
    elif score >= 0.65:
        return "Effortless"
    elif score >= 0.50:
        return "Moderate effort"
    elif score >= 0.35:
        return "Noticeable strain"
    else:
        return "High strain"

def aggregate_scores(m1, m2, m3, m4, m5, timestamps) -> dict:
    scores = {
        "tension_aus": m1["summary"]["score"],
        "micro_expressions": m2["summary"]["score"],
        "asymmetry": m3["summary"]["score"],
        "coherence": m4["summary"]["score"],
        "drift_blink": m5["summary"]["score"],
    }
    
    overall = sum(WEIGHTS[k] * v for k, v in scores.items())
    
    # Build per-second timeline from tension AU frame scores (most granular)
    timeline = []
    if "frame_scores" in m1:
        fs = pd.DataFrame(m1["frame_scores"])
        # Resample to 1-second bins
        fs["second"] = fs["timestamp"].apply(lambda t: int(t))
        per_sec = fs.groupby("second")["effortless"].mean().reset_index()
        for _, row in per_sec.iterrows():
            timeline.append({
                "second": int(row["second"]),
                "effortless": round(float(row["effortless"]), 3)
            })
    
    return {
        "overall_score": round(overall, 3),
        "label": label_score(overall),
        "method_scores": scores,
        "timeline": timeline
    }
```

---

### `pipeline/insights.py`

Generate human-readable insights from analysis results.

```python
def generate_insights(m1, m2, m3, m4, m5, ensemble, au_df) -> list:
    """
    Generate list of insight dicts with:
    - type: "good" | "watch" | "info"
    - text: natural language finding
    """
    insights = []
    
    # Micro-expression events
    events = m2.get("events", [])
    if len(events) == 0:
        insights.append({"type": "good", "text": "No micro-expression bursts detected. No suppressed effort found."})
    elif len(events) <= 2:
        insights.append({"type": "info", "text": f"{len(events)} micro-expression event(s) detected — within normal range."})
    else:
        timestamps_str = ", ".join(e["timestamp_str"] for e in events[:3])
        insights.append({"type": "watch", "text": f"{len(events)} micro-expression bursts detected at {timestamps_str}. Suppressed effort is present."})
    
    # Blink rate
    blink_rate = m5["blink_rate_per_min"]
    if 12 <= blink_rate <= 20:
        insights.append({"type": "good", "text": f"Blink rate of {blink_rate:.1f}/min is in the genuine relaxation range (12–20/min)."})
    elif blink_rate < 10:
        insights.append({"type": "watch", "text": f"Low blink rate ({blink_rate:.1f}/min) suggests active suppression or high cognitive load."})
    else:
        insights.append({"type": "info", "text": f"Blink rate {blink_rate:.1f}/min is slightly above the typical range."})
    
    # Asymmetry
    asym = m3["mean"]
    if 0.08 <= asym <= 0.30:
        insights.append({"type": "good", "text": f"Asymmetry index of {asym:.2f} falls in the genuine expression range (0.08–0.30)."})
    elif asym < 0.08:
        insights.append({"type": "watch", "text": f"Very low asymmetry ({asym:.2f}) may indicate a performed or held expression."})
    
    # Coherence
    lag_ms = abs(m4.get("peak_lag_ms", 0))
    if lag_ms <= 100:
        insights.append({"type": "good", "text": f"Upper and lower face sync within {lag_ms:.0f}ms — consistent with genuine expression."})
    elif lag_ms >= 200:
        insights.append({"type": "watch", "text": f"Upper/lower face lag of {lag_ms:.0f}ms suggests the smile may be running ahead of the eyes — a performed expression marker."})
    
    # Duchenne check from AU data
    au_cols = list(au_df.columns)
    if "AU06" in au_cols and "AU12" in au_cols:
        au6_mean = au_df["AU06"].mean()
        au12_mean = au_df["AU12"].mean()
        au6_variance = au_df["AU06"].std()
        if au6_mean > 1.5 and au6_variance > 0.5:
            pct = int(min(100, au6_mean / au12_mean * 100)) if au12_mean > 0 else 0
            insights.append({"type": "good", "text": f"AU6 (cheek raiser) active with high variance — Duchenne smile markers present, suggesting genuine positive affect."})
        elif au12_mean > 2.0 and au6_mean < 0.8:
            insights.append({"type": "watch", "text": "AU12 (lip pull) high but AU6 (cheek raiser) low — classic fake smile pattern. Positive affect may not be genuine."})
    
    return insights
```

---

### `models.py`

```python
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class MethodSummary(BaseModel):
    method: str
    tier: int
    weight: float
    score: float

class AnalysisResult(BaseModel):
    job_id: str
    duration_secs: float
    frames_analysed: int
    overall_score: float
    overall_label: str
    method_scores: Dict[str, Any]
    timeline: List[Dict]
    au_means: Dict[str, float]
    micro_expression_events: List[Dict]
    blink_rate_per_min: float
    asymmetry_mean: float
    insights: List[Dict]
```

---

## Frontend — React Dashboard

### `src/App.jsx`

```jsx
import { useState } from "react"
import Upload from "./components/Upload"
import Dashboard from "./components/Dashboard"

export default function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleUpload(file) {
    setLoading(true)
    setError(null)
    try {
      const form = new FormData()
      form.append("file", file)
      const res = await fetch("http://localhost:8000/api/analyse", {
        method: "POST",
        body: form
      })
      const { job_id } = await res.json()
      
      // Poll for result
      let data
      for (let i = 0; i < 60; i++) {
        await new Promise(r => setTimeout(r, 3000))
        const poll = await fetch(`http://localhost:8000/api/result/${job_id}`)
        data = await poll.json()
        if (data.status === "done" || data.status === "error") break
      }
      
      if (data.status === "done") setResult(data.result)
      else setError(data.message || "Analysis failed")
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-5xl mx-auto">
        <h1 className="text-2xl font-medium mb-6">Effortlessness Analyzer</h1>
        {!result && <Upload onUpload={handleUpload} loading={loading} error={error} />}
        {result && <Dashboard result={result} onReset={() => setResult(null)} />}
      </div>
    </div>
  )
}
```

### `src/components/Dashboard.jsx`

Build a Dashboard component that displays:

1. **Header row**: video filename, duration, baseline status badge
2. **Overall score card**: large number (0–1), label (Effortless / Moderate effort / etc.), and a horizontal row of 5 method scores with weights and tier badges
3. **Timeline chart** (Recharts LineChart): x-axis = seconds, y-axis = 0–1, one line for overall effortlessness, micro-expression events as red dot markers
4. **AU bar chart**: horizontal bars for AU04, AU07, AU43, AU10, AU06, AU12 — red bars for tension AUs, green for positive affect AUs
5. **Insights list**: each insight has a colored badge (Good/Watch/Info) and a one-sentence text
6. **Stats grid**: 4 metric cards — Frames Analysed, Micro-expr Events, Blink Rate/min, Asymmetry Mean
7. **Method detail cards**: one card per method showing score, tier, weight, and a 1-sentence interpretation

### `src/components/Timeline.jsx`

Use Recharts `ComposedChart` with:
- `Line` for overall effortlessness score per second
- `Line` (dashed) for tension_aus score per second
- `ReferenceDot` for each micro-expression event timestamp
- X-axis formatted as `mm:ss`
- Tooltip showing timestamp + score on hover

---

## Setup Instructions to Include in README

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

---

## Important Implementation Notes

1. **py-feat is slow on CPU** — for videos longer than 2 minutes, process frames in parallel using `ProcessPoolExecutor`, or use `feat`'s built-in batch processing. Consider running at 12 FPS instead of 24 to halve processing time.

2. **py-feat AU column names** — verify exact column names after calling `detector.detect_image()`. They may be "AU01" or "AU1" depending on version. Print `result.columns` on the first run to confirm.

3. **Baseline calibration window** — if the video is shorter than 30 seconds, use the first 25% of video as baseline.

4. **Head pose normalization** — before AU scoring, filter out frames where `abs(yaw) > 30 degrees` as AU detection is unreliable at extreme angles. Mark those frames as `null` in the timeline.

5. **Error handling** — wrap each method call in try/except and return a fallback score of 0.5 with a warning flag. Never let a single method failure crash the whole pipeline.

6. **File cleanup** — delete uploaded video files from `/tmp/uploads` after processing completes.

7. **Memory** — for long videos, do not keep all frames in memory at once. Process in chunks of 200 frames, save AU results to a temp CSV, then load the CSV for analysis.

---

## Optional Enhancements (Phase 2)

- **Audio coherence**: use `librosa` to extract pitch/energy time series, cross-correlate with overall effortlessness timeline
- **OpenFace upgrade**: replace py-feat with OpenFace 2.0 (Docker image available) for more accurate AU detection, especially for AU bilateral variants
- **Video scrubber**: add a video player in the frontend that syncs with the timeline — clicking a point on the chart seeks the video to that timestamp
- **Export**: PDF report generation using `reportlab` or `weasyprint`
- **Batch mode**: accept a folder of videos and produce a comparative leaderboard
```
