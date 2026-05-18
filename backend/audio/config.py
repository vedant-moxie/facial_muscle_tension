"""
Sonic Rebellion — audio analysis configuration.

Mirrors the spec in SONIC_REBELLION_SPEC.md and is intentionally kept
separate from the facial-tension pipeline's settings to make the two
pipelines independently tunable.

All env-var-overridable values use AUDIO_* prefixes so they don't collide
with face-pipeline env vars (TARGET_FPS, KEEP_UPLOADS, etc.).
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths ----------------------------------------------------------------
# __file__ is backend/audio/config.py → parents[1] == backend/.
BACKEND_DIR = Path(__file__).resolve().parents[1]
AUDIO_DATA_DIR = BACKEND_DIR / "data" / "audio"
AUDIO_DATA_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_UPLOAD_DIR = Path(os.getenv(
    "AUDIO_UPLOAD_DIR", "/tmp/sonic_rebellion_uploads"
))
AUDIO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

COOLNESS_TIERS_PATH  = AUDIO_DATA_DIR / "coolness_tiers.json"
TRENDING_INDEX_PATH  = AUDIO_DATA_DIR / "trending_pool" / "index.faiss"
TRENDING_HASHES_PATH = AUDIO_DATA_DIR / "trending_pool" / "hashes.pkl"

# --- Timing budget --------------------------------------------------------
CLIP_DURATION_S = int(os.getenv("AUDIO_CLIP_DURATION_S", "10"))
MAX_REELS       = int(os.getenv("AUDIO_MAX_REELS", "12"))

# --- Audio ----------------------------------------------------------------
SAMPLE_RATE          = 22050     # Hz — Spleeter / CREPE / librosa native
TARGET_LOUDNESS_DBFS = -23.0

# --- Models ---------------------------------------------------------------
WHISPER_MODEL    = os.getenv("AUDIO_WHISPER_MODEL", "tiny")
SPLEETER_MODEL   = "spleeter:2stems"
CLAP_MODEL       = "630k-audioset-best"
EMBED_DIM        = 512

# --- Fingerprinting -------------------------------------------------------
FINGERPRINT_THRESHOLD = 0.80      # chromaprint similarity above this = trending

# --- Scoring weights ------------------------------------------------------
WEIGHT_NOVELTY = 0.35
WEIGHT_VOCAL   = 0.35
WEIGHT_GENRE   = 0.30

# --- Vocal-edge norms (mainstream influencer baseline) -------------------
NORM_F0_FEMALE_HZ = 200.0
NORM_F0_MALE_HZ   = 125.0
NORM_F0_VARIANCE  = 800.0
NORM_WPM          = 155.0
NORM_ENERGY_SIGMA = 0.04

# --- Genre / tempo --------------------------------------------------------
EDGE_BPM_LOW  = 70.0
EDGE_BPM_HIGH = 180.0
MAINSTREAM_BPM_LOW  = 90.0
MAINSTREAM_BPM_HIGH = 135.0

# --- Job / concurrency ----------------------------------------------------
AUDIO_PIPELINE_TIMEOUT_SEC = int(os.getenv("AUDIO_PIPELINE_TIMEOUT_SEC", "1800"))
AUDIO_MAX_CONCURRENT_JOBS  = int(os.getenv("AUDIO_MAX_CONCURRENT_JOBS", "1"))
AUDIO_MAX_UPLOAD_MB        = int(os.getenv("AUDIO_MAX_UPLOAD_MB", "1500"))   # 12 reels × ~100 MB
KEEP_AUDIO_UPLOADS         = os.getenv("KEEP_AUDIO_UPLOADS", "1") == "1"
