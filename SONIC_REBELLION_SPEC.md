# Sonic Rebellion — Audio Analysis Pipeline
## Implementation Spec for Claude Code

> **Goal**: Score an Instagram creator's "sonic rebellion" across 12 Reels in under 30 seconds.
> Three sub-scores — sound novelty, vocal edge, genre edge — combine into a single composite score.

---

## Project Structure

```
sonic_rebellion/
├── main.py                  # entry point — run scorer for one creator
├── pipeline/
│   ├── __init__.py
│   ├── fetcher.py           # download reels from Instagram URL / scraper
│   ├── extractor.py         # ffmpeg audio extraction + smart sampling
│   ├── separator.py         # Spleeter source separation (vocals / accompaniment)
│   ├── novelty.py           # fingerprint check + CLAP embedding fallback
│   ├── vocal_edge.py        # pitch, pace, tone consistency scoring
│   ├── genre_edge.py        # genre classification + BPM + coolness tier
│   └── scorer.py            # weighted composite + percentile ranking
├── data/
│   ├── trending_pool/       # pre-built FAISS index + chromaprint hashes
│   │   ├── index.faiss
│   │   └── hashes.pkl
│   └── coolness_tiers.json  # genre → prestige score map
├── scripts/
│   ├── build_trending_pool.py   # offline: scrape + embed + index trending audio
│   └── update_coolness_tiers.py # offline: update genre prestige scores
├── requirements.txt
└── config.py
```

---

## Dependencies

### `requirements.txt`

```
# Audio I/O
ffmpeg-python==0.2.0
soundfile==0.12.1
librosa==0.10.1

# Source separation
spleeter==2.3.2

# Fingerprinting
pyacoustid==1.3.0
acoustid==1.1.0

# Embeddings + search
transformers==4.40.0
torch>=2.1.0
faiss-cpu==1.8.0           # swap to faiss-gpu if GPU available

# CLAP (audio-text embedding)
msclap==1.3.3              # Microsoft CLAP — pip install msclap

# Pitch tracking
crepe==0.0.14              # CNN pitch estimator
resampy==0.4.2             # required by crepe

# Speech recognition
openai-whisper==20231117   # use 'tiny' or 'base' model
torch-audiomentations==0.11.0

# Genre + beat
essentia==2.1b6.dev1110    # MTG Essentia with MusiCNN
madmom==0.16.1             # RNN beat tracker

# Utilities
numpy>=1.24
scipy>=1.11
requests==2.31.0
instaloader==4.10.1        # Instagram scraper
tqdm==4.66.1
joblib==1.3.2
```

### System dependencies (install before pip)

```bash
# Ubuntu / Debian
sudo apt-get install -y ffmpeg libchromaprint-dev libsndfile1

# macOS
brew install ffmpeg chromaprint libsndfile
```

---

## `config.py`

```python
import os

# --- Model selection ---
WHISPER_MODEL = "tiny"          # tiny=fastest, base=better multilingual
SPLEETER_MODEL = "spleeter:2stems"  # vocals + accompaniment only

# --- Timing budget (seconds) ---
CLIP_DURATION_S = 10            # extract this many seconds per reel
MAX_REELS = 12

# --- Audio ---
SAMPLE_RATE = 22050             # Hz — Spleeter and CREPE both work at this rate
TARGET_LOUDNESS_DBFS = -23.0    # normalise before processing

# --- Fingerprinting ---
FINGERPRINT_THRESHOLD = 0.80    # chromaprint similarity above this = trending match

# --- Embedding ---
CLAP_MODEL = "630k-audioset-best"  # best general-purpose CLAP checkpoint
EMBED_DIM = 512

# --- Scoring weights ---
WEIGHT_NOVELTY = 0.35
WEIGHT_VOCAL   = 0.35
WEIGHT_GENRE   = 0.30

# --- Coolness tier path ---
COOLNESS_TIERS_PATH = "data/coolness_tiers.json"
TRENDING_INDEX_PATH = "data/trending_pool/index.faiss"
TRENDING_HASHES_PATH = "data/trending_pool/hashes.pkl"

# --- Norms for vocal edge (mainstream influencer baseline) ---
NORM_F0_FEMALE_HZ   = 200.0    # average F0 for female mainstream creators
NORM_F0_MALE_HZ     = 125.0    # average F0 for male mainstream creators
NORM_F0_VARIANCE    = 800.0    # variance of F0 in mainstream delivery (Hz²)
NORM_WPM            = 155.0    # words per minute
NORM_ENERGY_SIGMA   = 0.04     # std dev of RMS energy in mainstream delivery

# --- Genre edge ---
EDGE_BPM_LOW  = 70             # below this = unusually slow
EDGE_BPM_HIGH = 180            # above this = unusually fast
```

---

## Module 1 — `pipeline/fetcher.py`

Downloads Reels from a creator's Instagram profile. Accepts either a profile URL or a list of Reel URLs.

```python
"""
fetcher.py
Uses instaloader to scrape up to MAX_REELS recent Reels from a profile.
Returns list of local MP4 file paths.
"""
import instaloader
import os
import tempfile
from config import MAX_REELS

def fetch_reels(profile_url: str, output_dir: str = None) -> list[str]:
    """
    Args:
        profile_url: Instagram profile URL e.g. 'https://www.instagram.com/username/'
        output_dir: where to save MP4s. Uses tempdir if None.
    Returns:
        List of paths to downloaded MP4 files (up to MAX_REELS).
    """
    username = profile_url.rstrip("/").split("/")[-1].lstrip("@")
    output_dir = output_dir or tempfile.mkdtemp(prefix="sr_reels_")
    os.makedirs(output_dir, exist_ok=True)

    L = instaloader.Instaloader(
        download_video_thumbnails=False,
        download_comments=False,
        save_metadata=False,
        post_metadata_txt_pattern="",
        filename_pattern="{profile}_{shortcode}",
        dirname_pattern=output_dir,
        quiet=True,
    )

    profile = instaloader.Profile.from_username(L.context, username)
    paths = []

    for post in profile.get_posts():
        if post.is_video and post.product_type in ("clips", "reels"):
            L.download_post(post, target=output_dir)
            # find the downloaded mp4
            for f in os.listdir(output_dir):
                if f.endswith(".mp4") and post.shortcode in f:
                    paths.append(os.path.join(output_dir, f))
                    break
        if len(paths) >= MAX_REELS:
            break

    return paths


def fetch_from_urls(reel_urls: list[str], output_dir: str = None) -> list[str]:
    """Alternative: pass explicit list of Reel URLs."""
    output_dir = output_dir or tempfile.mkdtemp(prefix="sr_reels_")
    L = instaloader.Instaloader(
        download_video_thumbnails=False,
        download_comments=False,
        save_metadata=False,
        post_metadata_txt_pattern="",
        dirname_pattern=output_dir,
        quiet=True,
    )
    paths = []
    for url in reel_urls[:MAX_REELS]:
        shortcode = url.rstrip("/").split("/")[-1]
        post = instaloader.Post.from_shortcode(L.context, shortcode)
        L.download_post(post, target=output_dir)
        for f in os.listdir(output_dir):
            if f.endswith(".mp4") and shortcode in f:
                paths.append(os.path.join(output_dir, f))
                break
    return paths
```

---

## Module 2 — `pipeline/extractor.py`

Extracts audio from MP4s using ffmpeg, then selects the peak-energy 10-second window.

```python
"""
extractor.py
1. Decode AAC audio from MP4 → WAV at target sample rate
2. Find the peak RMS energy window of CLIP_DURATION_S seconds
3. Return the clipped waveform as numpy array

Why peak energy window: avoids silent intros/outros and lands on the most
musically dense region — which is where style choices are most visible.
"""
import numpy as np
import subprocess
import tempfile
import os
import librosa
from config import SAMPLE_RATE, CLIP_DURATION_S


def extract_audio(mp4_path: str) -> tuple[np.ndarray, int]:
    """
    Returns (waveform, sample_rate) where waveform is the peak-energy
    CLIP_DURATION_S window, mono, at SAMPLE_RATE.
    """
    wav_path = _decode_to_wav(mp4_path)
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    os.unlink(wav_path)
    y = _peak_energy_window(y, sr)
    return y, sr


def _decode_to_wav(mp4_path: str) -> str:
    """Use ffmpeg to decode AAC → PCM WAV. Returns temp file path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-y", "-i", mp4_path,
        "-vn",                          # no video
        "-ar", str(SAMPLE_RATE),        # resample
        "-ac", "1",                     # mono
        "-sample_fmt", "s16",           # 16-bit PCM
        tmp.name,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return tmp.name


def _peak_energy_window(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Slide a window of CLIP_DURATION_S seconds across the signal.
    Return the window with highest mean RMS energy.
    Avoids pure-silence regions and captures the densest audio segment.
    """
    window_samples = int(CLIP_DURATION_S * sr)
    if len(y) <= window_samples:
        # clip shorter than window — pad with zeros
        return np.pad(y, (0, window_samples - len(y)))

    # compute RMS every 0.5s
    hop = sr // 2
    frame_rms = librosa.feature.rms(y=y, frame_length=window_samples, hop_length=hop)[0]
    best_frame = int(np.argmax(frame_rms))
    start = best_frame * hop
    end = start + window_samples
    if end > len(y):
        start = len(y) - window_samples
        end = len(y)
    return y[start:end]


def batch_extract(mp4_paths: list[str]) -> list[tuple[np.ndarray, int]]:
    """Extract audio from all paths. Returns list of (waveform, sr) tuples."""
    results = []
    for path in mp4_paths:
        try:
            results.append(extract_audio(path))
        except Exception as e:
            print(f"[extractor] failed on {path}: {e}")
            results.append((np.zeros(int(CLIP_DURATION_S * SAMPLE_RATE)), SAMPLE_RATE))
    return results
```

---

## Module 3 — `pipeline/separator.py`

Runs Spleeter 2-stem separation in a single batched GPU call.

```python
"""
separator.py
Wraps Spleeter 2stems (vocals + accompaniment).
Processes all clips in one batched call for GPU efficiency.

Why Spleeter over HTDemucs:
- ~5x faster on same hardware
- ~1 dB SDR loss on MUSDB18 — acceptable for 10s clip scoring
- Supports batch inference natively

Output: for each clip, returns (vocals_array, accompaniment_array)
"""
import numpy as np
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter
from config import SAMPLE_RATE, SPLEETER_MODEL

_separator = None  # loaded once, kept warm

def _get_separator() -> Separator:
    global _separator
    if _separator is None:
        _separator = Separator(SPLEETER_MODEL)
    return _separator


def separate_batch(
    waveforms: list[np.ndarray],
    sr: int = SAMPLE_RATE,
) -> list[dict[str, np.ndarray]]:
    """
    Args:
        waveforms: list of mono numpy arrays (all same length)
    Returns:
        list of dicts with keys 'vocals' and 'accompaniment',
        each a numpy array of same length as input.
    """
    sep = _get_separator()
    results = []

    for y in waveforms:
        # Spleeter expects (samples, channels) stereo
        stereo = np.stack([y, y], axis=-1)
        prediction = sep.separate(stereo)
        results.append({
            "vocals": prediction["vocals"][:, 0],           # take left channel → mono
            "accompaniment": prediction["accompaniment"][:, 0],
        })

    return results
```

---

## Module 4 — `pipeline/novelty.py`

Fingerprint check against trending pool → CLAP embedding fallback on misses.

```python
"""
novelty.py
Two-stage cascade:
  1. Chromaprint fingerprint vs. pre-built trending hash set  →  O(1) lookup
  2. CLAP embedding vs. FAISS index  →  cosine distance (only on fingerprint miss)

novelty_score = 1 - max_similarity_to_trending_pool ∈ [0, 1]
  0 = identical to trending content
  1 = completely unlike anything trending
"""
import pickle
import numpy as np
import acoustid
import faiss
from msclap import CLAP
from config import (
    FINGERPRINT_THRESHOLD, CLAP_MODEL, EMBED_DIM,
    TRENDING_INDEX_PATH, TRENDING_HASHES_PATH, SAMPLE_RATE,
)

_clap_model = None
_faiss_index = None
_trending_hashes = None


def _get_clap() -> CLAP:
    global _clap_model
    if _clap_model is None:
        _clap_model = CLAP(version=CLAP_MODEL, use_cuda=False)
    return _clap_model


def _get_faiss_index():
    global _faiss_index
    if _faiss_index is None:
        _faiss_index = faiss.read_index(TRENDING_INDEX_PATH)
    return _faiss_index


def _get_trending_hashes() -> set:
    global _trending_hashes
    if _trending_hashes is None:
        with open(TRENDING_HASHES_PATH, "rb") as f:
            _trending_hashes = pickle.load(f)
    return _trending_hashes


def _fingerprint(y: np.ndarray, sr: int) -> str | None:
    """Compute Chromaprint fingerprint. Returns fingerprint string or None."""
    import soundfile as sf
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, y, sr, subtype="PCM_16")
    try:
        duration, fp = acoustid.fingerprint_file(tmp.name)
        return fp
    except Exception:
        return None
    finally:
        os.unlink(tmp.name)


def _clap_embed(y: np.ndarray, sr: int) -> np.ndarray:
    """Embed audio clip using CLAP. Returns L2-normalised 512-dim vector."""
    clap = _get_clap()
    # msclap accepts audio arrays directly
    embed = clap.get_audio_embeddings_from_array([y], resample=True)
    vec = embed[0].cpu().numpy().astype("float32")
    # L2 normalise
    vec /= np.linalg.norm(vec) + 1e-9
    return vec


def score_novelty_batch(
    waveforms: list[np.ndarray],
    sr: int = SAMPLE_RATE,
) -> list[float]:
    """
    Returns novelty score ∈ [0, 1] for each clip.
    0 = trending, 1 = completely novel.
    """
    hashes = _get_trending_hashes()
    index = _get_faiss_index()
    scores = []

    embed_needed = []   # indices that need CLAP embedding
    results = [None] * len(waveforms)

    # Stage 1: fingerprint gate
    for i, y in enumerate(waveforms):
        fp = _fingerprint(y, sr)
        if fp and fp in hashes:
            results[i] = 0.0           # it's trending — novelty is zero
        else:
            embed_needed.append(i)

    # Stage 2: CLAP embedding for misses only
    if embed_needed:
        vecs = np.array([
            _clap_embed(waveforms[i], sr) for i in embed_needed
        ], dtype="float32")

        # batch ANN search: k=1, find closest trending embedding
        D, _ = index.search(vecs, k=1)
        for j, i in enumerate(embed_needed):
            max_sim = float(D[j, 0])   # FAISS inner product = cosine sim for L2-normed vecs
            results[i] = 1.0 - max_sim

    return results
```

---

## Module 5 — `pipeline/vocal_edge.py`

Scores vocal delivery deviation from the mainstream influencer norm.

```python
"""
vocal_edge.py
Four features measured on the vocal stem from Spleeter:
  1. F0 mean — deviation from gender-norm pitch
  2. F0 variance — flatness (deadpan) or wildness
  3. Speaking rate (WPM) — via Whisper tiny
  4. Energy sigma — tonal consistency proxy

vocal_edge_score ∈ [0, 1]
  0 = sounds exactly like a mainstream influencer
  1 = maximally deviant delivery
"""
import numpy as np
import whisper
import crepe
import soundfile as sf
import tempfile, os
from config import (
    SAMPLE_RATE, WHISPER_MODEL,
    NORM_F0_FEMALE_HZ, NORM_F0_MALE_HZ, NORM_F0_VARIANCE,
    NORM_WPM, NORM_ENERGY_SIGMA,
)

_whisper_model = None

def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(WHISPER_MODEL)
    return _whisper_model


def _tmp_wav(y: np.ndarray, sr: int) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, y, sr, subtype="PCM_16")
    return tmp.name


def _extract_f0(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Run CREPE pitch estimator on vocal waveform.
    Returns array of F0 values in Hz (only confident frames, confidence > 0.5).
    CREPE is robust to compressed audio — better than YIN/pYIN for Instagram AAC.
    """
    # CREPE expects float32 waveform at given sr
    time, frequency, confidence, _ = crepe.predict(y, sr, viterbi=True, verbose=0)
    # filter low-confidence frames (background music bleed-through)
    confident = frequency[confidence > 0.5]
    return confident


def _speaking_rate_wpm(y: np.ndarray, sr: int) -> float | None:
    """
    Transcribe with Whisper tiny and compute WPM from word count / speech duration.
    Returns None if no speech detected.
    """
    model = _get_whisper()
    tmp = _tmp_wav(y, sr)
    try:
        result = model.transcribe(tmp, word_timestamps=True, verbose=False)
        words = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                words.append(w)
        if not words:
            return None
        n_words = len(words)
        duration_s = words[-1]["end"] - words[0]["start"]
        if duration_s < 1.0:
            return None
        return (n_words / duration_s) * 60.0
    finally:
        os.unlink(tmp)


def _energy_sigma(y: np.ndarray, sr: int) -> float:
    """
    Standard deviation of short-term RMS energy over 0.5s frames.
    High sigma = highly variable energy (enthusiastic, dynamic).
    Low sigma = flat, consistent energy (deadpan, whispery).
    """
    import librosa
    rms = librosa.feature.rms(y=y, frame_length=int(sr * 0.5), hop_length=int(sr * 0.25))[0]
    return float(np.std(rms))


def _f0_slope_at_boundaries(f0_frames: np.ndarray) -> float:
    """
    Crude inflection score: measure whether F0 tends to rise at the end
    of detected utterances. Compare last-third vs first-third mean F0.
    Positive = rising (valley-girl), negative = falling (authoritative).
    """
    if len(f0_frames) < 6:
        return 0.0
    n = len(f0_frames)
    first_third = np.mean(f0_frames[:n // 3])
    last_third  = np.mean(f0_frames[2 * n // 3:])
    return float(last_third - first_third)


def _normalise_feature(value: float, norm: float, scale: float) -> float:
    """Return |deviation from norm| / scale, clipped to [0, 1]."""
    return float(np.clip(abs(value - norm) / scale, 0.0, 1.0))


def score_vocal_edge(
    vocal_stem: np.ndarray,
    sr: int = SAMPLE_RATE,
    gender_hint: str = "unknown",  # "male", "female", or "unknown"
) -> dict:
    """
    Returns dict with 'score' ∈ [0,1] and individual feature values.
    """
    f0_frames = _extract_f0(vocal_stem, sr)
    wpm       = _speaking_rate_wpm(vocal_stem, sr)
    e_sigma   = _energy_sigma(vocal_stem, sr)

    # --- F0 mean deviation ---
    if len(f0_frames) > 0:
        f0_mean = float(np.mean(f0_frames))
        f0_var  = float(np.var(f0_frames))
        # guess gender from F0 if not provided
        if gender_hint == "unknown":
            norm_f0 = NORM_F0_FEMALE_HZ if f0_mean > 155 else NORM_F0_MALE_HZ
        else:
            norm_f0 = NORM_F0_FEMALE_HZ if gender_hint == "female" else NORM_F0_MALE_HZ

        feat_f0_mean = _normalise_feature(f0_mean, norm_f0, norm_f0 * 0.5)
        feat_f0_var  = _normalise_feature(f0_var, NORM_F0_VARIANCE, NORM_F0_VARIANCE * 2)
    else:
        # no detected speech — silence or music-only
        f0_mean, f0_var = 0.0, 0.0
        feat_f0_mean, feat_f0_var = 0.0, 0.0

    # --- Speaking rate ---
    feat_wpm = _normalise_feature(wpm, NORM_WPM, 80.0) if wpm else 0.0

    # --- Tone consistency ---
    feat_energy = _normalise_feature(e_sigma, NORM_ENERGY_SIGMA, NORM_ENERGY_SIGMA * 3)

    # Weighted sum — F0 features weighted higher (more intentional signal)
    score = (
        0.30 * feat_f0_mean +
        0.30 * feat_f0_var  +
        0.20 * feat_wpm     +
        0.20 * feat_energy
    )

    return {
        "score": float(np.clip(score, 0.0, 1.0)),
        "f0_mean_hz": f0_mean,
        "f0_variance": f0_var,
        "wpm": wpm,
        "energy_sigma": e_sigma,
    }


def score_vocal_edge_batch(
    vocal_stems: list[np.ndarray],
    sr: int = SAMPLE_RATE,
) -> list[dict]:
    """Score vocal edge for all clips. Whisper and CREPE are called per-clip."""
    return [score_vocal_edge(v, sr) for v in vocal_stems]
```

---

## Module 6 — `pipeline/genre_edge.py`

Genre classification via Essentia MusiCNN + BPM via madmom + coolness tier lookup.

```python
"""
genre_edge.py
Runs on the accompaniment stem from Spleeter (clean of vocal content).

Pipeline:
  1. Essentia MusiCNN → genre label from Discogs 400-class taxonomy
  2. madmom RNN beat tracker → BPM (or no-pulse detection)
  3. Coolness tier lookup → genre prestige score
  4. BPM edge score — penalise very mainstream tempos (100–130 BPM)

genre_edge_score ∈ [0, 1]
  0 = mainstream genre + mainstream tempo
  1 = niche genre + unusual tempo
"""
import json
import numpy as np
import soundfile as sf
import tempfile, os
import essentia.standard as es
import madmom
from config import (
    SAMPLE_RATE, COOLNESS_TIERS_PATH,
    EDGE_BPM_LOW, EDGE_BPM_HIGH,
)

_coolness_tiers = None

def _get_coolness_tiers() -> dict:
    global _coolness_tiers
    if _coolness_tiers is None:
        with open(COOLNESS_TIERS_PATH) as f:
            _coolness_tiers = json.load(f)
    return _coolness_tiers


def _tmp_wav(y: np.ndarray, sr: int) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, y, sr, subtype="PCM_16")
    return tmp.name


def _classify_genre(y: np.ndarray, sr: int) -> tuple[str, float]:
    """
    Use Essentia MusiCNN with Discogs-Effnet embeddings.
    Returns (top_genre_label, confidence).

    Note: Essentia's TensorflowPredictEffnetDiscogs produces embeddings,
    then TensorflowPredict2D classifies over 400 genre classes.
    Download models from: https://essentia.upf.edu/models/
      - discogs-effnet-bs64-1.pb
      - genre_discogs400-discogs-effnet-1.pb
    Place in data/models/
    """
    audio = es.MonoLoader(filename=_tmp_wav(y, sr), sampleRate=sr)()

    embedding_model = es.TensorflowPredictEffnetDiscogs(
        graphFilename="data/models/discogs-effnet-bs64-1.pb",
        output="PartitionedCall:1"
    )
    classifier_model = es.TensorflowPredict2D(
        graphFilename="data/models/genre_discogs400-discogs-effnet-1.pb",
        input="serving_default_model_Placeholder",
        output="PartitionedCall:0"
    )

    embeddings = embedding_model(audio)
    predictions = classifier_model(embeddings)
    mean_preds = np.mean(predictions, axis=0)

    # Load class labels (400 Discogs genres)
    with open("data/models/genre_discogs400_labels.json") as f:
        labels = json.load(f)

    top_idx = int(np.argmax(mean_preds))
    return labels[top_idx], float(mean_preds[top_idx])


def _estimate_bpm(y: np.ndarray, sr: int) -> float | None:
    """
    Use madmom RNN beat tracker. Returns BPM float or None if no stable pulse.
    madmom handles non-4/4 time, very slow tempos, and unconventional rhythms
    better than librosa's autocorrelation-based tracker.
    """
    tmp = _tmp_wav(y, sr)
    try:
        proc = madmom.features.beats.RNNBeatProcessor()
        act = proc(tmp)
        bpm_proc = madmom.features.beats.BeatTrackingProcessor(fps=100)
        beats = bpm_proc(act)

        if len(beats) < 2:
            return None  # no detectable pulse — itself an edge signal

        # estimate BPM from median beat interval
        intervals = np.diff(beats)
        median_interval = np.median(intervals)
        if median_interval < 0.01:
            return None
        return round(60.0 / median_interval, 1)
    except Exception:
        return None
    finally:
        os.unlink(tmp)


def _bpm_edge_score(bpm: float | None) -> float:
    """
    Score how far BPM is from mainstream range (100–130 BPM).
    No pulse → 1.0 (maximally edge).
    Very slow (<70) or very fast (>180) → high score.
    Mainstream → 0.0.
    """
    if bpm is None:
        return 1.0  # no detectable beat is an edge choice
    mainstream_low, mainstream_high = 90.0, 135.0
    if mainstream_low <= bpm <= mainstream_high:
        dist = 0.0
    elif bpm < EDGE_BPM_LOW:
        dist = (EDGE_BPM_LOW - bpm) / EDGE_BPM_LOW
    elif bpm > EDGE_BPM_HIGH:
        dist = (bpm - EDGE_BPM_HIGH) / EDGE_BPM_HIGH
    else:
        # between mainstream and edge thresholds
        if bpm < mainstream_low:
            dist = (mainstream_low - bpm) / (mainstream_low - EDGE_BPM_LOW)
        else:
            dist = (bpm - mainstream_high) / (EDGE_BPM_HIGH - mainstream_high)
    return float(np.clip(dist, 0.0, 1.0))


def score_genre_edge(
    accompaniment_stem: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> dict:
    """
    Returns dict with 'score' ∈ [0,1] and intermediate values.
    """
    tiers = _get_coolness_tiers()

    genre_label, genre_conf = _classify_genre(accompaniment_stem, sr)
    bpm = _estimate_bpm(accompaniment_stem, sr)

    # Look up genre prestige (0.0 = mainstream, 1.0 = niche/edge)
    # normalise label: take top-level genre before " --- "
    top_genre = genre_label.split(" --- ")[0].strip().lower()
    genre_score = tiers.get(top_genre, 0.5)  # default 0.5 if unknown

    bpm_score = _bpm_edge_score(bpm)

    # Weighted: genre matters more than tempo
    score = 0.65 * genre_score + 0.35 * bpm_score

    return {
        "score": float(np.clip(score, 0.0, 1.0)),
        "genre_label": genre_label,
        "genre_confidence": genre_conf,
        "genre_prestige": genre_score,
        "bpm": bpm,
        "bpm_edge_score": bpm_score,
    }


def score_genre_edge_batch(
    accompaniment_stems: list[np.ndarray],
    sr: int = SAMPLE_RATE,
) -> list[dict]:
    return [score_genre_edge(a, sr) for a in accompaniment_stems]
```

---

## Module 7 — `pipeline/scorer.py`

Aggregates per-reel scores into a single creator-level composite.

```python
"""
scorer.py
Aggregates 12 per-reel sub-scores into a single creator sonic rebellion score.

Per reel:
  sonic_rebellion_reel = w_nov * novelty + w_voc * vocal_edge + w_gen * genre_edge

Creator score:
  - Mean across 12 reels (robust to single outlier)
  - Optionally ranked as percentile vs. pool of scored creators
"""
import numpy as np
from config import WEIGHT_NOVELTY, WEIGHT_VOCAL, WEIGHT_GENRE


def aggregate_reel_scores(
    novelty_scores:    list[float],
    vocal_edge_scores: list[dict],
    genre_edge_scores: list[dict],
) -> dict:
    """
    Args:
        novelty_scores: list of floats ∈ [0,1], one per reel
        vocal_edge_scores: list of dicts from vocal_edge.score_vocal_edge_batch
        genre_edge_scores: list of dicts from genre_edge.score_genre_edge_batch
    Returns:
        Dict with per-reel breakdown and creator-level composite score.
    """
    n = len(novelty_scores)
    per_reel = []

    for i in range(n):
        nov  = novelty_scores[i]
        voc  = vocal_edge_scores[i]["score"]
        gen  = genre_edge_scores[i]["score"]
        reel_score = WEIGHT_NOVELTY * nov + WEIGHT_VOCAL * voc + WEIGHT_GENRE * gen
        per_reel.append({
            "reel_index":    i,
            "novelty":       round(nov, 3),
            "vocal_edge":    round(voc, 3),
            "genre_edge":    round(gen, 3),
            "reel_score":    round(reel_score, 3),
            "vocal_details": vocal_edge_scores[i],
            "genre_details": genre_edge_scores[i],
        })

    reel_scores = [r["reel_score"] for r in per_reel]
    creator_score = float(np.mean(reel_scores))

    return {
        "creator_score":        round(creator_score, 3),
        "score_label":          _label(creator_score),
        "sub_score_means": {
            "novelty":    round(float(np.mean(novelty_scores)), 3),
            "vocal_edge": round(float(np.mean([v["score"] for v in vocal_edge_scores])), 3),
            "genre_edge": round(float(np.mean([g["score"] for g in genre_edge_scores])), 3),
        },
        "per_reel": per_reel,
        "n_reels_scored": n,
    }


def _label(score: float) -> str:
    if score >= 0.75: return "Sonic Rebel"
    if score >= 0.55: return "Edge-leaning"
    if score >= 0.35: return "Balanced"
    return "Mainstream"
```

---

## Module 8 — `main.py`

Orchestrates the full pipeline with parallel execution.

```python
"""
main.py
Entry point for scoring one creator.

Usage:
    python main.py --profile https://www.instagram.com/username/
    python main.py --urls url1 url2 url3 ...  (up to 12)

Architecture notes:
  - Audio extraction and separation run sequentially (CPU bound, fast)
  - Novelty, vocal, and genre scoring run concurrently via ThreadPoolExecutor
    (all three operate on different stems, no shared state)
  - Models are loaded once at module import and kept warm
"""
import argparse
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from pipeline.fetcher    import fetch_reels, fetch_from_urls
from pipeline.extractor  import batch_extract
from pipeline.separator  import separate_batch
from pipeline.novelty    import score_novelty_batch
from pipeline.vocal_edge import score_vocal_edge_batch
from pipeline.genre_edge import score_genre_edge_batch
from pipeline.scorer     import aggregate_reel_scores
from config              import SAMPLE_RATE


def run_pipeline(mp4_paths: list[str]) -> dict:
    t0 = time.time()

    # Phase 1: extract 10s peak-energy clips from all reels
    print(f"[1/5] Extracting audio from {len(mp4_paths)} reels...")
    clips = batch_extract(mp4_paths)          # list of (waveform, sr)
    waveforms = [c[0] for c in clips]
    t1 = time.time()
    print(f"      done in {t1-t0:.1f}s")

    # Phase 2: source separation — single batched call
    print("[2/5] Separating vocals and accompaniment (Spleeter)...")
    stems = separate_batch(waveforms, sr=SAMPLE_RATE)
    vocals   = [s["vocals"]        for s in stems]
    accomp   = [s["accompaniment"] for s in stems]
    t2 = time.time()
    print(f"      done in {t2-t1:.1f}s")

    # Phase 3: run three scoring branches concurrently
    print("[3/5] Scoring novelty, vocal edge, genre edge in parallel...")
    with ThreadPoolExecutor(max_workers=3) as executor:
        fut_nov  = executor.submit(score_novelty_batch,    waveforms, SAMPLE_RATE)
        fut_voc  = executor.submit(score_vocal_edge_batch, vocals,    SAMPLE_RATE)
        fut_gen  = executor.submit(score_genre_edge_batch, accomp,    SAMPLE_RATE)

        novelty_scores    = fut_nov.result()
        vocal_edge_scores = fut_voc.result()
        genre_edge_scores = fut_gen.result()
    t3 = time.time()
    print(f"      done in {t3-t2:.1f}s")

    # Phase 4: aggregate
    print("[4/5] Aggregating scores...")
    result = aggregate_reel_scores(novelty_scores, vocal_edge_scores, genre_edge_scores)
    t4 = time.time()

    result["wall_clock_s"] = round(t4 - t0, 1)
    print(f"[5/5] Complete in {t4-t0:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser(description="Sonic Rebellion scorer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--profile", type=str, help="Instagram profile URL")
    group.add_argument("--urls",    nargs="+", help="List of Reel URLs (max 12)")
    parser.add_argument("--outfile", type=str, default=None,
                        help="Save JSON result to file")
    args = parser.parse_args()

    print("Fetching reels...")
    if args.profile:
        mp4_paths = fetch_reels(args.profile)
    else:
        mp4_paths = fetch_from_urls(args.urls)

    if not mp4_paths:
        print("No reels found. Exiting.")
        return

    result = run_pipeline(mp4_paths)

    print("\n" + "="*50)
    print(f"SONIC REBELLION SCORE: {result['creator_score']} — {result['score_label']}")
    print(f"  Novelty:    {result['sub_score_means']['novelty']}")
    print(f"  Vocal edge: {result['sub_score_means']['vocal_edge']}")
    print(f"  Genre edge: {result['sub_score_means']['genre_edge']}")
    print(f"  Wall clock: {result['wall_clock_s']}s")
    print("="*50)

    if args.outfile:
        with open(args.outfile, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to {args.outfile}")


if __name__ == "__main__":
    main()
```

---

## Offline Script 1 — `scripts/build_trending_pool.py`

Run nightly to update the FAISS index and fingerprint hash set.

```python
"""
build_trending_pool.py
Offline script — run on a schedule (nightly recommended).
Scrapes top trending Instagram sounds, embeds them with CLAP,
builds FAISS index and chromaprint hash set.

Usage:
    python scripts/build_trending_pool.py --top-n 500

Output:
    data/trending_pool/index.faiss
    data/trending_pool/hashes.pkl
"""
import argparse
import pickle
import os
import numpy as np
import faiss
import acoustid
from msclap import CLAP
import instaloader

# NOTE: Replace this stub with your actual trending-audio scrape logic.
# Instagram doesn't expose a public trending-sounds API.
# Options:
#   - Scrape the "Reels" tab of high-follower accounts systematically
#   - Use a third-party Instagram data API (Apify, PhantomBuster)
#   - Maintain a manual curated list updated weekly
def scrape_trending_audio_clips(top_n: int) -> list[tuple[str, str]]:
    """
    Returns list of (audio_file_path, sound_label) tuples.
    Replace stub with real scraping logic.
    """
    raise NotImplementedError("Implement trending audio scraping for your data source")


def build_index(top_n: int = 500):
    os.makedirs("data/trending_pool", exist_ok=True)
    clap = CLAP(version="630k-audioset-best", use_cuda=False)

    print(f"Scraping {top_n} trending audio clips...")
    clips = scrape_trending_audio_clips(top_n)

    hashes = set()
    embeddings = []

    for audio_path, label in clips:
        # fingerprint
        try:
            _, fp = acoustid.fingerprint_file(audio_path)
            hashes.add(fp)
        except Exception:
            pass

        # embed
        try:
            embed = clap.get_audio_embeddings([audio_path])
            vec = embed[0].cpu().numpy().astype("float32")
            vec /= np.linalg.norm(vec) + 1e-9
            embeddings.append(vec)
        except Exception as e:
            print(f"  embed failed for {label}: {e}")

    # save hashes
    with open("data/trending_pool/hashes.pkl", "wb") as f:
        pickle.dump(hashes, f)
    print(f"Saved {len(hashes)} fingerprint hashes")

    # build FAISS flat inner-product index (cosine via L2-normed vecs)
    if embeddings:
        matrix = np.stack(embeddings, axis=0)
        dim = matrix.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)
        faiss.write_index(index, "data/trending_pool/index.faiss")
        print(f"Built FAISS index with {len(embeddings)} vectors (dim={dim})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=500)
    args = parser.parse_args()
    build_index(args.top_n)
```

---

## Offline Script 2 — `data/coolness_tiers.json`

Static seed file. Update weekly via `scripts/update_coolness_tiers.py`.

```json
{
  "_meta": {
    "description": "Genre prestige scores. 0.0 = mainstream, 1.0 = maximally niche/edge. Update weekly.",
    "last_updated": "2025-01-01",
    "methodology": "Based on Discogs genre adoption rate, mainstream crossover, brand usage frequency"
  },
  "pop": 0.05,
  "hip hop": 0.15,
  "lo-fi": 0.20,
  "r&b": 0.20,
  "dance": 0.15,
  "electronic": 0.25,
  "indie": 0.40,
  "alternative rock": 0.40,
  "jazz": 0.45,
  "folk": 0.45,
  "soul": 0.40,
  "funk": 0.42,
  "classical": 0.50,
  "metal": 0.55,
  "punk": 0.60,
  "post-punk": 0.70,
  "goth rock": 0.72,
  "cold wave": 0.78,
  "industrial": 0.80,
  "ebm": 0.82,
  "power electronics": 0.92,
  "noise": 0.93,
  "hyperpop": 0.60,
  "digicore": 0.75,
  "bubblegum bass": 0.78,
  "ambient": 0.65,
  "drone": 0.80,
  "dark ambient": 0.85,
  "experimental": 0.88,
  "musique concrete": 0.95,
  "no wave": 0.92,
  "post-rock": 0.65,
  "shoegaze": 0.68,
  "vaporwave": 0.62,
  "synthwave": 0.45,
  "chillwave": 0.48,
  "reggae": 0.42,
  "afrobeats": 0.35,
  "salsa": 0.38,
  "cumbia": 0.55,
  "bossa nova": 0.60
}
```

---

## Setup & First Run

```bash
# 1. Install system dependencies
sudo apt-get install -y ffmpeg libchromaprint-dev libsndfile1

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install Python packages
pip install -r requirements.txt

# 4. Download Essentia models
mkdir -p data/models
# Download from https://essentia.upf.edu/models/
# Required files:
#   discogs-effnet-bs64-1.pb
#   genre_discogs400-discogs-effnet-1.pb
#   genre_discogs400_labels.json

# 5. Build initial trending pool (implement scraping stub first)
python scripts/build_trending_pool.py --top-n 500

# 6. Run scorer
python main.py --profile https://www.instagram.com/targetcreator/
# or
python main.py --urls https://www.instagram.com/reel/ABC/ https://www.instagram.com/reel/XYZ/

# 7. Save output
python main.py --profile https://www.instagram.com/username/ --outfile results/creator_score.json
```

---

## Performance Notes for Claude Code

- **Model warm-up**: All models load at first call and are cached as module-level globals. For repeated creator scoring, wrap `run_pipeline()` in a service loop rather than calling `main.py` per creator — this avoids reloading Spleeter, Whisper, and CLAP on every run.

- **GPU**: If a CUDA GPU is available, change `faiss-cpu` to `faiss-gpu` in requirements, and set `use_cuda=True` in the CLAP constructor. Spleeter auto-detects GPU via TensorFlow. This cuts the batch separation step from ~5s to ~1.5s.

- **Threading vs multiprocessing**: The `ThreadPoolExecutor` in `main.py` works for I/O-bound tasks. Because novelty, vocal, and genre scoring each use different models with no shared state, true parallelism is achievable — but Python's GIL limits CPU parallelism. For production, consider running three separate worker processes with a message queue (Celery, RQ) and merging results.

- **Spleeter temp files**: Spleeter writes intermediate files to disk. For high-throughput use, configure it with `in_memory=True` if your Spleeter version supports it, or pre-allocate a RAM disk at `/tmp/spleeter_cache`.

- **Instagram rate limiting**: `instaloader` respects Instagram's rate limits by default. For bulk scoring (>20 creators/hour), add `sleep_between_requests=True` and consider rotating user agents. Instagram will temporarily block IPs that scrape aggressively.

- **Whisper tiny vs base**: `tiny` is 39M params and transcribes ~10s audio in ~0.8s on CPU. `base` is 74M params and ~1.5s. Switch to `base` if your creator pool includes significant non-English content — WER on multilingual audio degrades more steeply on `tiny`.
```
