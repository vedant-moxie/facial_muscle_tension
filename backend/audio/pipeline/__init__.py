"""Sonic Rebellion — audio analysis pipeline.

Sibling to `pipeline/` (which handles facial muscle tension). Same backend
process, same job-polling pattern, separate config and modules.

Public surface:

    from audio.pipeline.runner import run_pipeline
    from audio.pipeline.fetcher import fetch_reels, fetch_from_urls

Each sub-module follows the same pattern: a primary entry point that
accepts a numpy waveform (or list thereof) and returns a per-reel dict.
Heavy ML deps (Spleeter, Whisper, CREPE, msclap, Essentia, madmom) are
detected at import time; if missing, the module degrades to a librosa-
based fallback so the pipeline still produces an end-to-end answer.
"""
__version__ = "0.1.0"
