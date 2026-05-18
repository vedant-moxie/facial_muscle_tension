"""
Orchestrator for the Sonic Rebellion pipeline.

Sequential stages (CPU-bound, fast once warm):
  1. fetch         — gather MP4 paths from URLs / direct uploads
  2. extract       — ffmpeg → peak-energy 10s mono clip per reel
  3. separate      — Spleeter (or HPSS fallback) → vocals + accompaniment

Parallel branches (different models, no shared state):
  4a. score_novelty       (full clip)
  4b. score_vocal_edge    (vocal stem)
  4c. score_genre_edge    (accompaniment stem)

  5. aggregate → composite creator score

A `progress_cb` callback is invoked at every stage so the FastAPI layer
can update job["stage"] / job["progress"] for client polling.
"""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional

from audio.config import SAMPLE_RATE
from audio.pipeline.extractor   import batch_extract
from audio.pipeline.genre_edge  import score_genre_edge_batch
from audio.pipeline.novelty     import score_novelty_batch
from audio.pipeline.scorer      import aggregate_reel_scores
from audio.pipeline.separator   import separate_batch
from audio.pipeline.vocal_edge  import score_vocal_edge_batch

log = logging.getLogger(__name__)

ProgressCb = Optional[Callable[[str, float], None]]


def _emit(cb: ProgressCb, stage: str, p: float) -> None:
    if cb is not None:
        try:
            cb(stage, p)
        except Exception:                           # noqa: BLE001
            pass


def run_pipeline(
    reel_paths: List[str],
    progress_cb: ProgressCb = None,
) -> Dict:
    if not reel_paths:
        raise ValueError("run_pipeline called with no reel paths.")

    started = time.time()
    n = len(reel_paths)

    _emit(progress_cb, "extracting audio", 0.10)
    clips = batch_extract(reel_paths)                # list of (waveform, sr)
    waveforms = [c[0] for c in clips]
    sr = clips[0][1] if clips else SAMPLE_RATE

    _emit(progress_cb, "separating vocals & accompaniment", 0.30)
    stems = separate_batch(waveforms, sr=sr)
    vocals = [s["vocals"]        for s in stems]
    accomp = [s["accompaniment"] for s in stems]
    separator_backend = stems[0].get("separator", "unknown") if stems else "unknown"

    # 3-way concurrent scoring — different models, no shared state.
    _emit(progress_cb, "scoring · novelty / vocal edge / genre edge", 0.55)
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="sr_score") as ex:
        fut_nov = ex.submit(score_novelty_batch,    waveforms, sr)
        fut_voc = ex.submit(score_vocal_edge_batch, vocals,    sr)
        fut_gen = ex.submit(score_genre_edge_batch, accomp,    sr)
        novelty_results    = fut_nov.result()
        vocal_edge_scores  = fut_voc.result()
        genre_edge_scores  = fut_gen.result()

    _emit(progress_cb, "aggregating", 0.92)
    result = aggregate_reel_scores(
        novelty_results,
        vocal_edge_scores,
        genre_edge_scores,
        reel_filenames=[os.path.basename(p) for p in reel_paths],
    )

    result["wall_clock_s"]     = round(time.time() - started, 1)
    result["separator_backend"] = separator_backend
    result["sample_rate"]       = sr
    result["n_reels_input"]     = n

    _emit(progress_cb, "done", 1.0)
    log.info("Sonic Rebellion run: %d reels in %.1fs → %.2f (%s)",
             n, result["wall_clock_s"], result["creator_score"], result["score_label"])
    return result
