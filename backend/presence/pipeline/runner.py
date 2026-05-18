"""Adapter that runs the standalone `presence-analyzer` CLI as a subprocess.

Why a subprocess instead of an in-process import?

The presence-analyzer package pins `numpy==1.26.4`, `scipy==1.13.1`, and
`mediapipe==0.10.18`. The backend's existing facial-tension pipeline (py-feat)
pins `numpy==1.23.5`, `scipy<1.12`, `mediapipe<0.10.15`. Co-installing the two
trees breaks py-feat. A subprocess hop costs ~1-2s of cold start but lets the
two analyzers coexist cleanly.

The CLI's `--json` mode prints a single JSON document to stdout that mirrors
the in-process `AnalysisResult`.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

log = logging.getLogger("presence_runner")

# Where the standalone presence-analyzer project lives. Resolves relative to
# this file so it works regardless of cwd.
# __file__ is backend/presence/pipeline/runner.py → parents[3] == repo root.
PRESENCE_ROOT = (Path(__file__).resolve().parents[3] / "presence-analyzer").resolve()


def _presence_binary() -> Path:
    """The `presence` console script in the presence-analyzer venv."""
    bin_path = PRESENCE_ROOT / ".venv" / "bin" / "presence"
    if not bin_path.exists():
        raise FileNotFoundError(
            f"presence CLI not found at {bin_path}. "
            f"Did you run `uv sync` in {PRESENCE_ROOT}?"
        )
    return bin_path


# Regex to spot rich-progress / model-download chatter in stderr so we can
# turn them into progress events for the UI.
_STAGE_PATTERNS = [
    (re.compile(r"Warming up", re.IGNORECASE),         "warming up filters",       0.10),
    (re.compile(r"Downloading", re.IGNORECASE),        "downloading mediapipe models", 0.20),
    (re.compile(r"GL version", re.IGNORECASE),         "initializing graph",       0.30),
    (re.compile(r"Analyzing", re.IGNORECASE),          "extracting pose",          0.40),
    (re.compile(r"inference_feedback", re.IGNORECASE), "running inference",        0.60),
]


def run_presence(
    video_path: Path,
    progress_cb: Callable[[str, float], None] | None = None,
    timeout_sec: float = 600.0,
) -> dict:
    """Run the presence analyzer on the given video; return the JSON result.

    Raises subprocess.CalledProcessError on non-zero exit, FileNotFoundError if
    the CLI isn't installed, and TimeoutError on wall-clock timeout.
    """
    binary = _presence_binary()
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    cmd = [str(binary), "analyze", str(video_path), "--json"]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    log.info("spawning %s", " ".join(cmd))

    started = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(PRESENCE_ROOT),
        env=env,
        text=True,
        bufsize=1,
    )

    # Tail stderr in a background thread so progress flows even while the
    # subprocess is mid-flight.
    def _tail_stderr() -> None:
        if proc.stderr is None:
            return
        for line in proc.stderr:
            line = line.rstrip()
            if not line:
                continue
            log.debug("[presence stderr] %s", line)
            if progress_cb is None:
                continue
            for pat, stage, pct in _STAGE_PATTERNS:
                if pat.search(line):
                    progress_cb(stage, pct)
                    break

    t = threading.Thread(target=_tail_stderr, daemon=True)
    t.start()

    try:
        stdout, _ = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired as e:
        proc.kill()
        proc.communicate()
        raise TimeoutError(
            f"presence-analyzer exceeded {timeout_sec}s on {video_path.name}"
        ) from e

    if proc.returncode != 0:
        # stderr already streamed via the tailing thread; expose the last bit
        # in the error message.
        last_err = ""
        if proc.stderr is not None:
            try:
                last_err = proc.stderr.read() or ""
            except Exception:
                pass
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output=stdout, stderr=last_err
        )

    elapsed = time.time() - started
    log.info("presence finished in %.1fs", elapsed)

    # The CLI emits ONLY JSON on stdout in --json mode, so this should parse.
    # We still defensively grab the last JSON object in case anything sneaks in.
    raw = stdout.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Find the last "{" and parse from there
        start = raw.rfind("{\n")
        if start < 0:
            start = raw.find("{")
        if start < 0:
            raise RuntimeError(f"presence CLI produced no JSON; output was:\n{raw[:500]}")
        return json.loads(raw[start:])
