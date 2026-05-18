"""End-to-end performance benchmarks.

Runs every video under tests/data/ from cold (cache cleared) and warm
(cache hit), prints a table, and writes a JSON report keyed by git SHA.
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from presence.config import Config
from presence.filters.one_euro import warmup_filter
from presence.io.cache import PoseCache
from presence.io.video import compute_content_hash
from presence.pipeline import analyze_video


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "no-git"


def main() -> int:
    console = Console()
    config = Config.load()
    test_dir = Path("tests/data")
    videos = sorted(list(test_dir.glob("*.mp4")) + list(test_dir.glob("*.mov")))
    if not videos:
        console.print(f"[yellow]no test videos in {test_dir}[/yellow]")
        return 0

    warmup_filter()
    cache = PoseCache(config.cache.dir)

    table = Table(title="Presence Analyzer Benchmarks")
    for col in ("Video", "Duration", "Cold", "Warm", "Budget", "Status"):
        table.add_column(col, justify="right" if col not in ("Video", "Status") else "left")

    results = []
    all_pass = True
    for v in videos:
        h = compute_content_hash(v)
        cache.invalidate(h, model=config.pose.model, sample_fps=config.pose.sample_fps)
        t0 = time.perf_counter()
        result = analyze_video(v, config)
        cold = time.perf_counter() - t0

        t0 = time.perf_counter()
        analyze_video(v, config)
        warm = time.perf_counter() - t0

        budget = max(12.0, result.features.duration_sec * 0.20)
        hard_limit = budget * 2.0
        ok = cold <= hard_limit and warm <= 1.0
        all_pass = all_pass and ok
        status = "[green]✓[/green]" if ok else "[red]✗[/red]"
        table.add_row(
            v.name,
            f"{result.features.duration_sec:.1f}s",
            f"{cold:.2f}s",
            f"{warm:.2f}s",
            f"≤{budget:.1f}s",
            status,
        )
        results.append(
            {
                "video": v.name,
                "duration_sec": result.features.duration_sec,
                "cold_sec": cold,
                "warm_sec": warm,
                "budget_sec": budget,
                "ok": ok,
                "timings": result.timings,
            }
        )

    console.print(table)
    out_dir = Path("data/bench")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"{_git_sha()}-{stamp}.json"
    out.write_text(json.dumps(results, indent=2))
    console.print(f"[dim]→ wrote {out}[/dim]")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
