"""Standalone batch runner script (mirrors `presence batch`)."""
from __future__ import annotations

import sys
from pathlib import Path

import typer

from presence.config import Config
from presence.pipeline import batch_analyze


def main(
    video_dir: Path = typer.Argument(...),
    output: Path = typer.Option(Path("results.parquet"), "--output", "-o"),
    config_path: Path | None = typer.Option(None, "--config", "-c"),
    workers: int | None = typer.Option(None, "--workers", "-w"),
) -> None:
    config = Config.load(config_path)
    if workers is not None:
        config.runtime.num_workers = workers
    n_ok, n_err = batch_analyze(video_dir, output, config)
    typer.echo(f"{n_ok} ok, {n_err} errors → {output}")
    sys.exit(0 if n_err == 0 else 1)


if __name__ == "__main__":
    typer.run(main)
