"""Typer CLI entry point for the Presence Analyzer."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from presence.config import Config

app = typer.Typer(
    name="presence",
    help="Analyze creator presence from video — pose-based scoring with confidence bounds.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _load_config(config_path: Optional[Path]) -> Config:
    return Config.load(config_path)


def _warmup(config: Config) -> None:
    """Trigger numba JIT compile + model load so first user call is fast."""
    from presence.filters.one_euro import warmup_filter

    warmup_filter()


@app.command()
def analyze(
    video: Annotated[Path, typer.Argument(help="Path to video file (mp4/mov/webm)")],
    config_path: Annotated[
        Optional[Path], typer.Option("--config", "-c", help="Path to config YAML")
    ] = None,
    output: Annotated[
        Optional[Path], typer.Option("--output", "-o", help="Write JSON result to this path")
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit JSON to stdout instead of a rich panel")
    ] = False,
) -> None:
    """Analyze a single video and print the result."""
    if not video.exists():
        console.print(f"[red]error:[/red] {video} does not exist")
        raise typer.Exit(1)

    config = _load_config(config_path)

    from presence.pipeline import analyze_video

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
        disable=json_output,
    ) as progress:
        t = progress.add_task("Warming up…", total=None)
        _warmup(config)
        progress.update(t, description="Analyzing video…")
        result = analyze_video(video, config)

    if json_output:
        typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, default=str))
    else:
        _render_result(result)

    if output is not None:
        output.write_text(json.dumps(result.model_dump(mode="json"), indent=2, default=str))
        if not json_output:
            console.print(f"[dim]wrote result to {output}[/dim]")


def _render_result(result) -> None:
    score = result.score
    feats = result.features
    duration = feats.duration_sec
    cal = "calibrated" if score.is_calibrated else "uncalibrated (fallback)"
    body = []
    body.append(f"[bold]Video[/bold]: {result.video_path.name}  ({duration:.1f}s)")
    body.append(
        f"[bold]Overall Score[/bold]: {score.score:.0f} ± "
        f"{(score.confidence_high - score.confidence_low) / 2:.0f}  [dim][{cal}][/dim]"
    )
    body.append("")
    for name, value in score.components.items():
        qkey = f"quality_{name}"
        q = getattr(feats, qkey, None)
        q_str = "high" if q is not None and q > 0.8 else "medium" if q is not None and q > 0.5 else "low"
        body.append(f"  • {name:<20} {value:>5.0f}  [dim]({q_str} quality)[/dim]")
    body.append("")
    body.append(
        f"[dim]Timing: {sum(result.timings.values()):.2f}s total  "
        + " | ".join(f"{k}: {v:.2f}s" for k, v in result.timings.items())
        + "[/dim]"
    )
    console.print(Panel("\n".join(body), title="Presence Analysis", border_style="cyan"))


@app.command()
def batch(
    directory: Annotated[Path, typer.Argument(help="Directory of videos")],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output Parquet path")
    ] = Path("results.parquet"),
    config_path: Annotated[
        Optional[Path], typer.Option("--config", "-c", help="Config YAML path")
    ] = None,
    workers: Annotated[
        Optional[int], typer.Option("--workers", "-w", help="Override worker count")
    ] = None,
) -> None:
    """Analyze every video in a directory in parallel; write Parquet."""
    if not directory.is_dir():
        console.print(f"[red]error:[/red] {directory} is not a directory")
        raise typer.Exit(1)
    config = _load_config(config_path)
    if workers is not None:
        config.runtime.num_workers = workers

    from presence.pipeline import batch_analyze

    n_ok, n_err = batch_analyze(directory, output, config)
    console.print(f"[green]✓[/green] {n_ok} videos analyzed, {n_err} errors")
    console.print(f"results: {output}")


@app.command()
def bench(
    test_dir: Annotated[
        Path, typer.Option("--test-dir", help="Directory of bench videos")
    ] = Path("tests/data"),
    config_path: Annotated[
        Optional[Path], typer.Option("--config", "-c", help="Config YAML path")
    ] = None,
) -> None:
    """Run the performance benchmark suite."""
    config = _load_config(config_path)

    if not test_dir.exists():
        console.print(f"[yellow]no bench videos found at {test_dir}[/yellow]")
        console.print("Drop sample .mp4 files into tests/data/ and re-run.")
        return

    videos = sorted(list(test_dir.glob("*.mp4")) + list(test_dir.glob("*.mov")))
    if not videos:
        console.print(f"[yellow]no .mp4 or .mov files in {test_dir}[/yellow]")
        return

    from presence.io.cache import PoseCache
    from presence.io.video import compute_content_hash
    from presence.pipeline import analyze_video

    _warmup(config)
    cache = PoseCache(config.cache.dir)

    table = Table(title="Presence Analyzer Benchmarks")
    table.add_column("Video", style="cyan")
    table.add_column("Duration", justify="right")
    table.add_column("Cold", justify="right")
    table.add_column("Warm", justify="right")
    table.add_column("Budget", justify="right")
    table.add_column("Status")

    all_pass = True
    for v in videos:
        # Cold: invalidate cache for this video
        h = compute_content_hash(v)
        cache.invalidate(h, model=config.pose.model, sample_fps=config.pose.sample_fps)
        t0 = time.perf_counter()
        result = analyze_video(v, config)
        cold = time.perf_counter() - t0

        t0 = time.perf_counter()
        analyze_video(v, config)
        warm = time.perf_counter() - t0

        budget = max(8.0, result.features.duration_sec * 0.20)
        ok = cold <= budget * 2.0  # 2× = hard limit
        status = "[green]✓[/green]" if ok else "[red]✗[/red]"
        all_pass = all_pass and ok
        table.add_row(
            v.name,
            f"{result.features.duration_sec:.1f}s",
            f"{cold:.2f}s",
            f"{warm:.2f}s",
            f"≤{budget:.1f}s",
            status,
        )

    console.print(table)
    if not all_pass:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
