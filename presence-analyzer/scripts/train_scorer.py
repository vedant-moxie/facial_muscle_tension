"""Train a calibrated presence-score regression.

Gated on labels: expects labels.csv with columns video_path, rating_1..rating_N.
Outputs data/models/scorer_v1.joblib (sklearn Pipeline + bootstrap_std).

Run AFTER ratings are collected. Until then, the pipeline runs in fallback mode.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import typer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from presence.config import Config
from presence.pipeline import analyze_video

FEATURE_ORDER = [
    "shoulder_median_deg",
    "shoulder_iqr_deg",
    "head_pitch_dev_median",
    "head_pitch_dev_iqr",
    "head_yaw_dev_median",
    "head_yaw_dev_iqr",
    "elbow_var_median",
    "gesture_freq_per_min",
    "gesture_amplitude_median",
    "sparc_nose",
    "sparc_wrist_mean",
]


def _safe_features(row, config: Config) -> np.ndarray | None:
    try:
        path = Path(row["video_path"])
        if not path.exists():
            return None
        result = analyze_video(path, config)
        return np.array([getattr(result.features, k) for k in FEATURE_ORDER], dtype=np.float64)
    except Exception as e:
        print(f"skipping {row['video_path']}: {e}", file=sys.stderr)
        return None


def main(
    labels_csv: Path = typer.Argument(..., help="CSV with video_path and rating columns"),
    output: Path = typer.Option(
        Path("data/models/scorer_v1.joblib"), "--output", "-o"
    ),
    config_path: Path | None = typer.Option(None, "--config", "-c"),
    n_bootstrap: int = typer.Option(200, "--bootstrap"),
) -> None:
    config = Config.load(config_path)
    df = pl.read_csv(labels_csv)
    rating_cols = [c for c in df.columns if c.startswith("rating")]
    if not rating_cols:
        typer.echo("[error] no rating_* columns found", err=True)
        raise typer.Exit(1)

    # Mean rating per video; reject high-disagreement
    df = df.with_columns(
        [
            pl.mean_horizontal(rating_cols).alias("y"),
            pl.concat_list(rating_cols).list.std().alias("rating_std"),
        ]
    )
    n_total = df.height
    df = df.filter(pl.col("rating_std") < 1.5)
    typer.echo(f"using {df.height}/{n_total} videos (rest filtered for rater disagreement)")

    X_rows, y_rows = [], []
    for row in df.iter_rows(named=True):
        feats = _safe_features(row, config)
        if feats is None or np.any(np.isnan(feats)):
            continue
        X_rows.append(feats)
        y_rows.append(row["y"])
    X = np.vstack(X_rows)
    y = np.array(y_rows, dtype=np.float64)
    typer.echo(f"final training set: {X.shape[0]} videos, {X.shape[1]} features")

    pipe = Pipeline(
        [("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]
    )
    scores = cross_val_score(pipe, X, y, scoring="r2", cv=KFold(n_splits=5, shuffle=True, random_state=42))
    typer.echo(f"5-fold CV R² = {scores.mean():.3f} ± {scores.std():.3f}")

    pipe.fit(X, y)

    rng = np.random.default_rng(42)
    boot_preds = np.empty((n_bootstrap, len(y)))
    for b in range(n_bootstrap):
        idx = rng.integers(0, len(y), size=len(y))
        bs = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        bs.fit(X[idx], y[idx])
        boot_preds[b] = bs.predict(X)
    bootstrap_std = float(np.median(np.std(boot_preds, axis=0)))
    typer.echo(f"bootstrap_std = {bootstrap_std:.2f}")

    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": pipe,
            "feature_order": FEATURE_ORDER,
            "bootstrap_std": bootstrap_std,
            "cv_r2_mean": float(scores.mean()),
            "n_train": int(X.shape[0]),
        },
        output,
    )
    typer.echo(f"wrote {output}")


if __name__ == "__main__":
    typer.run(main)
