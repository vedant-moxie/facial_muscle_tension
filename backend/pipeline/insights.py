"""
Natural-language insight generator.

Each insight is a dict:
    { type: "good" | "watch" | "info", text: "..." }
"""
from __future__ import annotations

from typing import List

import pandas as pd


def _badge(t: str, text: str) -> dict:
    return {"type": t, "text": text}


def generate_insights(m1, m2, m3, m4, m5, ensemble, au_df: pd.DataFrame) -> List[dict]:
    out: List[dict] = []

    # 1) Overall headline
    score = ensemble["overall_score"]
    label = ensemble["label"]
    if score >= 0.8:
        out.append(_badge("good", f"Overall {score:.2f} ({label}) — the speaker reads as physically at ease."))
    elif score >= 0.5:
        out.append(_badge("info", f"Overall {score:.2f} ({label}) — measurable effort but within normal speaking range."))
    else:
        out.append(_badge("watch", f"Overall {score:.2f} ({label}) — sustained strain markers across multiple channels."))

    # 2) Micro-expression events
    events = m2.get("events", [])
    if len(events) == 0:
        out.append(_badge("good", "No micro-expression bursts detected — no leaked suppression."))
    elif len(events) <= 2:
        out.append(_badge("info", f"{len(events)} micro-expression event(s) — within normal range for spontaneous speech."))
    else:
        ts = ", ".join(e["timestamp_str"] for e in events[:3])
        out.append(_badge("watch", f"{len(events)} micro-expression bursts (at {ts}…). Suppressed effort is leaking through."))

    # 3) Blink rate
    blink = m5["blink_rate_per_min"]
    if 12 <= blink <= 20:
        out.append(_badge("good", f"Blink rate {blink:.1f}/min is in the genuine-relaxation band (12–20/min)."))
    elif blink < 10:
        out.append(_badge("watch", f"Low blink rate ({blink:.1f}/min) — typical of active suppression or high cognitive load."))
    elif blink > 25:
        out.append(_badge("watch", f"Elevated blink rate ({blink:.1f}/min) — possible anxiety or dry-eye / lighting artefact."))
    else:
        out.append(_badge("info", f"Blink rate {blink:.1f}/min — slightly outside the relaxed band but not concerning."))

    # 4) Blink regularity
    if m5["summary"].get("blink_regularity", "").startswith("regular"):
        out.append(_badge("watch", "Blink intervals are unusually regular — possible deliberate control of expression."))

    # 5) Asymmetry
    asym = m3["mean"]
    if 0.08 <= asym <= 0.30:
        out.append(_badge("good", f"Asymmetry index {asym:.2f} sits in the genuine-expression band (0.08–0.30)."))
    elif asym < 0.08:
        out.append(_badge("watch", f"Very low asymmetry ({asym:.2f}) — held or performed expression marker."))
    else:
        out.append(_badge("info", f"High asymmetry ({asym:.2f}) — strain or transient distress."))

    # 6) Coherence
    lag = abs(m4["summary"].get("peak_lag_ms", 0))
    if lag <= 100:
        out.append(_badge("good", f"Upper/lower face sync within {lag:.0f} ms — consistent with spontaneous emotion."))
    elif lag >= 200:
        out.append(_badge("watch", f"Upper/lower face lag of {lag:.0f} ms — smile may be running ahead of the eyes (performed pattern)."))

    # 7) Duchenne smile check (AU6 + AU12)
    cols = list(au_df.columns)
    if "AU06" in cols and "AU12" in cols:
        au6_mean = float(au_df["AU06"].mean())
        au12_mean = float(au_df["AU12"].mean())
        au6_var = float(au_df["AU06"].std() or 0)
        if au6_mean > 1.0 and au12_mean > 1.0 and au6_var > 0.4:
            out.append(_badge("good", "AU6 (cheek raiser) co-active with AU12 (lip pull) and varies dynamically — Duchenne markers present."))
        elif au12_mean > 1.5 and au6_mean < 0.6:
            out.append(_badge("watch", "AU12 (lip pull) high but AU6 (cheek raiser) flat — classic non-Duchenne ‘polite’ smile."))

    # 8) Tension AUs — worst moments
    worst = m1["summary"].get("worst_moments", [])
    if worst and worst[0]["effortless"] < 0.4:
        t = worst[0]["timestamp"]
        out.append(_badge("watch", f"Peak strain at {int(t//60)}:{int(t%60):02d} — sustained AU4/AU7 activation."))

    return out
