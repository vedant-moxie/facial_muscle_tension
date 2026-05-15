const NAME = {
  tension_aus:        "Tension AUs",
  micro_expressions:  "Micro-expressions",
  asymmetry:          "Asymmetry",
  coherence:          "Upper/lower coherence",
  drift_blink:        "Drift + blink",
}

const TIER_COLOR = {
  2: "border-good/40 text-good bg-good/5",
  3: "border-info/40 text-info bg-info/5",
  4: "border-muted/40 text-muted bg-muted/5",
}

function barColor(s) {
  if (s >= 0.75) return "from-good to-good/60"
  if (s >= 0.5)  return "from-info to-info/60"
  if (s >= 0.35) return "from-watch to-watch/60"
  return "from-watch to-watch"
}

export default function MethodScorecard({ name, summary }) {
  const pct = Math.max(0, Math.min(100, (summary.score ?? 0) * 100))
  return (
    <div className="rounded-xl border border-line p-3 bg-panel2/40">
      <div className="flex items-center justify-between gap-2">
        <div className="text-sm font-medium truncate">{NAME[name] || name}</div>
        <div className="flex items-center gap-1.5">
          <span className={`chip ${TIER_COLOR[summary.tier] || TIER_COLOR[4]}`}>T{summary.tier}</span>
          <span className="chip border-line text-muted">w {summary.weight}</span>
          <span className="font-mono text-sm tabular-nums">{(summary.score ?? 0).toFixed(2)}</span>
        </div>
      </div>
      <div className="mt-2 h-1.5 bg-line rounded-full overflow-hidden">
        <div
          className={`h-full bg-gradient-to-r ${barColor(summary.score ?? 0)}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      {summary.interpretation && (
        <div className="mt-2 text-xs text-muted leading-snug">{summary.interpretation}</div>
      )}
    </div>
  )
}
