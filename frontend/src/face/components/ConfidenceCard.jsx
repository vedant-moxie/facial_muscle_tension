/*
 * Confidence card — surfaces how trustworthy the headline score is.
 *
 * We do NOT have ground truth (no EMG, no labelled videos), so we can't claim
 * "the result is X% accurate". What we CAN show is how robust the inputs
 * were: tracking coverage, method agreement, baseline calibration quality,
 * and per-frame signal stability. Each is a 0–1 number; the overall
 * confidence is their mean, bucketed into High / Medium / Low.
 *
 * Reading these signals:
 *
 *   • Tracking      — fraction of frames where the face was found AND the
 *                     head was within the pose gate (|yaw|<35°, |pitch|<30°).
 *                     Frames outside this are skipped, so a low number means
 *                     the score is based on less data than the video length.
 *
 *   • Methods agree — variance across the 5 method scores. If they all land
 *                     near the same value, that's converging evidence; if
 *                     they disagree, one channel is picking up something the
 *                     others aren't and the headline is fragile.
 *
 *   • Baseline      — whether per-person calibration used the proper
 *                     25th-percentile-of-whole-video path or the fallback
 *                     window mean (short or low-coverage video).
 *
 *   • Signal        — std of per-frame effortlessness from the tension AU
 *                     method. Tiny std on a long video usually means the
 *                     signal is flat (frame-zero filler) rather than
 *                     genuinely steady; we treat that as low confidence.
 */

function pct(v) { return `${Math.round((v || 0) * 100)}%` }
function band(v) {
  if (v >= 0.75) return { label: "High",   color: "#3ddc97", chip: "good"  }
  if (v >= 0.50) return { label: "Medium", color: "#5aa9ff", chip: "info"  }
  return                  { label: "Low",    color: "#ff7a59", chip: "watch" }
}

function computeConfidence(result) {
  const ms = result?.method_scores || {}
  const scores = Object.values(ms).map(m => m?.score).filter(s => typeof s === "number")
  const meanScore = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0.5
  const variance = scores.length
    ? scores.reduce((a, b) => a + (b - meanScore) ** 2, 0) / scores.length
    : 0.25

  // How much do the methods agree? std=0 → perfect agreement → 1.0
  const std = Math.sqrt(variance)
  const agreement = Math.max(0, 1 - std / 0.35)            // 0.35 std = total disagreement

  // Tracking: scoreable_fraction is plumbed through baseline; fall back to
  // tension_aus.summary.scoreable_fraction if not present.
  const tracking =
    result?.baseline?.scoreable_fraction
    ?? ms.tension_aus?.scoreable_fraction
    ?? 1.0

  // Baseline calibration quality. "percentile" with reasonable frame count
  // = good; "window" fallback = mediocre; short_video chips weight down.
  const baselineMethod = result?.baseline?.method
  const calibrationFrames = result?.baseline?.calibration_frames || 0
  let baseline = 1.0
  if (baselineMethod === "window") baseline = 0.55
  if (calibrationFrames < 120 && baselineMethod !== "percentile") baseline = Math.min(baseline, 0.5)

  // Signal stability — tension_aus reports std of its per-frame score.
  const signalStd = ms.tension_aus?.std ?? 0.05
  // A bit of variance is HEALTHY; complete flatness is suspicious.
  let signal = 1.0
  if (signalStd < 0.02) signal = 0.4               // suspiciously flat
  else if (signalStd > 0.30) signal = 0.6          // very volatile

  const overall = (tracking + agreement + baseline + signal) / 4
  return { tracking, agreement, baseline, signal, overall, std, scores }
}

export default function ConfidenceCard({ result }) {
  const c = computeConfidence(result)
  const b = band(c.overall)
  return (
    <div className="card p-5">
      <div className="flex items-baseline justify-between mb-3">
        <div>
          <div className="text-sm font-medium">How confident should you be in this result?</div>
          <div className="text-xs text-muted">
            These are quality signals, not ground-truth accuracy — there's no labelled
            "effortless / strained" ground truth for this video.
          </div>
        </div>
        <span className={`chip border-${b.chip}/40 text-${b.chip} bg-${b.chip}/5`}>
          {b.label} confidence · {pct(c.overall)}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
        <Probe
          title="Tracking coverage"
          value={c.tracking}
          hint="Fraction of frames where the face was found AND head-pose was within the gate. Frames outside are skipped, not zero-filled."
        />
        <Probe
          title="Methods agree"
          value={c.agreement}
          hint={`Spread across the 5 method scores. σ = ${c.std.toFixed(2)} (lower = methods converge on the same answer).`}
        />
        <Probe
          title="Baseline calibration"
          value={c.baseline}
          hint={
            result?.baseline?.method === "percentile"
              ? "Person-neutral derived from 25th-percentile across the whole video (OpenFace-style)."
              : "Fallback baseline: first window mean (used when too few scoreable frames). Less robust."
          }
        />
        <Probe
          title="Signal stability"
          value={c.signal}
          hint={`Per-frame effortlessness σ = ${(result?.method_scores?.tension_aus?.std ?? 0).toFixed(2)}. Healthy faces show some variance; near-zero is suspicious.`}
        />
      </div>

      <div className="mt-4 grid grid-cols-1 md:grid-cols-5 gap-2 text-[11px]">
        {Object.entries(result?.method_scores || {}).map(([name, m]) => (
          <div key={name} className="card p-2">
            <div className="text-muted">{name.replace(/_/g, " ")}</div>
            <div className="font-mono mt-0.5">{(m?.score ?? 0).toFixed(2)}</div>
          </div>
        ))}
      </div>

      <details className="mt-4 text-xs text-muted">
        <summary className="cursor-pointer hover:text-text">
          How to read these numbers
        </summary>
        <div className="mt-2 space-y-1.5 leading-relaxed">
          <div>• <span className="text-text">High</span>: tracking ≥ 80%, methods within σ&nbsp;≤&nbsp;0.10, baseline on percentile path, signal σ in [0.02, 0.30]. Treat the headline number as a reliable estimate.</div>
          <div>• <span className="text-text">Medium</span>: at least one probe in the 50–75% band. The direction (effortless / strained) is probably right; the exact number is less reliable.</div>
          <div>• <span className="text-text">Low</span>: tracking failed often, methods disagree, or the baseline fell back. The number is descriptive of what we did measure, but extrapolating to "this person is X" is not warranted.</div>
          <div className="pt-2">
            What this card does NOT tell you: whether the system measures what FACS theory says it does on <em>this person specifically</em> (would require EMG-instrumented test recording). It only tells you whether the inputs were good enough for the math we ran.
          </div>
        </div>
      </details>
    </div>
  )
}

function Probe({ title, value, hint }) {
  const b = band(value)
  return (
    <div className="card p-3">
      <div className="flex items-baseline justify-between">
        <div className="text-[11px] uppercase tracking-wider text-muted">{title}</div>
        <div className="font-mono text-sm" style={{ color: b.color }}>{pct(value)}</div>
      </div>
      <div className="mt-2 h-1.5 rounded bg-panel2 overflow-hidden">
        <div className="h-full rounded" style={{ width: `${value * 100}%`, background: b.color }} />
      </div>
      <div className="text-[11px] text-muted mt-2 leading-snug">{hint}</div>
    </div>
  )
}
