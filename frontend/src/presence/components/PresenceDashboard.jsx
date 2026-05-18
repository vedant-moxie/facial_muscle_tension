import { useEffect, useRef } from "react"

function scoreColor(s) {
  if (s >= 80) return "text-good"
  if (s >= 65) return "text-good/80"
  if (s >= 50) return "text-info"
  if (s >= 35) return "text-watch"
  return "text-watch"
}

function qualityChip(q) {
  if (q == null) return null
  const label =
    q > 0.8 ? "high quality" : q > 0.5 ? "medium quality" : "low quality"
  const cls =
    q > 0.8 ? "border-good/40 text-good bg-good/5"
    : q > 0.5 ? "border-info/40 text-info bg-info/5"
    : "border-watch/40 text-watch bg-watch/5"
  return <span className={`chip ${cls}`}>{label}</span>
}

function fmt(v, digits = 2) {
  if (v == null || Number.isNaN(v)) return "—"
  return Number(v).toFixed(digits)
}

const COMPONENT_LABELS = {
  shoulder: "Shoulder symmetry",
  head:     "Head position",
  arms:     "Arm naturalness",
  fluidity: "Movement fluidity",
}

const FEATURE_LABELS = {
  shoulder_median_deg:        "Shoulder line · median (°)",
  shoulder_iqr_deg:           "Shoulder line · IQR (°)",
  head_pitch_dev_median:      "Head pitch · median dev (°)",
  head_pitch_dev_iqr:         "Head pitch · IQR (°)",
  head_yaw_dev_median:        "Head yaw · median dev (°)",
  head_yaw_dev_iqr:           "Head yaw · IQR (°)",
  elbow_var_median:           "Elbow angle · variance",
  gesture_freq_per_min:       "Gestures · per minute",
  gesture_amplitude_median:   "Gesture amplitude · median",
  sparc_nose:                 "SPARC · nose",
  sparc_wrist_mean:           "SPARC · wrists (mean)",
}

export default function PresenceDashboard({ result, videoUrl }) {
  const {
    filename, duration_secs, fps_effective,
    score = {}, features = {}, timings = {}, warnings = [],
  } = result

  const components = score.components || {}
  const totalTime = Object.values(timings).reduce((a, b) => a + Number(b || 0), 0)

  const videoRef = useRef(null)

  return (
    <div className="space-y-6">
      {/* Header card */}
      <div className="card p-5 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="text-xs uppercase tracking-wider text-muted mb-1">Source</div>
          <div className="font-mono text-sm truncate max-w-[480px]">{filename || "video"}</div>
          <div className="text-xs text-muted mt-1">
            {fmt(duration_secs, 1)} s · sampled @ {fmt(fps_effective, 0)} fps
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {warnings.map((w, i) => (
            <span key={i} className="chip border-watch/40 text-watch bg-watch/5">{w}</span>
          ))}
          <span
            className={[
              "chip",
              score.is_calibrated
                ? "border-good/40 text-good bg-good/5"
                : "border-info/40 text-info bg-info/5",
            ].join(" ")}
          >
            {score.is_calibrated ? "calibrated" : "uncalibrated (fallback)"}
          </span>
        </div>
      </div>

      {/* Score + video */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="card p-6 lg:col-span-1">
          <div className="text-xs uppercase tracking-wider text-muted">Overall</div>
          <div className={`mt-2 font-mono text-6xl font-semibold ${scoreColor(score.score ?? 0)}`}>
            {fmt(score.score, 0)}
          </div>
          <div className="text-sm text-muted mt-1">
            ± {fmt(((score.confidence_high ?? 0) - (score.confidence_low ?? 0)) / 2, 1)} ·
            95% CI [{fmt(score.confidence_low, 0)}, {fmt(score.confidence_high, 0)}]
          </div>
          <div className="text-xs text-muted mt-1">
            Quality overall: {fmt((score.quality_overall ?? 0) * 100, 0)}%
          </div>

          <div className="mt-6 space-y-2">
            {Object.entries(components).map(([k, v]) => {
              const qkey = `quality_${k}`
              const q = features[qkey]
              return (
                <div
                  key={k}
                  className="card p-3 border-line/60 flex items-center justify-between"
                >
                  <div>
                    <div className="text-xs uppercase tracking-wider text-muted">
                      {COMPONENT_LABELS[k] || k}
                    </div>
                    <div className={`mt-1 font-mono text-2xl ${scoreColor(v)}`}>
                      {fmt(v, 0)}
                    </div>
                  </div>
                  <div>{qualityChip(q)}</div>
                </div>
              )
            })}
          </div>
        </div>

        <div className="card p-2 lg:col-span-2 overflow-hidden">
          {videoUrl ? (
            <video
              ref={videoRef}
              src={videoUrl}
              controls
              className="w-full h-full max-h-[520px] rounded-lg bg-black object-contain"
            />
          ) : (
            <div className="h-full min-h-[300px] flex items-center justify-center text-muted text-sm p-8 text-center">
              Video preview unavailable.<br />
              Upload another to see it side-by-side with the score.
            </div>
          )}
        </div>
      </div>

      {/* Feature table */}
      <div className="card p-5">
        <div className="text-xs uppercase tracking-wider text-muted mb-3">
          Raw features
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-2 text-sm">
          {Object.entries(FEATURE_LABELS).map(([k, label]) => (
            <div key={k} className="flex items-center justify-between border-b border-line/40 py-1.5">
              <div className="text-muted">{label}</div>
              <div className="font-mono">{fmt(features[k], 3)}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Timings */}
      <div className="card p-5">
        <div className="flex items-baseline justify-between mb-3">
          <div className="text-xs uppercase tracking-wider text-muted">
            Stage timings
          </div>
          <div className="font-mono text-xs text-muted">
            total · {fmt(totalTime, 2)} s
          </div>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {Object.entries(timings).map(([k, v]) => (
            <div key={k} className="card p-2 border-line/60">
              <div className="text-[10px] uppercase tracking-wider text-muted">
                {k.replace(/_/g, " ")}
              </div>
              <div className="mt-0.5 font-mono text-sm">{fmt(v, 3)} s</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
