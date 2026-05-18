/*
 * Sonic Rebellion dashboard.
 *
 * Top:  Creator composite score (big number + label) + 3 sub-score gauges.
 * Mid:  Run metadata — backends used, separator type, wall-clock, n_reels.
 * Grid: One ReelCard per reel with novelty / vocal / genre breakdown.
 */

function scoreColor(s) {
  if (s >= 0.75) return "#a78bfa"      // rebel
  if (s >= 0.55) return "#5aa9ff"      // edge-leaning
  if (s >= 0.35) return "#3ddc97"      // balanced
  return "#8a90a4"                     // mainstream
}

function Gauge({ label, value }) {
  const color = scoreColor(value ?? 0)
  return (
    <div className="card p-3">
      <div className="text-[11px] uppercase tracking-wider text-muted">{label}</div>
      <div className="mt-1 font-mono text-2xl" style={{ color }}>{(value ?? 0).toFixed(2)}</div>
      <div className="mt-2 h-1.5 rounded bg-panel2 overflow-hidden">
        <div
          className="h-full rounded"
          style={{ width: `${(value ?? 0) * 100}%`, background: color }}
        />
      </div>
    </div>
  )
}

function backendBadge(name, status) {
  const tone = status === "primary" ? "info" : status === "fallback" ? "watch" : "muted"
  const cls = {
    info:   "border-info/40 text-info bg-info/5",
    watch:  "border-watch/40 text-watch bg-watch/5",
    muted:  "border-line text-muted bg-panel2",
  }[tone]
  return <span className={`chip ${cls}`}>{name}: {status}</span>
}

function backendsFromResult(result) {
  // Inspect details across reels to identify which backend each component used.
  if (!result?.per_reel?.length) return []
  const r0 = result.per_reel[0]
  const out = []
  out.push(["separator", result.separator_backend === "spleeter" ? "primary" : "fallback"])
  out.push(["novelty", r0.novelty_details?.basis === "clap" || r0.novelty_details?.basis === "chromaprint" ? "primary" : "fallback"])
  out.push(["f0", r0.vocal_details?.f0_backend === "crepe" ? "primary" : "fallback"])
  out.push(["wpm", r0.vocal_details?.wpm_backend === "whisper" ? "primary" : "fallback"])
  out.push(["genre", r0.genre_details?.genre_backend === "essentia" ? "primary" : "fallback"])
  out.push(["bpm", r0.genre_details?.bpm_backend === "madmom" ? "primary" : "fallback"])
  return out
}

function ReelCard({ reel }) {
  const d = reel
  return (
    <div className="card p-4">
      <div className="flex items-baseline justify-between mb-2">
        <div className="text-xs text-muted font-mono">reel #{d.reel_index + 1}</div>
        <div className="font-mono text-lg" style={{ color: scoreColor(d.reel_score) }}>
          {d.reel_score.toFixed(2)}
        </div>
      </div>
      <div className="text-xs text-muted font-mono truncate mb-3">{d.filename || "—"}</div>

      <div className="grid grid-cols-3 gap-2 text-[11px] mb-3">
        <Pill label="novelty"    value={d.novelty}    z={d.z_scores?.novelty} />
        <Pill label="vocal edge" value={d.vocal_edge} z={d.z_scores?.vocal_edge} />
        <Pill label="genre edge" value={d.genre_edge} z={d.z_scores?.genre_edge} />
      </div>

      <div className="space-y-1.5 text-[11px] text-muted leading-snug">
        <div>
          <span className="text-text">Genre:</span> {d.genre_details?.genre_label} ·{" "}
          <span className="font-mono">BPM {d.genre_details?.bpm ?? "—"}</span>
        </div>
        <div>
          <span className="text-text">Vocals:</span>{" "}
          F0 <span className="font-mono">{d.vocal_details?.f0_mean_hz ?? "—"} Hz</span>
          {d.vocal_details?.f0_range_semis > 0 && (
            <> · range <span className="font-mono">{d.vocal_details.f0_range_semis}st</span></>
          )}
          {" · "}WPM <span className="font-mono">{d.vocal_details?.wpm ?? "—"}</span>
        </div>
        {d.vocal_details?.spectral_flatness != null && (
          <div>
            <span className="text-text">Vocal flatness:</span>{" "}
            <span className="font-mono">{d.vocal_details.spectral_flatness.toFixed(3)}</span>
            {d.vocal_details.f0_uptalk_semis != null && (
              <> · uptalk <span className="font-mono">{d.vocal_details.f0_uptalk_semis > 0 ? "+" : ""}{d.vocal_details.f0_uptalk_semis}st</span></>
            )}
          </div>
        )}
        <div>
          <span className="text-text">Novelty:</span>{" "}
          {d.novelty_details?.basis === "lite_hybrid" ? (
            <>canon-dist <span className="font-mono">{d.novelty_details.canonical_distance}</span> · diversity <span className="font-mono">{d.novelty_details.within_creator_diversity}</span></>
          ) : (
            <>basis {d.novelty_details?.basis}
              {d.novelty_details?.max_sim != null &&
                <> · max_sim <span className="font-mono">{d.novelty_details.max_sim.toFixed(2)}</span></>}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

function Pill({ label, value, z }) {
  const color = scoreColor(value)
  const zBadge = z != null && Math.abs(z) >= 1.0 ? (
    <span className={`ml-1 font-mono ${z > 0 ? "text-accent" : "text-muted"}`}>
      {z > 0 ? "↑" : "↓"}{Math.abs(z).toFixed(1)}σ
    </span>
  ) : null
  return (
    <div className="rounded bg-panel2 px-2 py-1 flex items-center justify-between">
      <span className="text-muted">{label}</span>
      <span className="font-mono flex items-center" style={{ color }}>
        {value.toFixed(2)}{zBadge}
      </span>
    </div>
  )
}

export default function AudioDashboard({ result }) {
  const backends = backendsFromResult(result)
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card p-5">
        <div className="flex flex-wrap items-baseline justify-between gap-4">
          <div>
            <div className="text-xs uppercase tracking-wider text-muted mb-1">Sonic Rebellion</div>
            <div className="text-3xl font-mono font-semibold" style={{ color: scoreColor(result.creator_score) }}>
              {result.creator_score.toFixed(2)}
            </div>
            <div className="text-sm mt-1">{result.score_label}</div>
          </div>
          <div className="text-xs text-muted text-right">
            <div>{result.n_reels_scored} reel{result.n_reels_scored === 1 ? "" : "s"} scored</div>
            <div className="font-mono">{result.wall_clock_s}s wall clock</div>
          </div>
        </div>
      </div>

      {/* Sub-score gauges */}
      <div className="grid grid-cols-3 gap-3">
        <Gauge label="Novelty mean"    value={result.sub_score_means?.novelty} />
        <Gauge label="Vocal edge mean" value={result.sub_score_means?.vocal_edge} />
        <Gauge label="Genre edge mean" value={result.sub_score_means?.genre_edge} />
      </div>

      {/* Backends used */}
      <div className="card p-4">
        <div className="text-[11px] uppercase tracking-wider text-muted mb-2">Backends</div>
        <div className="flex flex-wrap gap-2">
          {backends.map(([n, s]) => backendBadge(n, s))}
        </div>
        <div className="text-[11px] text-muted mt-2 leading-snug">
          Modules tagged <span className="text-watch">fallback</span> are running the librosa-based lite path.
          Install Spleeter, CLAP, CREPE, Whisper, Essentia, madmom to unlock the primary backends
          (see backend/requirements.txt and SONIC_REBELLION_SPEC.md).
        </div>
      </div>

      {/* Insights */}
      {result.insights?.length > 0 && (
        <div className="card p-4">
          <div className="text-[11px] uppercase tracking-wider text-muted mb-2">Insights</div>
          <ul className="space-y-1.5 text-sm">
            {result.insights.map((ins, i) => {
              const dot = ins.type === "good" ? "#3ddc97" : ins.type === "watch" ? "#ff7a59" : "#5aa9ff"
              return (
                <li key={i} className="flex items-start gap-2">
                  <span className="mt-1.5 h-1.5 w-1.5 rounded-full flex-shrink-0" style={{ background: dot }} />
                  <span className="leading-snug">{ins.text}</span>
                </li>
              )
            })}
          </ul>
        </div>
      )}

      {/* Within-creator dispersion */}
      {result.sub_score_stds && (
        <div className="card p-4">
          <div className="text-[11px] uppercase tracking-wider text-muted mb-2">
            Within-creator variation (σ across reels)
          </div>
          <div className="grid grid-cols-3 gap-3 text-xs">
            {Object.entries(result.sub_score_stds).map(([k, v]) => (
              <div key={k} className="flex items-center justify-between">
                <span className="text-muted">{k.replace(/_/g, " ")}</span>
                <span className="font-mono">σ = {v.toFixed(2)}</span>
              </div>
            ))}
          </div>
          <div className="text-[11px] text-muted mt-2 leading-snug">
            High σ = the creator varies that dimension across reels. Low σ = very consistent (or stuck on one approach).
          </div>
        </div>
      )}

      {/* Per-reel grid */}
      <div>
        <div className="text-sm font-medium mb-3">Per-reel breakdown</div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {result.per_reel?.map((r) => <ReelCard key={r.reel_index} reel={r} />)}
        </div>
      </div>

      {/* Weights footnote */}
      <div className="text-[11px] text-muted">
        Weights: novelty {result.weights?.novelty}, vocal {result.weights?.vocal_edge}, genre {result.weights?.genre_edge}.
        Composite = weighted mean per reel, then mean across reels.
      </div>
    </div>
  )
}
