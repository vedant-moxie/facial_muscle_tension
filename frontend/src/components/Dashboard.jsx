import { useRef, useState } from "react"
import Timeline from "./Timeline.jsx"
import AUBarChart from "./AUBarChart.jsx"
import MethodScorecard from "./MethodScorecard.jsx"
import InsightList from "./InsightList.jsx"
import StatGrid from "./StatGrid.jsx"
import SimulationPanel from "./SimulationPanel.jsx"
import ConfidenceCard from "./ConfidenceCard.jsx"

function scoreColor(s) {
  if (s >= 0.8) return "text-good"
  if (s >= 0.65) return "text-good/80"
  if (s >= 0.5) return "text-info"
  if (s >= 0.35) return "text-watch"
  return "text-watch"
}

export default function Dashboard({ result, videoUrl }) {
  const {
    job_id,
    filename, duration_secs, frames_analysed, fps,
    overall_score, overall_label,
    method_scores, timeline, au_means,
    micro_expression_events, blink_rate_per_min, asymmetry_mean,
    insights, baseline, warnings = [],
  } = result

  const simRef = useRef(null)
  const [seekedTo, setSeekedTo] = useState(null)

  function seekTo(seconds) {
    if (simRef.current) {
      simRef.current.seekTo(seconds)
      setSeekedTo(seconds)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card p-5 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="text-xs uppercase tracking-wider text-muted mb-1">Source</div>
          <div className="font-mono text-sm truncate max-w-[480px]">{filename || "video"}</div>
          <div className="text-xs text-muted mt-1">
            {duration_secs?.toFixed?.(1)} s · {frames_analysed} frames @ {fps} fps
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {warnings.map((w, i) => (
            <span key={i} className="chip border-watch/40 text-watch bg-watch/5">{w}</span>
          ))}
          {baseline && (
            <span className="chip border-info/40 text-info bg-info/5">
              Baseline · {baseline.calibration_frames} frames · {baseline.blink_rate_per_min?.toFixed?.(1)} bpm
            </span>
          )}
        </div>
      </div>

      {/* Score + video */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="card p-6 lg:col-span-1">
          <div className="text-xs uppercase tracking-wider text-muted">Overall</div>
          <div className={`mt-2 font-mono text-6xl font-semibold ${scoreColor(overall_score)}`}>
            {overall_score.toFixed(2)}
          </div>
          <div className="mt-1 text-sm">{overall_label}</div>
          <div className="mt-6 grid grid-cols-1 gap-2">
            {Object.entries(method_scores).map(([name, s]) => (
              <MethodScorecard key={name} name={name} summary={s} />
            ))}
          </div>
        </div>

        <div className="lg:col-span-2">
          <SimulationPanel
            ref={simRef}
            jobId={job_id}
            videoUrl={videoUrl}
          />
          {seekedTo !== null && (
            <div className="text-xs text-muted mt-2 font-mono">
              ↳ jumped to {Math.floor(seekedTo / 60)}:{String(Math.floor(seekedTo % 60)).padStart(2, "0")}
            </div>
          )}
        </div>
      </div>

      {/* Confidence */}
      <ConfidenceCard result={result} />

      {/* Timeline */}
      <div className="card p-5">
        <div className="flex items-baseline justify-between mb-4">
          <div>
            <div className="text-sm font-medium">Effortlessness timeline</div>
            <div className="text-xs text-muted">click a micro-expression marker to jump the video</div>
          </div>
          <div className="text-xs text-muted">y = 0 (strain) … 1 (effortless)</div>
        </div>
        <Timeline
          timeline={timeline}
          events={micro_expression_events}
          onSeek={seekTo}
        />
      </div>

      {/* AU bars + stats */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="card p-5 lg:col-span-2">
          <div className="text-sm font-medium mb-3">Action-Unit intensity (mean over video)</div>
          <AUBarChart auMeans={au_means} />
        </div>
        <div className="lg:col-span-1">
          <StatGrid
            frames={frames_analysed}
            events={micro_expression_events?.length || 0}
            blinkRate={blink_rate_per_min}
            asym={asymmetry_mean}
          />
        </div>
      </div>

      {/* Insights */}
      <div className="card p-5">
        <div className="text-sm font-medium mb-3">Key insights</div>
        <InsightList insights={insights} />
      </div>

      {/* Micro-expression events table */}
      {micro_expression_events?.length > 0 && (
        <div className="card p-5">
          <div className="text-sm font-medium mb-3">Micro-expression events ({micro_expression_events.length})</div>
          <div className="overflow-auto">
            <table className="w-full text-sm">
              <thead className="text-xs text-muted">
                <tr>
                  <th className="text-left py-2 font-medium">Time</th>
                  <th className="text-left py-2 font-medium">Duration</th>
                  <th className="text-left py-2 font-medium">Dominant AU</th>
                  <th className="text-left py-2 font-medium">Intensity</th>
                  <th className="text-right py-2 font-medium"></th>
                </tr>
              </thead>
              <tbody>
                {micro_expression_events.map((e, i) => (
                  <tr key={i} className="border-t border-line/60">
                    <td className="py-2 font-mono">{e.timestamp_str}</td>
                    <td className="py-2 font-mono text-muted">{e.duration_ms} ms</td>
                    <td className="py-2 font-mono">{e.dominant_au}</td>
                    <td className="py-2 font-mono text-muted">{e.intensity}</td>
                    <td className="py-2 text-right">
                      <button
                        onClick={() => seekTo(e.timestamp)}
                        className="text-xs text-info hover:text-text underline-offset-2 hover:underline"
                      >jump →</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
