import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react"

/*
 * Live face-analysis overlay.
 *
 * Fetches /api/visualization/{jobId}, plays the user's uploaded video, and
 * draws everything the pipeline saw — landmarks, eye ring with EAR, head-pose
 * cross, blink flash, scoreable status — on a canvas pinned over the <video>.
 * The right rail shows the same metrics as live numeric / bar readouts.
 *
 * Parent components seek the video via a ref (forwardRef → seekTo).
 */

const COLORS = {
  accent:   "#a78bfa",
  info:     "#5aa9ff",
  good:     "#3ddc97",
  watch:    "#ff7a59",
  muted:    "#8a90a4",
  red:      "#ef4444",
  white:    "#ffffff",
}

// Tension AUs (red) vs positive AUs (green) for the per-frame bar chart.
const TENSION_AUS  = new Set(["AU04", "AU07", "AU10", "AU14", "AU17", "AU43"])
const POSITIVE_AUS = new Set(["AU06", "AU12", "AU25"])

function findNearestFrame(frames, t) {
  if (!frames || frames.length === 0) return -1
  let lo = 0, hi = frames.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (frames[mid].t < t) lo = mid + 1
    else hi = mid
  }
  // pick whichever neighbour is closer
  if (lo > 0 && Math.abs(frames[lo - 1].t - t) < Math.abs(frames[lo].t - t)) return lo - 1
  return lo
}

/**
 * When the tracker briefly loses the face (motion blur, partial occlusion),
 * the visualization frame has `present: false` and empty landmarks. Rather
 * than blanking the overlay on every such frame and looking jittery, we
 * carry the last good frame's landmarks forward — but only for a short
 * window (default 6 frames ≈ 0.5 s @ 12 fps) so we don't draw stale data on
 * a real face-absent gap. This does NOT affect scoring: backend already
 * marked those frames un-scoreable and dropped them.
 */
function withCarriedLandmarks(frames, maxCarryFrames = 6) {
  if (!Array.isArray(frames)) return frames
  let lastGood = null
  let gap = 0
  const out = new Array(frames.length)
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i]
    if (f.present && f.landmarks?.length) {
      lastGood = f
      gap = 0
      out[i] = f
    } else if (lastGood && gap < maxCarryFrames) {
      gap += 1
      // Shallow-clone so we don't mutate the original frame.
      out[i] = { ...f, landmarks: lastGood.landmarks, _carried: gap }
    } else {
      out[i] = f
    }
  }
  return out
}

function drawOverlay(ctx, w, h, frame, groups) {
  ctx.clearRect(0, 0, w, h)
  if (!frame) return

  // Top status badge
  const isCarried = !!frame._carried
  drawBadge(
    ctx,
    isCarried
      ? `CARRIED · last good +${frame._carried}f`
      : frame.present
      ? (frame.scoreable ? "TRACKED" : "TRACKED · OFF-ANGLE")
      : "NO FACE",
    isCarried
      ? COLORS.muted
      : frame.present ? (frame.scoreable ? COLORS.good : COLORS.watch) : COLORS.red,
    12, 12,
  )

  if (!frame.present || !frame.landmarks?.length) return
  const lm = frame.landmarks

  // 1) All landmark points (faint)
  ctx.fillStyle = "rgba(167, 139, 250, 0.65)"
  for (const [x, y] of lm) {
    ctx.beginPath()
    ctx.arc(x * w, y * h, 1.8, 0, 2 * Math.PI)
    ctx.fill()
  }

  // 2) Group polylines
  connect(ctx, lm, groups.left_eye,   frame.blink ? COLORS.red : COLORS.info, w, h, true)
  connect(ctx, lm, groups.right_eye,  frame.blink ? COLORS.red : COLORS.info, w, h, true)
  connect(ctx, lm, groups.left_brow,  asymColor(frame.asym.brow),  w, h, false)
  connect(ctx, lm, groups.right_brow, asymColor(frame.asym.brow),  w, h, false)
  connect(ctx, lm, groups.mouth,      asymColor(frame.asym.mouth), w, h, true)

  // 3) EAR readouts next to each eye
  const lc = centroid(lm, groups.left_eye)
  const rc = centroid(lm, groups.right_eye)
  drawText(ctx, `EAR ${frame.ear_l.toFixed(2)}`, lc[0] * w + 16, lc[1] * h - 4, COLORS.info)
  drawText(ctx, `EAR ${frame.ear_r.toFixed(2)}`, rc[0] * w - 80, rc[1] * h - 4, COLORS.info)

  // 4) Head-pose axes anchored at nose tip
  const nose = lm[groups.nose[0]]
  drawPoseAxes(ctx, nose[0] * w, nose[1] * h, frame.pitch, frame.yaw, frame.roll)

  // 5) Tension red wash overlay if frame tension is high
  if (frame.tension != null && frame.tension > 1.0) {
    const alpha = Math.min(0.25, (frame.tension - 1.0) / 4)
    ctx.fillStyle = `rgba(239, 68, 68, ${alpha})`
    ctx.fillRect(0, 0, w, h)
  }
}

function connect(ctx, lm, idxs, color, w, h, close) {
  if (!idxs || idxs.length < 2) return
  ctx.strokeStyle = color
  ctx.lineWidth = 1.6
  ctx.beginPath()
  idxs.forEach((i, k) => {
    const p = lm[i]; if (!p) return
    const x = p[0] * w, y = p[1] * h
    if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y)
  })
  if (close) ctx.closePath()
  ctx.stroke()
}

function centroid(lm, idxs) {
  let sx = 0, sy = 0, n = 0
  for (const i of idxs) {
    const p = lm[i]; if (!p) continue
    sx += p[0]; sy += p[1]; n++
  }
  return n ? [sx / n, sy / n] : [0, 0]
}

function asymColor(v) {
  if (v < 0.08) return COLORS.watch    // over-symmetric (held)
  if (v <= 0.30) return COLORS.good    // genuine range
  return COLORS.watch                  // high asymmetry
}

function drawPoseAxes(ctx, ox, oy, pitch, yaw, roll) {
  // Quick-and-cheerful: draw 3 small axes whose orientation reflects yaw/pitch/roll.
  const L = 36
  const pRad = (pitch * Math.PI) / 180
  const yRad = (yaw   * Math.PI) / 180
  const rRad = (roll  * Math.PI) / 180
  const cosR = Math.cos(rRad), sinR = Math.sin(rRad)

  function arrow(dx, dy, color) {
    const rx = dx * cosR - dy * sinR
    const ry = dx * sinR + dy * cosR
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(ox, oy)
    ctx.lineTo(ox + rx, oy + ry)
    ctx.stroke()
  }
  // x: yaw → horizontal length scaled by cos(yaw)
  arrow(L * Math.cos(yRad), 0, COLORS.info)
  // y: pitch → vertical length scaled by cos(pitch)
  arrow(0, -L * Math.cos(pRad), COLORS.good)
  // z: into screen — small dot
  ctx.fillStyle = COLORS.accent
  ctx.beginPath()
  ctx.arc(ox, oy, 3, 0, Math.PI * 2)
  ctx.fill()
}

function drawBadge(ctx, text, color, x, y) {
  ctx.font = "600 11px ui-monospace, SFMono-Regular, Menlo, monospace"
  const padX = 8, padY = 5
  const m = ctx.measureText(text)
  const wBox = m.width + padX * 2, hBox = 22
  ctx.fillStyle = "rgba(11, 12, 16, 0.7)"
  roundRect(ctx, x, y, wBox, hBox, 6)
  ctx.fill()
  ctx.strokeStyle = color; ctx.lineWidth = 1
  roundRect(ctx, x, y, wBox, hBox, 6); ctx.stroke()
  ctx.fillStyle = color
  ctx.fillText(text, x + padX, y + hBox - padY - 2)
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.lineTo(x + w - r, y); ctx.arcTo(x + w, y, x + w, y + r, r)
  ctx.lineTo(x + w, y + h - r); ctx.arcTo(x + w, y + h, x + w - r, y + h, r)
  ctx.lineTo(x + r, y + h); ctx.arcTo(x, y + h, x, y + h - r, r)
  ctx.lineTo(x, y + r); ctx.arcTo(x, y, x + r, y, r)
  ctx.closePath()
}

function drawText(ctx, text, x, y, color) {
  ctx.font = "600 11px ui-monospace, SFMono-Regular, Menlo, monospace"
  ctx.fillStyle = "rgba(11,12,16,0.55)"
  const m = ctx.measureText(text)
  ctx.fillRect(x - 3, y - 11, m.width + 6, 14)
  ctx.fillStyle = color
  ctx.fillText(text, x, y)
}

function topActiveAUs(aus, n = 5) {
  if (!aus) return []
  return Object.entries(aus)
    .filter(([, v]) => v > 0.05)
    .sort((a, b) => b[1] - a[1])
    .slice(0, n)
}

function auTone(name) {
  if (TENSION_AUS.has(name)) return COLORS.watch
  if (POSITIVE_AUS.has(name)) return COLORS.good
  return COLORS.info
}

// ---------------------------------------------------------------------------

const SimulationPanel = forwardRef(function SimulationPanel(
  { jobId, videoUrl, enabled = true },
  ref,
) {
  const videoRef  = useRef(null)
  const canvasRef = useRef(null)
  const wrapRef   = useRef(null)
  const rafRef    = useRef(0)
  const [viz, setViz]       = useState(null)
  const [vizErr, setVizErr] = useState(null)
  const [overlay, setOverlay] = useState(true)
  const [frame, setFrame] = useState(null)
  const [videoFallback, setVideoFallback] = useState(false)

  // If no local blob URL is provided (e.g. dashboard was loaded by job_id
  // after a page refresh), fall back to the backend-served copy.
  const effectiveVideoUrl = videoUrl || (jobId ? `/api/video/${jobId}` : null)

  useImperativeHandle(ref, () => ({
    seekTo(seconds) {
      const v = videoRef.current
      if (!v) return
      v.currentTime = seconds
      v.play().catch(() => {})
    },
  }), [])

  // ---- fetch the visualization payload once ----
  useEffect(() => {
    if (!jobId) return
    let alive = true
    fetch(`/api/visualization/${jobId}`)
      .then((r) => {
        if (!r.ok) throw new Error(`viz HTTP ${r.status}`)
        return r.json()
      })
      .then((v) => {
        if (!alive) return
        // Smooth visual gaps: carry landmarks forward for up to ~0.5 s.
        // Scoring is untouched — this is presentational only.
        v.frames = withCarriedLandmarks(v.frames, 6)
        setViz(v)
      })
      .catch((e) => { if (alive) setVizErr(e.message) })
    return () => { alive = false }
  }, [jobId])

  // ---- size canvas to match the rendered video element ----
  useEffect(() => {
    const v = videoRef.current; const c = canvasRef.current
    if (!v || !c) return
    function resize() {
      const rect = v.getBoundingClientRect()
      const dpr = window.devicePixelRatio || 1
      c.width  = Math.max(1, Math.round(rect.width  * dpr))
      c.height = Math.max(1, Math.round(rect.height * dpr))
      c.style.width  = `${rect.width}px`
      c.style.height = `${rect.height}px`
      const ctx = c.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(v)
    v.addEventListener("loadedmetadata", resize)
    return () => { ro.disconnect(); v.removeEventListener("loadedmetadata", resize) }
  }, [videoUrl])

  // ---- animation loop ----
  useEffect(() => {
    if (!viz || !overlay || !enabled) {
      // clear and stop
      const c = canvasRef.current
      if (c) c.getContext("2d").clearRect(0, 0, c.width, c.height)
      cancelAnimationFrame(rafRef.current)
      return
    }
    const v = videoRef.current; const c = canvasRef.current
    if (!v || !c) return
    const ctx = c.getContext("2d")
    const groups = viz.landmark_groups

    function tick() {
      const t = v.currentTime
      const i = findNearestFrame(viz.frames, t)
      const f = viz.frames[i]
      setFrame(f)
      drawOverlay(ctx, c.clientWidth, c.clientHeight, f, groups)
      rafRef.current = requestAnimationFrame(tick)
    }
    tick()
    return () => cancelAnimationFrame(rafRef.current)
  }, [viz, overlay, enabled])

  const tops = useMemo(() => topActiveAUs(frame?.aus, 5), [frame])

  const overlayAvailable = !!viz && !vizErr
  const status = vizErr
    ? `overlay data unavailable for this job — re-run analysis to enable simulation`
    : !viz
    ? "loading overlay data…"
    : `overlay ready · ${viz.frames.length} frames @ ${viz.fps} fps`

  return (
    <div className="card p-4 flex flex-col gap-4">
      {/* ---- header ---- */}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex-1 min-w-[200px]">
          <div className="flex items-center gap-2">
            <span className="inline-block h-2 w-2 rounded-full bg-accent animate-pulse" />
            <div className="text-base font-semibold">Live face analysis</div>
          </div>
          <div className="text-xs text-muted mt-1">
            Frame-by-frame overlay: mediapipe landmarks · solvePnP head pose · py-feat AU intensities
          </div>
          <div className="text-[11px] text-muted mt-1 font-mono">{status}</div>
        </div>

        {/* Big, obvious toggle */}
        <button
          type="button"
          onClick={() => setOverlay((o) => !o)}
          disabled={!overlayAvailable}
          className={[
            "inline-flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium border transition",
            !overlayAvailable
              ? "bg-panel2 border-line text-muted cursor-not-allowed"
              : overlay
              ? "bg-accent/15 border-accent/50 text-accent hover:bg-accent/25"
              : "bg-panel2 border-line text-muted hover:text-text",
          ].join(" ")}
        >
          <span
            className={[
              "h-2 w-2 rounded-full",
              overlay && overlayAvailable ? "bg-accent" : "bg-muted",
            ].join(" ")}
          />
          Overlay {overlay && overlayAvailable ? "ON" : "OFF"}
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        {/* Video + canvas */}
        <div ref={wrapRef} className="lg:col-span-2 relative rounded-xl overflow-hidden bg-black">
          {effectiveVideoUrl && !videoFallback ? (
            <video
              ref={videoRef}
              src={effectiveVideoUrl}
              controls
              onError={() => setVideoFallback(true)}
              className="block w-full max-h-[500px] object-contain bg-black"
            />
          ) : (
            <div className="flex flex-col items-center justify-center h-[300px] text-muted text-sm gap-2 px-6 text-center">
              <div>Video preview unavailable</div>
              <div className="text-[11px] max-w-md">
                This job was analysed before video-retention was enabled, or the file has been cleared.
                Re-upload the clip to play it back with the overlay.
              </div>
            </div>
          )}
          <canvas
            ref={canvasRef}
            className="absolute inset-0 pointer-events-none"
            style={{ width: "100%", height: "100%" }}
          />
          {vizErr && (
            <div className="absolute bottom-2 left-2 px-2 py-1 rounded bg-watch/20 border border-watch/40 text-watch text-[11px]">
              {vizErr}
            </div>
          )}
          {!viz && !vizErr && (
            <div className="absolute bottom-2 left-2 px-2 py-1 rounded bg-bg/60 border border-line text-muted text-[11px]">
              loading overlay…
            </div>
          )}
        </div>

        {/* Live metrics rail */}
        <div className="lg:col-span-1 space-y-3">
          <LiveBadges frame={frame} />
          <LivePose frame={frame} />
          <LiveAsym frame={frame} />
          <LiveAUs aus={tops} />
          <LiveEffort frame={frame} />
        </div>
      </div>

      {/* ---- legend ---- */}
      {overlayAvailable && (
        <div className="border-t border-line/60 pt-3">
          <div className="text-[11px] uppercase tracking-wider text-muted mb-2">
            What you're looking at
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-2 text-[11px]">
            <LegendItem dot="#a78bfa" label="26 tracked landmarks (eyes, brows, mouth, nose, chin, cheeks)" />
            <LegendItem dot="#5aa9ff" label="Eye ring with EAR readout — turns red on blink" />
            <LegendItem dot="#3ddc97" label="Mouth/brow contours green when asymmetry is in genuine 0.08–0.30 band" />
            <LegendItem dot="#ff7a59" label="Orange contours = held / over-asymmetric expression" />
            <LegendItem dot="#3ddc97" small label="Vertical cross arm = pitch · horizontal = yaw · whole cross rotates with roll" />
            <LegendItem dot="#ef4444" small label="Red wash over frame = high per-frame tension AU activation" />
            <LegendItem dot="#3ddc97" small label="TRACKED badge = frame counted toward scores" />
            <LegendItem dot="#ff7a59" small label="OFF-ANGLE badge = pose gate failed, frame skipped" />
          </div>
        </div>
      )}
    </div>
  )
})

function LegendItem({ dot, label, small }) {
  return (
    <div className="flex items-start gap-2">
      <span
        className={`mt-1 inline-block rounded-full ${small ? "h-1.5 w-1.5" : "h-2 w-2"}`}
        style={{ background: dot }}
      />
      <span className="text-muted leading-snug">{label}</span>
    </div>
  )
}

export default SimulationPanel

// --- Live readouts ----------------------------------------------------------

function LiveBadges({ frame }) {
  if (!frame) return <Card title="Status"><div className="text-xs text-muted">—</div></Card>
  const carried = !!frame._carried
  const tone = carried
    ? "watch"
    : frame.present ? (frame.scoreable ? "good" : "watch") : "watch"
  const label = carried
    ? `carried +${frame._carried}f`
    : frame.present ? (frame.scoreable ? "tracked" : "off-angle") : "no face"
  return (
    <Card title="Status">
      <div className="flex flex-wrap gap-2 text-[11px]">
        <Chip color={tone}>{label}</Chip>
        <Chip color={frame.blink ? "watch" : "info"}>blink {frame.blink ? "yes" : "no"}</Chip>
        <Chip color="info">t {frame.t.toFixed(2)}s</Chip>
      </div>
    </Card>
  )
}

function LivePose({ frame }) {
  if (!frame) return null
  return (
    <Card title="Head pose">
      <Row label="pitch" value={`${frame.pitch.toFixed(1)}°`} />
      <Row label="yaw"   value={`${frame.yaw.toFixed(1)}°`} />
      <Row label="roll"  value={`${frame.roll.toFixed(1)}°`} />
    </Card>
  )
}

function LiveAsym({ frame }) {
  if (!frame) return null
  const probes = [
    ["brow",     frame.asym.brow],
    ["cheek",    frame.asym.cheek],
    ["mouth",    frame.asym.mouth],
    ["eye open", frame.asym.eye_open],
  ]
  return (
    <Card title="Asymmetry probes (L vs R)">
      <div className="space-y-1.5">
        {probes.map(([k, v]) => (
          <div key={k} className="flex items-center gap-2 text-[11px]">
            <div className="w-16 text-muted">{k}</div>
            <div className="flex-1 h-2 rounded bg-panel2 overflow-hidden">
              <div
                className="h-full rounded"
                style={{
                  width: `${Math.min(100, v * 200)}%`,
                  background: v < 0.08 ? "#ff7a59" : v <= 0.30 ? "#3ddc97" : "#ff7a59",
                }}
              />
            </div>
            <div className="font-mono w-10 text-right">{v.toFixed(2)}</div>
          </div>
        ))}
      </div>
    </Card>
  )
}

function LiveAUs({ aus }) {
  return (
    <Card title="Active AUs (this frame)">
      {(!aus || aus.length === 0) && (
        <div className="text-xs text-muted">no AU above 0.05</div>
      )}
      <div className="space-y-1.5">
        {aus.map(([name, v]) => (
          <div key={name} className="flex items-center gap-2 text-[11px]">
            <div className="font-mono w-14">{name}</div>
            <div className="flex-1 h-2 rounded bg-panel2 overflow-hidden">
              <div
                className="h-full rounded"
                style={{ width: `${Math.min(100, (v / 3) * 100)}%`, background: auTone(name) }}
              />
            </div>
            <div className="font-mono w-10 text-right">{v.toFixed(2)}</div>
          </div>
        ))}
      </div>
    </Card>
  )
}

function LiveEffort({ frame }) {
  if (!frame || frame.effortless == null) return null
  const v = frame.effortless
  const color = v >= 0.65 ? "#3ddc97" : v >= 0.5 ? "#5aa9ff" : "#ff7a59"
  return (
    <Card title="Frame effortlessness (tension AUs)">
      <div className="flex items-center gap-3">
        <div className="font-mono text-2xl" style={{ color }}>{v.toFixed(2)}</div>
        <div className="flex-1 h-2 rounded bg-panel2 overflow-hidden">
          <div className="h-full rounded" style={{ width: `${v * 100}%`, background: color }} />
        </div>
      </div>
      {frame.tension != null && (
        <div className="text-[11px] text-muted mt-1 font-mono">raw tension {frame.tension.toFixed(2)}</div>
      )}
    </Card>
  )
}

function Card({ title, children }) {
  return (
    <div className="card p-3">
      <div className="text-[11px] uppercase tracking-wider text-muted mb-2">{title}</div>
      {children}
    </div>
  )
}

function Row({ label, value }) {
  return (
    <div className="flex items-center justify-between text-[11px] py-0.5">
      <div className="text-muted">{label}</div>
      <div className="font-mono">{value}</div>
    </div>
  )
}

function Chip({ color, children }) {
  const map = {
    good:  "border-good/40 text-good bg-good/5",
    watch: "border-watch/40 text-watch bg-watch/5",
    info:  "border-info/40 text-info bg-info/5",
  }
  return (
    <span className={`chip ${map[color] || map.info}`}>{children}</span>
  )
}
