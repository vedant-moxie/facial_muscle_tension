import { useRef, useState } from "react"

function ProgressBar({ progress }) {
  if (!progress) return null
  const pct = Math.round((progress.progress ?? 0) * 100)
  return (
    <div className="mt-6 space-y-3">
      <div className="flex items-baseline justify-between">
        <div className="text-sm text-text">{progress.stage || "Starting…"}</div>
        <div className="font-mono text-xs text-muted">{pct}%</div>
      </div>
      <div className="h-1.5 w-full bg-line rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-accent to-info transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="text-xs text-muted">
        Cold start (first run) takes ~10 s for model download + JIT. Subsequent runs hit the pose cache.
      </div>
    </div>
  )
}

export default function PresenceUpload({ onUpload, loading, error, progress }) {
  const inputRef = useRef(null)
  const [drag, setDrag] = useState(false)
  const [file, setFile] = useState(null)

  function pickFile(f) {
    if (!f) return
    setFile(f)
    onUpload(f)
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div
        onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault(); setDrag(false)
          const f = e.dataTransfer.files?.[0]
          if (f) pickFile(f)
        }}
        className={[
          "card p-10 text-center transition-all",
          drag ? "border-accent ring-2 ring-accent/30" : "border-line",
          loading ? "opacity-90 pointer-events-none" : "cursor-pointer hover:border-accent/60",
        ].join(" ")}
        onClick={() => !loading && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/mp4,video/quicktime,video/webm,video/x-matroska,video/x-msvideo,.mp4,.mov,.webm,.mkv,.m4v,.avi"
          className="hidden"
          onChange={(e) => pickFile(e.target.files?.[0])}
        />
        <div className="mx-auto mb-4 h-12 w-12 rounded-2xl bg-gradient-to-br from-accent to-info opacity-80" />
        <div className="text-base font-medium">
          {loading ? "Analysing presence…" : "Drop a video here or click to choose"}
        </div>
        <div className="text-xs text-muted mt-1">
          MP4 / MOV / WebM · pose-based score · 60-sec video ≈ 8 s cold, ≤0.5 s warm
        </div>
        {file && (
          <div className="mt-4 text-xs font-mono text-muted truncate">{file.name}</div>
        )}
        <ProgressBar progress={progress} />
      </div>

      {error && (
        <div className="mt-4 card p-4 border-watch/40 text-watch text-sm">
          {String(error)}
        </div>
      )}

      <div className="mt-8 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs text-muted">
        {[
          ["Shoulder",  "symmetry — head-up posture"],
          ["Head",      "pitch deviation from baseline"],
          ["Arms",      "gesture variance + amplitude"],
          ["Fluidity",  "SPARC smoothness of motion"],
        ].map(([title, sub]) => (
          <div key={title} className="card p-3">
            <div className="text-text text-sm font-medium">{title}</div>
            <div className="mt-1 leading-snug">{sub}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
