import { useRef, useState } from "react"

/*
 * Sonic Rebellion — input UI.
 *
 * Three input modes, each its own pill-button row:
 *   1. Drop / pick multiple video files (up to 12)
 *   2. Paste an Instagram profile URL  → backend scrapes recent Reels
 *   3. Paste multiple Reel URLs        → backend downloads each
 */

const MODES = [
  { key: "files",   label: "Upload files",   hint: "Drop up to 12 reel MP4s" },
  { key: "profile", label: "Profile URL",    hint: "https://instagram.com/username/" },
  { key: "urls",    label: "Reel URLs",      hint: "One per line" },
]

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
        Heavy ML deps run in lite-mode on this install — see the panel below for which backends are active.
      </div>
    </div>
  )
}

export default function AudioUpload({ onSubmit, loading, error, progress }) {
  const [mode, setMode]           = useState("files")
  const [files, setFiles]         = useState([])
  const [profileUrl, setProfileUrl] = useState("")
  const [reelUrls, setReelUrls]     = useState("")
  const inputRef = useRef(null)

  function pickFiles(list) {
    if (!list || !list.length) return
    setFiles(Array.from(list).slice(0, 12))
  }

  function submit() {
    if (loading) return
    if (mode === "files") {
      if (!files.length) return
      onSubmit({ files })
    } else if (mode === "profile") {
      if (!profileUrl.trim()) return
      onSubmit({ profileUrl: profileUrl.trim() })
    } else {
      const urls = reelUrls.split(/[\n,]+/).map(s => s.trim()).filter(Boolean)
      if (!urls.length) return
      onSubmit({ reelUrls: urls })
    }
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="mb-4 text-center">
        <div className="text-xl font-semibold">Sonic Rebellion</div>
        <div className="text-xs text-muted mt-1">
          Score a creator's audio signature across 12 reels — novelty, vocal edge, genre edge.
        </div>
      </div>

      {/* Mode selector */}
      <div className="flex gap-2 mb-4">
        {MODES.map(m => (
          <button
            key={m.key}
            onClick={() => setMode(m.key)}
            className={[
              "flex-1 rounded-lg px-3 py-2 text-sm border transition",
              mode === m.key
                ? "bg-accent/15 border-accent/50 text-accent"
                : "bg-panel2 border-line text-muted hover:text-text",
            ].join(" ")}
          >
            {m.label}
          </button>
        ))}
      </div>

      <div className="card p-6">
        {mode === "files" && (
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => { e.preventDefault(); pickFiles(e.dataTransfer.files) }}
            onClick={() => !loading && inputRef.current?.click()}
            className={[
              "border-2 border-dashed rounded-xl p-10 text-center transition cursor-pointer",
              "border-line hover:border-accent/60",
              loading ? "opacity-90 pointer-events-none" : "",
            ].join(" ")}
          >
            <input
              ref={inputRef}
              type="file"
              accept="video/*"
              multiple
              className="hidden"
              onChange={(e) => pickFiles(e.target.files)}
            />
            <div className="mx-auto mb-3 h-12 w-12 rounded-2xl bg-gradient-to-br from-accent to-info opacity-80" />
            <div className="text-sm font-medium">
              {files.length ? `${files.length} file${files.length === 1 ? "" : "s"} selected` : "Drop reels here or click to pick"}
            </div>
            <div className="text-xs text-muted mt-1">MP4 / MOV / WebM / MKV · up to 12 reels</div>
            {files.length > 0 && (
              <div className="mt-3 text-xs font-mono text-muted truncate max-h-24 overflow-auto">
                {files.map(f => <div key={f.name}>{f.name}</div>)}
              </div>
            )}
          </div>
        )}

        {mode === "profile" && (
          <div className="space-y-2">
            <label className="text-xs text-muted">Instagram profile URL</label>
            <input
              type="text"
              value={profileUrl}
              onChange={(e) => setProfileUrl(e.target.value)}
              placeholder={MODES[1].hint}
              className="w-full bg-panel2 border border-line rounded-lg px-3 py-2.5 text-sm font-mono"
            />
            <div className="text-xs text-muted">
              Backend uses instaloader to fetch up to 12 recent reels. Works on public profiles;
              requires instaloader to be installed (otherwise you'll get a clear error message).
            </div>
          </div>
        )}

        {mode === "urls" && (
          <div className="space-y-2">
            <label className="text-xs text-muted">Reel URLs (one per line)</label>
            <textarea
              value={reelUrls}
              onChange={(e) => setReelUrls(e.target.value)}
              placeholder={`https://www.instagram.com/reel/AAAA/\nhttps://www.instagram.com/reel/BBBB/`}
              rows={6}
              className="w-full bg-panel2 border border-line rounded-lg px-3 py-2 text-sm font-mono"
            />
          </div>
        )}

        <button
          onClick={submit}
          disabled={loading}
          className={[
            "w-full mt-4 rounded-lg py-2.5 text-sm font-medium border transition",
            loading
              ? "bg-panel2 border-line text-muted cursor-not-allowed"
              : "bg-accent/15 border-accent/50 text-accent hover:bg-accent/25",
          ].join(" ")}
        >
          {loading ? "Analysing…" : "Score sonic rebellion"}
        </button>

        <ProgressBar progress={progress} />
      </div>

      {error && (
        <div className="mt-4 card p-4 border-watch/40 text-watch text-sm">
          {String(error)}
        </div>
      )}

      <div className="mt-8 grid grid-cols-3 gap-3 text-xs text-muted">
        {[
          ["Novelty",    "chromaprint + CLAP vs. trending pool (or MFCC fallback)"],
          ["Vocal edge", "F0, F0-variance, WPM, energy sigma vs. influencer norm"],
          ["Genre edge", "Discogs-400 genre + BPM vs. mainstream tempo band"],
        ].map(([t, d]) => (
          <div key={t} className="card p-3">
            <div className="text-text text-sm font-medium">{t}</div>
            <div className="mt-1 leading-snug">{d}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
