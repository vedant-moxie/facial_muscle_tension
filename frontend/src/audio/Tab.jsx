import { useEffect, useState } from "react"
import AudioUpload from "./components/AudioUpload.jsx"
import AudioDashboard from "./components/AudioDashboard.jsx"
import ErrorBoundary from "../shared/ErrorBoundary.jsx"
import { uploadAudioReels, pollAudioResult, getAudioResult } from "./api.js"

const AUDIO_JOB_KEY = "sonic_rebellion:current_job"

function load(key)  { try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : null } catch { return null } }
function save(key, val) { try { localStorage.setItem(key, JSON.stringify(val)) } catch {} }
function clear(key) { try { localStorage.removeItem(key) } catch {} }

export default function AudioTab() {
  const [result, setResult] = useState(null)
  const [progress, setProgress] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [resumingFromStorage, setResumingFromStorage] = useState(false)

  useEffect(() => {
    const stored = load(AUDIO_JOB_KEY)
    if (!stored?.jobId) return
    let alive = true
    setResumingFromStorage(true); setLoading(true)
    getAudioResult(stored.jobId)
      .then(async (data) => {
        if (!alive) return
        if (data.status === "done") setResult(data.result)
        else if (data.status === "processing" || data.status === "queued") {
          setProgress({ stage: data.stage, progress: data.progress, status: data.status })
          const done = await pollAudioResult(stored.jobId, (d) =>
            setProgress({ stage: d.stage, progress: d.progress, status: d.status }))
          if (alive) setResult(done.result)
        } else clear(AUDIO_JOB_KEY)
      })
      .catch(() => clear(AUDIO_JOB_KEY))
      .finally(() => { if (alive) { setLoading(false); setResumingFromStorage(false) } })
    return () => { alive = false }
  }, [])

  useEffect(() => { if (result) clear(AUDIO_JOB_KEY) }, [result])

  async function handleSubmit({ files, profileUrl, reelUrls }) {
    setLoading(true); setError(null); setProgress(null); setResult(null)
    try {
      const { job_id } = await uploadAudioReels({ files, profileUrl, reelUrls })
      save(AUDIO_JOB_KEY, { jobId: job_id })
      const done = await pollAudioResult(job_id, (data) =>
        setProgress({ stage: data.stage, progress: data.progress, status: data.status }))
      setResult(done.result)
    } catch (e) { setError(e.message) } finally { setLoading(false) }
  }

  function reset() {
    clear(AUDIO_JOB_KEY)
    setResult(null); setError(null); setProgress(null)
  }

  if (result) {
    return (
      <>
        <div className="mb-3 flex justify-end">
          <button onClick={reset} className="text-sm text-muted hover:text-text transition">
            ← Score another creator
          </button>
        </div>
        <ErrorBoundary>
          <AudioDashboard result={result} />
        </ErrorBoundary>
      </>
    )
  }

  return (
    <>
      {resumingFromStorage && (
        <div className="max-w-2xl mx-auto mb-4 card p-3 border-info/40 text-info text-xs flex items-center gap-2">
          <span className="inline-block h-2 w-2 rounded-full bg-info animate-pulse" />
          Resuming your previous sonic rebellion run…
        </div>
      )}
      <AudioUpload onSubmit={handleSubmit} loading={loading} error={error} progress={progress} />
    </>
  )
}
