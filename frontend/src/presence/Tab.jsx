import { useEffect, useRef, useState } from "react"
import PresenceUpload from "./components/PresenceUpload.jsx"
import PresenceDashboard from "./components/PresenceDashboard.jsx"
import ErrorBoundary from "../shared/ErrorBoundary.jsx"
import { uploadPresenceVideo, pollPresenceResult, getPresenceResult } from "./api.js"

const PRESENCE_JOB_KEY = "presence:current_job"

function load(key)  { try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : null } catch { return null } }
function save(key, val) { try { localStorage.setItem(key, JSON.stringify(val)) } catch {} }
function clear(key) { try { localStorage.removeItem(key) } catch {} }

export default function PresenceTab() {
  const [result, setResult]                       = useState(null)
  const [videoUrl, setVideoUrl]                   = useState(null)
  const [progress, setProgress]                   = useState(null)
  const [loading, setLoading]                     = useState(false)
  const [error, setError]                         = useState(null)
  const [resumingFromStorage, setResumingFromStorage] = useState(false)
  const videoUrlRef = useRef(null)

  useEffect(() => {
    const stored = load(PRESENCE_JOB_KEY)
    if (!stored?.jobId) return
    let alive = true
    setResumingFromStorage(true); setLoading(true)
    getPresenceResult(stored.jobId)
      .then(async (data) => {
        if (!alive) return
        if (data.status === "done") setResult(data.result)
        else if (data.status === "processing" || data.status === "queued") {
          setProgress({ stage: data.stage, progress: data.progress, status: data.status })
          const done = await pollPresenceResult(stored.jobId, (d) =>
            setProgress({ stage: d.stage, progress: d.progress, status: d.status }))
          if (alive) setResult(done.result)
        } else clear(PRESENCE_JOB_KEY)
      })
      .catch(() => clear(PRESENCE_JOB_KEY))
      .finally(() => { if (alive) { setLoading(false); setResumingFromStorage(false) } })
    return () => { alive = false }
  }, [])

  useEffect(() => () => { if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current) }, [])
  useEffect(() => { if (result) clear(PRESENCE_JOB_KEY) }, [result])

  async function handleUpload(file) {
    setLoading(true); setError(null); setProgress(null); setResult(null)
    if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current)
    const url = URL.createObjectURL(file)
    videoUrlRef.current = url; setVideoUrl(url)
    try {
      const { job_id } = await uploadPresenceVideo(file)
      save(PRESENCE_JOB_KEY, { jobId: job_id, filename: file.name })
      const done = await pollPresenceResult(job_id, (data) =>
        setProgress({ stage: data.stage, progress: data.progress, status: data.status }))
      setResult(done.result)
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  function reset() {
    clear(PRESENCE_JOB_KEY)
    if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current)
    videoUrlRef.current = null
    setVideoUrl(null); setResult(null); setError(null); setProgress(null)
  }

  if (result) {
    return (
      <>
        <div className="mb-3 flex justify-end">
          <button onClick={reset} className="text-sm text-muted hover:text-text transition">
            ← Analyse another video
          </button>
        </div>
        <ErrorBoundary>
          <PresenceDashboard result={result} videoUrl={videoUrl} />
        </ErrorBoundary>
      </>
    )
  }

  return (
    <>
      {resumingFromStorage && (
        <div className="max-w-2xl mx-auto mb-4 card p-3 border-info/40 text-info text-xs flex items-center gap-2">
          <span className="inline-block h-2 w-2 rounded-full bg-info animate-pulse" />
          Resuming your previous presence run…
        </div>
      )}
      <PresenceUpload onUpload={handleUpload} loading={loading} error={error} progress={progress} />
    </>
  )
}
