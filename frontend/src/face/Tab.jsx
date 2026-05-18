import { useEffect, useRef, useState } from "react"
import Upload from "./components/Upload.jsx"
import Dashboard from "./components/Dashboard.jsx"
import ErrorBoundary from "../shared/ErrorBoundary.jsx"
import { uploadVideo, pollResult, getResult } from "./api.js"

const JOB_KEY = "effortlessness:current_job"

function load(key)  { try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : null } catch { return null } }
function save(key, val) { try { localStorage.setItem(key, JSON.stringify(val)) } catch {} }
function clear(key) { try { localStorage.removeItem(key) } catch {} }

export default function FaceTab() {
  const [result, setResult] = useState(null)
  const [videoUrl, setVideoUrl] = useState(null)
  const [progress, setProgress] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [resumingFromStorage, setResumingFromStorage] = useState(false)
  const videoUrlRef = useRef(null)

  useEffect(() => {
    const stored = load(JOB_KEY)
    if (!stored?.jobId) return
    let alive = true
    setResumingFromStorage(true); setLoading(true)
    getResult(stored.jobId)
      .then(async (data) => {
        if (!alive) return
        if (data.status === "done") setResult(data.result)
        else if (data.status === "processing" || data.status === "queued") {
          setProgress({ stage: data.stage, progress: data.progress, status: data.status })
          const done = await pollResult(stored.jobId, (d) =>
            setProgress({ stage: d.stage, progress: d.progress, status: d.status }))
          if (alive) setResult(done.result)
        } else clear(JOB_KEY)
      })
      .catch(() => clear(JOB_KEY))
      .finally(() => { if (alive) { setLoading(false); setResumingFromStorage(false) } })
    return () => { alive = false }
  }, [])

  useEffect(() => () => { if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current) }, [])
  useEffect(() => { if (result) clear(JOB_KEY) }, [result])

  async function handleUpload(file) {
    setLoading(true); setError(null); setProgress(null); setResult(null)
    if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current)
    const url = URL.createObjectURL(file)
    videoUrlRef.current = url; setVideoUrl(url)
    try {
      const { job_id } = await uploadVideo(file)
      save(JOB_KEY, { jobId: job_id, filename: file.name })
      const done = await pollResult(job_id, (data) =>
        setProgress({ stage: data.stage, progress: data.progress, status: data.status }))
      setResult(done.result)
    } catch (e) {
      setError(e.message)
    } finally { setLoading(false) }
  }

  async function loadByJobId(jobId) {
    setLoading(true); setError(null); setProgress(null); setResult(null)
    try {
      const data = await getResult(jobId)
      if (data.status === "processing" || data.status === "queued") {
        save(JOB_KEY, { jobId, filename: "(loaded by id)" })
        setProgress({ stage: data.stage, progress: data.progress, status: data.status })
        const done = await pollResult(jobId, (d) =>
          setProgress({ stage: d.stage, progress: d.progress, status: d.status }))
        setResult(done.result)
      } else if (data.status === "done") setResult(data.result)
      else throw new Error(`Job status is "${data.status}"`)
    } catch (e) { setError(e.message) } finally { setLoading(false) }
  }

  function reset() {
    clear(JOB_KEY)
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
          <Dashboard result={result} videoUrl={videoUrl} />
        </ErrorBoundary>
      </>
    )
  }

  return (
    <>
      {resumingFromStorage && (
        <div className="max-w-2xl mx-auto mb-4 card p-3 border-info/40 text-info text-xs flex items-center gap-2">
          <span className="inline-block h-2 w-2 rounded-full bg-info animate-pulse" />
          Resuming your previous face analysis…
        </div>
      )}
      <Upload onUpload={handleUpload} loading={loading} error={error} progress={progress} />
      {!loading && <LoadFaceByJobId onLoad={loadByJobId} />}
    </>
  )
}

function LoadFaceByJobId({ onLoad }) {
  const [id, setId] = useState("")
  return (
    <div className="max-w-2xl mx-auto mt-6 card p-4">
      <div className="text-xs uppercase tracking-wider text-muted mb-2">Already analysed?</div>
      <div className="flex gap-2">
        <input
          value={id}
          onChange={(e) => setId(e.target.value.trim())}
          placeholder="paste job_id from backend logs (uuid)"
          className="flex-1 bg-panel2 border border-line rounded-lg px-3 py-2 text-sm font-mono"
        />
        <button
          onClick={() => id && onLoad(id)}
          className="px-4 py-2 text-sm rounded-lg bg-accent/15 text-accent border border-accent/40 hover:bg-accent/25 transition"
        >Load</button>
      </div>
    </div>
  )
}
