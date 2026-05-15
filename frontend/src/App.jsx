import { useEffect, useRef, useState } from "react"
import Upload from "./components/Upload.jsx"
import Dashboard from "./components/Dashboard.jsx"
import ErrorBoundary from "./components/ErrorBoundary.jsx"
import { uploadVideo, pollResult, getResult } from "./api.js"

const JOB_KEY = "effortlessness:current_job"

function loadStoredJob() {
  try {
    const raw = localStorage.getItem(JOB_KEY)
    return raw ? JSON.parse(raw) : null
  } catch { return null }
}
function storeJob(jobId, filename) {
  try { localStorage.setItem(JOB_KEY, JSON.stringify({ jobId, filename })) } catch {}
}
function clearStoredJob() {
  try { localStorage.removeItem(JOB_KEY) } catch {}
}

export default function App() {
  const [result, setResult] = useState(null)
  const [videoUrl, setVideoUrl] = useState(null)
  const [progress, setProgress] = useState(null)        // {stage, progress, status}
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [resumingFromStorage, setResumingFromStorage] = useState(false)
  const videoUrlRef = useRef(null)

  // Auto-resume any in-flight or finished job after a page refresh.
  useEffect(() => {
    const stored = loadStoredJob()
    if (!stored?.jobId) return
    let alive = true
    setResumingFromStorage(true)
    setLoading(true)
    getResult(stored.jobId)
      .then(async (data) => {
        if (!alive) return
        if (data.status === "done") {
          setResult(data.result)
        } else if (data.status === "processing" || data.status === "queued") {
          setProgress({ stage: data.stage, progress: data.progress, status: data.status })
          const done = await pollResult(stored.jobId, (d) => {
            setProgress({ stage: d.stage, progress: d.progress, status: d.status })
          })
          if (alive) setResult(done.result)
        } else {
          // error / unknown — drop the stale id
          clearStoredJob()
        }
      })
      .catch(() => clearStoredJob())
      .finally(() => { if (alive) { setLoading(false); setResumingFromStorage(false) } })
    return () => { alive = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => () => {
    if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current)
  }, [])

  // Clear the stored job once we successfully have a result.
  useEffect(() => { if (result) clearStoredJob() }, [result])

  async function handleUpload(file) {
    setLoading(true); setError(null); setProgress(null); setResult(null)
    if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current)
    const url = URL.createObjectURL(file)
    videoUrlRef.current = url
    setVideoUrl(url)
    try {
      const { job_id } = await uploadVideo(file)
      storeJob(job_id, file.name)
      const done = await pollResult(job_id, (data) => {
        setProgress({ stage: data.stage, progress: data.progress, status: data.status })
      })
      setResult(done.result)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function loadByJobId(jobId) {
    setLoading(true); setError(null); setProgress(null); setResult(null)
    try {
      const data = await getResult(jobId)
      if (data.status === "processing" || data.status === "queued") {
        storeJob(jobId, "(loaded by id)")
        setProgress({ stage: data.stage, progress: data.progress, status: data.status })
        const done = await pollResult(jobId, (d) => {
          setProgress({ stage: d.stage, progress: d.progress, status: d.status })
        })
        setResult(done.result)
      } else if (data.status === "done") {
        setResult(data.result)
      } else {
        throw new Error(`Job status is "${data.status}"`)
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function reset() {
    clearStoredJob()
    if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current)
    videoUrlRef.current = null
    setVideoUrl(null); setResult(null); setError(null); setProgress(null)
  }

  return (
    <div className="min-h-screen">
      <header className="border-b border-line/60 backdrop-blur-md sticky top-0 z-30 bg-bg/70">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-accent to-info shadow-card" />
            <div>
              <div className="text-sm font-semibold tracking-tight">Effortlessness Analyzer</div>
              <div className="text-xs text-muted -mt-0.5">FACS · 5-method ensemble · per-person baseline</div>
            </div>
          </div>
          {result && (
            <button onClick={reset} className="text-sm text-muted hover:text-text transition">
              ← Analyse another video
            </button>
          )}
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {!result && (
          <>
            {resumingFromStorage && (
              <div className="max-w-2xl mx-auto mb-4 card p-3 border-info/40 text-info text-xs flex items-center gap-2">
                <span className="inline-block h-2 w-2 rounded-full bg-info animate-pulse" />
                Resuming your previous analysis — checking backend for the job we kicked off…
              </div>
            )}
            <Upload
              onUpload={handleUpload}
              loading={loading}
              error={error}
              progress={progress}
            />
            {!loading && <LoadByJobId onLoad={loadByJobId} />}
          </>
        )}
        {result && (
          <ErrorBoundary>
            <Dashboard result={result} videoUrl={videoUrl} />
          </ErrorBoundary>
        )}
      </main>

      <footer className="max-w-7xl mx-auto px-6 py-6 text-xs text-muted">
        py-feat · mediapipe · FastAPI · React/Vite — analysis is local, no data leaves your machine.
      </footer>
    </div>
  )
}

function LoadByJobId({ onLoad }) {
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
        >
          Load
        </button>
      </div>
    </div>
  )
}
