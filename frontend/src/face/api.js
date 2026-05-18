const API_BASE = import.meta.env.VITE_API_BASE || ""

export async function uploadVideo(file) {
  const form = new FormData()
  form.append("file", file)
  const res = await fetch(`${API_BASE}/api/analyse`, { method: "POST", body: form })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `Upload failed (${res.status})`)
  }
  return res.json()
}

export async function getResult(jobId) {
  const res = await fetch(`${API_BASE}/api/result/${jobId}`)
  if (!res.ok) throw new Error(`Result fetch failed (${res.status})`)
  return res.json()
}

/**
 * Poll /api/result/{jobId} until the job is done or errors out.
 *
 * Resilience improvements over the previous version:
 *   • Default timeout is now 60 minutes (was 15 — long videos on CPU exceed
 *     that easily; the latest 49 s clip took 22 min in one user's run).
 *   • Transient network/server errors no longer kill the poll — they are
 *     retried up to 10 consecutive times with a short back-off before the
 *     promise is rejected.
 *   • Poll cadence widens after the first ~30 polls so we stop hammering
 *     the backend during the long AU-detection phase.
 */
export function pollResult(
  jobId,
  onProgress,
  {
    intervalMs = 2000,
    maxIntervalMs = 5000,
    timeoutMs = 60 * 60 * 1000,    // 60 min
    maxConsecutiveErrors = 10,
  } = {},
) {
  const start = Date.now()
  return new Promise((resolve, reject) => {
    let consecutiveErrors = 0
    let pollCount = 0

    const tick = async () => {
      try {
        const data = await getResult(jobId)
        consecutiveErrors = 0
        onProgress?.(data)
        if (data.status === "done") return resolve(data)
        if (data.status === "error") {
          return reject(new Error(data.message || "Pipeline error"))
        }
      } catch (e) {
        consecutiveErrors += 1
        if (consecutiveErrors >= maxConsecutiveErrors) {
          return reject(new Error(
            `Lost contact with backend after ${consecutiveErrors} attempts: ${e.message}`,
          ))
        }
        // swallow transient errors and keep trying
      }
      if (Date.now() - start > timeoutMs) {
        return reject(new Error(
          `Timed out after ${Math.round(timeoutMs / 60000)} min. The backend may still be ` +
          `running — try refreshing the page; we'll resume polling automatically.`,
        ))
      }
      pollCount += 1
      // Widen interval once the heavy AU detection phase is under way.
      const next = pollCount < 30 ? intervalMs : maxIntervalMs
      setTimeout(tick, next)
    }
    tick()
  })
}
