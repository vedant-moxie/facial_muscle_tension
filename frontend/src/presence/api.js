const API_BASE = import.meta.env.VITE_API_BASE || ""

export async function uploadPresenceVideo(file) {
  const form = new FormData()
  form.append("file", file)
  const res = await fetch(`${API_BASE}/api/presence/analyse`, { method: "POST", body: form })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `Upload failed (${res.status})`)
  }
  return res.json()
}

export async function getPresenceResult(jobId) {
  const res = await fetch(`${API_BASE}/api/presence/result/${jobId}`)
  if (!res.ok) throw new Error(`Result fetch failed (${res.status})`)
  return res.json()
}

/**
 * Poll /api/presence/result/{jobId} until the job completes or errors.
 *
 * Presence analysis is much faster than facial tension (≤30s typically), so
 * we use a tighter cadence and a 10-minute timeout.
 */
export function pollPresenceResult(
  jobId,
  onProgress,
  {
    intervalMs = 1000,
    maxIntervalMs = 3000,
    timeoutMs = 10 * 60 * 1000,
    maxConsecutiveErrors = 10,
  } = {},
) {
  const start = Date.now()
  return new Promise((resolve, reject) => {
    let consecutiveErrors = 0
    let pollCount = 0
    const tick = async () => {
      try {
        const data = await getPresenceResult(jobId)
        consecutiveErrors = 0
        onProgress?.(data)
        if (data.status === "done") return resolve(data)
        if (data.status === "error") {
          return reject(new Error(data.message || "Presence pipeline error"))
        }
      } catch (e) {
        consecutiveErrors += 1
        if (consecutiveErrors >= maxConsecutiveErrors) {
          return reject(new Error(
            `Lost contact with backend after ${consecutiveErrors} attempts: ${e.message}`,
          ))
        }
      }
      if (Date.now() - start > timeoutMs) {
        return reject(new Error(
          `Timed out after ${Math.round(timeoutMs / 60000)} min — backend may still be running.`,
        ))
      }
      pollCount += 1
      const next = pollCount < 15 ? intervalMs : maxIntervalMs
      setTimeout(tick, next)
    }
    tick()
  })
}
