/*
 * Sonic Rebellion — frontend API client.
 *
 * Mirrors the polling pattern used by the face pipeline (api.js) but talks
 * to the /api/audio/* endpoints.
 */
const API_BASE = import.meta.env.VITE_API_BASE || ""

export async function uploadAudioReels({ files = null, profileUrl = null, reelUrls = null } = {}) {
  const form = new FormData()
  if (files && files.length) {
    for (const f of files) form.append("files", f)
  }
  if (profileUrl) form.append("profile_url", profileUrl)
  if (reelUrls && reelUrls.length) form.append("reel_urls", reelUrls.join(","))

  const res = await fetch(`${API_BASE}/api/audio/analyse`, { method: "POST", body: form })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `Upload failed (${res.status})`)
  }
  return res.json()    // { job_id, n_reels }
}

export async function getAudioResult(jobId) {
  const res = await fetch(`${API_BASE}/api/audio/result/${jobId}`)
  if (!res.ok) throw new Error(`Result fetch failed (${res.status})`)
  return res.json()
}

export function pollAudioResult(
  jobId,
  onProgress,
  {
    intervalMs = 2000,
    maxIntervalMs = 5000,
    timeoutMs = 60 * 60 * 1000,
    maxConsecutiveErrors = 10,
  } = {},
) {
  const start = Date.now()
  return new Promise((resolve, reject) => {
    let consecutiveErrors = 0
    let pollCount = 0

    const tick = async () => {
      try {
        const data = await getAudioResult(jobId)
        consecutiveErrors = 0
        onProgress?.(data)
        if (data.status === "done") return resolve(data)
        if (data.status === "error") return reject(new Error(data.message || "Pipeline error"))
      } catch (e) {
        consecutiveErrors += 1
        if (consecutiveErrors >= maxConsecutiveErrors) {
          return reject(new Error(
            `Lost contact with backend after ${consecutiveErrors} attempts: ${e.message}`,
          ))
        }
      }
      if (Date.now() - start > timeoutMs) {
        return reject(new Error(`Timed out after ${Math.round(timeoutMs / 60000)} min.`))
      }
      pollCount += 1
      const next = pollCount < 30 ? intervalMs : maxIntervalMs
      setTimeout(tick, next)
    }
    tick()
  })
}
