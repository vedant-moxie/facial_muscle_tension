import { useEffect, useState } from "react"
import FaceTab     from "./face/Tab.jsx"
import AudioTab    from "./audio/Tab.jsx"
import PresenceTab from "./presence/Tab.jsx"

const TAB_KEY = "ui:active_tab"

export default function App() {
  const [tab, setTab] = useState(() => localStorage.getItem(TAB_KEY) || "face")
  useEffect(() => { localStorage.setItem(TAB_KEY, tab) }, [tab])

  return (
    <div className="min-h-screen">
      <header className="border-b border-line/60 backdrop-blur-md sticky top-0 z-30 bg-bg/70">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-accent to-info shadow-card" />
            <div>
              <div className="text-sm font-semibold tracking-tight">Creator Signal Lab</div>
              <div className="text-xs text-muted -mt-0.5">
                Facial-tension AUs · Sonic rebellion audio · Pose presence · per-creator analysis
              </div>
            </div>
          </div>
          <nav className="flex gap-1">
            {[
              { key: "face",     label: "Face analysis"     },
              { key: "audio",    label: "Audio analysis"    },
              { key: "presence", label: "Presence analysis" },
            ].map(t => (
              <button
                key={t.key}
                onClick={() => setTab(t.key)}
                className={[
                  "px-3 py-1.5 rounded-lg text-sm transition border",
                  tab === t.key
                    ? "bg-accent/15 border-accent/50 text-accent"
                    : "bg-transparent border-transparent text-muted hover:text-text",
                ].join(" ")}
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {tab === "face"     && <FaceTab />}
        {tab === "audio"    && <AudioTab />}
        {tab === "presence" && <PresenceTab />}
      </main>

      <footer className="max-w-7xl mx-auto px-6 py-6 text-xs text-muted">
        py-feat · mediapipe · librosa · numba · FastAPI · React/Vite — analysis is local, no data leaves your machine.
      </footer>
    </div>
  )
}
