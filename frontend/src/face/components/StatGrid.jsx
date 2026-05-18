function Stat({ label, value, sub }) {
  return (
    <div className="card p-4">
      <div className="text-xs uppercase tracking-wider text-muted">{label}</div>
      <div className="mt-1 font-mono text-2xl tabular-nums">{value}</div>
      {sub && <div className="text-xs text-muted mt-1">{sub}</div>}
    </div>
  )
}

export default function StatGrid({ frames, events, blinkRate, asym }) {
  return (
    <div className="grid grid-cols-2 gap-3">
      <Stat label="Frames" value={frames} sub="analysed" />
      <Stat label="Micro-events" value={events} sub="40–200 ms bursts" />
      <Stat
        label="Blink rate"
        value={Number(blinkRate || 0).toFixed(1)}
        sub={`${blinkRate >= 12 && blinkRate <= 20 ? "in" : "out of"} relaxed band (12–20)`}
      />
      <Stat
        label="Asymmetry"
        value={Number(asym || 0).toFixed(2)}
        sub={`${asym >= 0.08 && asym <= 0.30 ? "in" : "out of"} genuine band (0.08–0.30)`}
      />
    </div>
  )
}
