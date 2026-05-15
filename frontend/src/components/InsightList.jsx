const STYLE = {
  good:  { dot: "bg-good",  label: "Good",  border: "border-good/40", text: "text-good" },
  watch: { dot: "bg-watch", label: "Watch", border: "border-watch/40", text: "text-watch" },
  info:  { dot: "bg-info",  label: "Info",  border: "border-info/40", text: "text-info" },
}

export default function InsightList({ insights }) {
  if (!insights?.length) return <div className="text-sm text-muted">No insights yet.</div>
  return (
    <ul className="space-y-2">
      {insights.map((ins, i) => {
        const s = STYLE[ins.type] || STYLE.info
        return (
          <li
            key={i}
            className={`flex items-start gap-3 rounded-xl border ${s.border} bg-panel2/40 p-3`}
          >
            <span className={`mt-1 h-2 w-2 rounded-full ${s.dot} flex-shrink-0`} />
            <div className="flex-1">
              <div className={`text-[10px] uppercase tracking-widest ${s.text}`}>{s.label}</div>
              <div className="text-sm leading-snug mt-0.5">{ins.text}</div>
            </div>
          </li>
        )
      })}
    </ul>
  )
}
