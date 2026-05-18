import {
  ComposedChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceDot, Legend,
} from "recharts"

function fmt(sec) {
  const s = Math.round(sec)
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-lg bg-panel2 border border-line p-3 text-xs shadow-card">
      <div className="font-mono text-muted">{fmt(label)}</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="flex items-center gap-2 mt-1">
          <span className="h-2 w-2 rounded-full" style={{ background: p.color }} />
          <span className="text-text">{p.name}</span>
          <span className="font-mono text-muted ml-auto">{Number(p.value).toFixed(2)}</span>
        </div>
      ))}
    </div>
  )
}

export default function Timeline({ timeline, events, onSeek }) {
  const data = (timeline || []).map((p) => ({
    second: p.second,
    effortless: p.effortless,
    tension: p.tension ?? null,
  }))

  return (
    <div className="w-full h-[280px]">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 10, right: 16, left: -10, bottom: 0 }}>
          <CartesianGrid stroke="#2a2d3a" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="second"
            tickFormatter={fmt}
            stroke="#8a90a4" fontSize={11} tickLine={false} axisLine={false}
          />
          <YAxis
            domain={[0, 1]} stroke="#8a90a4"
            fontSize={11} tickLine={false} axisLine={false}
            tickFormatter={(v) => v.toFixed(1)}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            verticalAlign="top" align="right" height={28} iconType="circle"
            wrapperStyle={{ fontSize: 12, color: "#8a90a4" }}
          />
          <Line
            type="monotone" dataKey="effortless" name="Effortlessness"
            stroke="#a78bfa" strokeWidth={2} dot={false} activeDot={{ r: 4 }}
          />
          <Line
            type="monotone" dataKey="tension" name="Tension (raw)"
            stroke="#5aa9ff" strokeWidth={1.5} strokeDasharray="4 4" dot={false}
          />
          {(events || []).map((e, i) => (
            <ReferenceDot
              key={i}
              x={Math.round(e.timestamp)} y={Math.min(1, (e.intensity || 0) / 5 + 0.3)}
              r={5} fill="#ff7a59" stroke="#0b0c10" strokeWidth={2}
              isFront
              onClick={() => onSeek?.(e.timestamp)}
              style={{ cursor: "pointer" }}
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
