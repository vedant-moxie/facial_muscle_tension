import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, LabelList,
} from "recharts"

const AU_META = {
  AU04: { label: "Brow lowerer (AU4)",       kind: "tension" },
  AU07: { label: "Lid tightener (AU7)",      kind: "tension" },
  AU10: { label: "Upper-lip raise (AU10)",   kind: "tension" },
  AU43: { label: "Eyes closed (AU43)",       kind: "tension" },
  AU17: { label: "Chin raiser (AU17)",       kind: "tension" },
  AU23: { label: "Lip tightener (AU23)",     kind: "tension" },
  AU06: { label: "Cheek raiser (AU6)",       kind: "positive" },
  AU12: { label: "Lip-corner pull (AU12)",   kind: "positive" },
  AU01: { label: "Inner brow raise (AU1)",   kind: "neutral" },
  AU02: { label: "Outer brow raise (AU2)",   kind: "neutral" },
  AU25: { label: "Lips part (AU25)",         kind: "neutral" },
  AU26: { label: "Jaw drop (AU26)",          kind: "neutral" },
}

const COLOR = {
  tension:  "#ff7a59",
  positive: "#3ddc97",
  neutral:  "#5aa9ff",
}

export default function AUBarChart({ auMeans }) {
  const data = Object.keys(AU_META)
    .filter((k) => k in (auMeans || {}))
    .map((k) => ({
      key: k,
      label: AU_META[k].label,
      value: Number(auMeans[k] || 0),
      kind: AU_META[k].kind,
    }))
    .sort((a, b) => b.value - a.value)

  return (
    <div className="w-full" style={{ height: Math.max(220, data.length * 28) }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, left: 4, bottom: 4 }}>
          <CartesianGrid stroke="#2a2d3a" strokeDasharray="3 3" horizontal={false} />
          <XAxis type="number" domain={[0, 5]} stroke="#8a90a4" fontSize={11} tickLine={false} axisLine={false} />
          <YAxis
            type="category" dataKey="label" width={170}
            stroke="#8a90a4" fontSize={11} tickLine={false} axisLine={false}
          />
          <Tooltip
            contentStyle={{ background: "#1d1f2a", border: "1px solid #2a2d3a", borderRadius: 8, fontSize: 12 }}
            cursor={{ fill: "#ffffff05" }}
            formatter={(v) => Number(v).toFixed(2)}
          />
          <Bar dataKey="value" radius={[4, 4, 4, 4]}>
            {data.map((d, i) => <Cell key={i} fill={COLOR[d.kind]} />)}
            <LabelList dataKey="value" position="right" fontSize={11} fill="#8a90a4"
                       formatter={(v) => Number(v).toFixed(2)} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
