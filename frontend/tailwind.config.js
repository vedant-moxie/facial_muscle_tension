/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        bg:       "#0b0c10",
        panel:    "#15171f",
        panel2:   "#1d1f2a",
        line:     "#2a2d3a",
        text:     "#e7e9ef",
        muted:    "#8a90a4",
        good:     "#3ddc97",
        watch:    "#ff7a59",
        info:     "#5aa9ff",
        accent:   "#a78bfa",
      },
      fontFamily: {
        sans: ['"Inter"', "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      boxShadow: {
        card: "0 1px 0 rgba(255,255,255,0.04) inset, 0 1px 24px rgba(0,0,0,0.30)",
      },
    },
  },
  plugins: [],
}
