import { Component } from "react"

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { error: null, info: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    console.error("Dashboard render error:", error, info)
    this.setState({ info })
  }

  reset = () => this.setState({ error: null, info: null })

  render() {
    const { error, info } = this.state
    if (!error) return this.props.children

    return (
      <div className="card p-6 space-y-3">
        <div className="text-watch font-semibold">Dashboard render failed</div>
        <div className="text-sm text-muted">
          The analysis completed but the UI hit an error while rendering it.
          Open the browser console for the full stack — the message below is
          just the short form.
        </div>
        <pre className="card p-3 text-xs whitespace-pre-wrap overflow-auto max-h-64 font-mono text-watch">
          {String(error?.stack || error)}
        </pre>
        {info?.componentStack && (
          <details className="text-xs text-muted">
            <summary className="cursor-pointer">component stack</summary>
            <pre className="mt-2 whitespace-pre-wrap">{info.componentStack}</pre>
          </details>
        )}
        <button onClick={this.reset} className="text-sm text-info hover:text-text underline-offset-2 hover:underline">
          dismiss
        </button>
      </div>
    )
  }
}
