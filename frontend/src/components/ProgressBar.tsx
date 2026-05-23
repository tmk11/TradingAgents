import type { PipelineStep, StepStatus } from '../types'

// Same order the backend uses, so the user sees the same progression
// they'd see watching the CLI.
const ORDERED_STEPS: { key: PipelineStep; label: string; group: string }[] = [
  { key: 'market_analyst', label: 'Market', group: 'Analysts' },
  { key: 'sentiment_analyst', label: 'Sentiment', group: 'Analysts' },
  { key: 'news_analyst', label: 'News', group: 'Analysts' },
  { key: 'fundamentals_analyst', label: 'Fundamentals', group: 'Analysts' },
  { key: 'bull_researcher', label: 'Bull', group: 'Research' },
  { key: 'bear_researcher', label: 'Bear', group: 'Research' },
  { key: 'research_manager', label: 'Research Mgr', group: 'Research' },
  { key: 'trader', label: 'Trader', group: 'Trading' },
  { key: 'risk_aggressive', label: 'Aggressive', group: 'Risk' },
  { key: 'risk_conservative', label: 'Conservative', group: 'Risk' },
  { key: 'risk_neutral', label: 'Neutral', group: 'Risk' },
  { key: 'portfolio_manager', label: 'Portfolio Mgr', group: 'Risk' },
]

const STATUS_DOT: Record<StepStatus, string> = {
  pending: 'bg-gold-200',
  running: 'bg-blue-500 animate-pulse',
  completed: 'bg-emerald-500',
  skipped: 'bg-gold-300 opacity-50',
}

export function ProgressBar({
  progress,
  compact = false,
}: {
  progress: Record<string, StepStatus>
  compact?: boolean
}) {
  // Compact view (used in list rows): single horizontal bar with one
  // dot per step.
  if (compact) {
    const total = ORDERED_STEPS.filter(
      (s) => progress[s.key] !== 'skipped',
    ).length
    const done = ORDERED_STEPS.filter(
      (s) => progress[s.key] === 'completed',
    ).length
    const pct = total === 0 ? 0 : Math.round((done / total) * 100)
    return (
      <div className="flex items-center gap-2">
        <div className="flex-1 bg-gold-100 rounded-full h-2 overflow-hidden">
          <div
            className="h-full bg-emerald-500 transition-all"
            style={{ width: `${pct}%` }}
          />
        </div>
        <span className="text-xs font-mono tabular-nums text-gold-700 w-14 text-right">
          {done}/{total}
        </span>
      </div>
    )
  }

  // Full view (detail page): grouped checklist of every step.
  const groups: Record<string, typeof ORDERED_STEPS> = {}
  for (const step of ORDERED_STEPS) {
    ;(groups[step.group] ??= []).push(step)
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {Object.entries(groups).map(([group, steps]) => (
        <div
          key={group}
          className="border border-gold-200 rounded-lg p-3 bg-white"
        >
          <h4 className="text-xs font-semibold text-gold-700 uppercase tracking-wide mb-2">
            {group}
          </h4>
          <ul className="space-y-1.5">
            {steps.map((step) => {
              const status = progress[step.key] ?? 'pending'
              return (
                <li
                  key={step.key}
                  className="flex items-center gap-2 text-sm"
                >
                  <span
                    className={`w-2.5 h-2.5 rounded-full ${STATUS_DOT[status]}`}
                    aria-hidden
                  />
                  <span
                    className={
                      status === 'skipped'
                        ? 'text-gold-400 line-through'
                        : status === 'completed'
                          ? 'text-gold-900'
                          : 'text-gold-700'
                    }
                  >
                    {step.label}
                  </span>
                </li>
              )
            })}
          </ul>
        </div>
      ))}
    </div>
  )
}
