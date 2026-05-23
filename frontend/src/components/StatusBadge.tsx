import type { AnalysisStatus } from '../types'

const STYLES: Record<AnalysisStatus, { label: string; className: string }> = {
  pending: {
    label: 'Pending',
    className: 'bg-gold-100 text-gold-800 border-gold-200',
  },
  running: {
    label: 'Running',
    className:
      'bg-blue-100 text-blue-800 border-blue-200 animate-pulse',
  },
  completed: {
    label: 'Completed',
    className: 'bg-emerald-100 text-emerald-800 border-emerald-200',
  },
  failed: {
    label: 'Failed',
    className: 'bg-rose-100 text-rose-800 border-rose-200',
  },
}

export function StatusBadge({ status }: { status: AnalysisStatus }) {
  const cfg = STYLES[status] ?? STYLES.pending
  return (
    <span
      className={`inline-flex items-center px-2.5 py-1 text-xs font-semibold rounded-full border ${cfg.className}`}
    >
      {cfg.label}
    </span>
  )
}

const DECISION_STYLES: Record<string, string> = {
  Buy: 'bg-emerald-600 text-white',
  Overweight: 'bg-emerald-500 text-white',
  Hold: 'bg-amber-500 text-white',
  Underweight: 'bg-rose-500 text-white',
  Sell: 'bg-rose-700 text-white',
}

export function DecisionBadge({ decision }: { decision: string | null }) {
  if (!decision) return null
  // Normalise common renderings (e.g. "BUY", "buy", "Strong Buy").
  const normal =
    Object.keys(DECISION_STYLES).find((k) =>
      decision.toUpperCase().includes(k.toUpperCase()),
    ) ?? null
  const cls = normal
    ? DECISION_STYLES[normal]
    : 'bg-gold-200 text-gold-900'
  return (
    <span
      className={`inline-flex items-center px-2.5 py-1 text-xs font-bold rounded-md uppercase tracking-wide ${cls}`}
    >
      {normal ?? decision}
    </span>
  )
}
