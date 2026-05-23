import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'

import { api } from '../api'
import type {
  AnalysisSummary,
  DecisionStats,
  HorizonStats,
  TrackRecord,
} from '../types'

// Order matches server.backtest.HORIZONS so the UI mirrors the
// backend's canonical sequence.
const HORIZON_ORDER = ['1d', '5d', '21d', '63d'] as const
type HorizonKey = (typeof HORIZON_ORDER)[number]

const HORIZON_LABELS: Record<HorizonKey, string> = {
  '1d': '1 day',
  '5d': '1 week',
  '21d': '1 month',
  '63d': '1 quarter',
}

// Canonical 5-tier scale from tradingagents.agents.utils.rating —
// keep the order so the per-decision table reads bullish-to-bearish.
const DECISION_ORDER = [
  'Buy',
  'Overweight',
  'Hold',
  'Underweight',
  'Sell',
  'Unknown',
] as const

export function TrackRecordPage() {
  const tr = useQuery({
    queryKey: ['track-record'],
    queryFn: () => api.trackRecord(),
    // Refresh in the background so newly-elapsed horizons fill in
    // without the user having to reload manually.
    refetchInterval: 60_000,
  })

  const list = useQuery({
    queryKey: ['analyses'],
    queryFn: () => api.list(),
  })

  return (
    <div className="space-y-6">
      <Link
        to="/"
        className="inline-flex items-center gap-1 text-sm text-gold-700 hover:text-gold-900"
      >
        ← Back to dashboard
      </Link>

      <div>
        <h2 className="text-2xl font-bold text-gold-900 mb-1">
          Track record
        </h2>
        <p className="text-sm text-gold-600 max-w-2xl">
          Forward-return scoring for every completed analysis. Each call
          (Buy / Overweight / Hold / Underweight / Sell) is graded against
          actual price action at four horizons. Outcomes fill in
          automatically as time passes — older calls always have more
          horizons resolved.
        </p>
      </div>

      <RealityCheck />

      {tr.isLoading && (
        <div className="text-center py-12 text-gold-600">Loading…</div>
      )}

      {tr.error && (
        <div className="border border-rose-200 bg-rose-50 text-rose-800 rounded-lg p-4">
          <div className="font-semibold mb-1">Could not load track record</div>
          <div className="font-mono text-xs break-all">
            {(tr.error as Error).message}
          </div>
        </div>
      )}

      {tr.data && (
        <>
          <Summary data={tr.data} />
          <PerDecisionTable data={tr.data} />
        </>
      )}

      {list.data && tr.data && (
        <PerAnalysisTable analyses={list.data} />
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------

function RealityCheck() {
  return (
    <div className="border border-amber-200 bg-amber-50 rounded-lg p-4 text-sm text-amber-900">
      <div className="font-semibold mb-1">A note on accuracy</div>
      <p className="leading-relaxed">
        Even top-tier quantitative funds hit roughly 52–58% directional
        accuracy on short-horizon commodity calls. A track record that
        hovers around 50% on a small sample is{' '}
        <em>statistically indistinguishable from a coin flip</em> — the
        signal only emerges after dozens of calls. Don't over-react to
        the first few rows.
      </p>
    </div>
  )
}

// ---------------------------------------------------------------------------

function Summary({ data }: { data: TrackRecord }) {
  return (
    <section>
      <h3 className="text-sm font-semibold text-gold-700 uppercase tracking-wide mb-3">
        Hit rate by horizon
      </h3>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {HORIZON_ORDER.map((h) => (
          <HorizonCard key={h} horizon={h} stats={data.horizons[h]} />
        ))}
      </div>
      <div className="text-xs text-gold-500 mt-2">
        {data.total_with_outcomes} of {data.total_completed} completed
        {' '}analyses scored ·{' '}
        last computed {formatDate(data.computed_at)}
      </div>
    </section>
  )
}

function HorizonCard({
  horizon,
  stats,
}: {
  horizon: HorizonKey
  stats: HorizonStats | undefined
}) {
  const total = stats?.total ?? 0
  const hitRate = stats?.hit_rate ?? null
  const tone = hitRateTone(hitRate, total)

  return (
    <div className="bg-white border border-gold-200 rounded-xl p-4 shadow-sm">
      <div className="text-xs text-gold-600 uppercase tracking-wide mb-1">
        {HORIZON_LABELS[horizon]}
      </div>
      <div className="flex items-baseline gap-1.5">
        <span
          className={`text-3xl font-bold tabular-nums ${tone.text}`}
        >
          {hitRate === null ? '—' : `${(hitRate * 100).toFixed(1)}%`}
        </span>
        <span className="text-xs text-gold-500">hit rate</span>
      </div>
      <div className="text-xs text-gold-600 mt-1">
        {stats?.correct ?? 0} of {total} calls correct
      </div>
      {/* Tiny progress bar visual */}
      <div className="mt-2 h-1.5 rounded-full bg-gold-100 overflow-hidden">
        <div
          className={`h-full ${tone.bar}`}
          style={{
            width: hitRate === null ? '0%' : `${Math.min(100, hitRate * 100)}%`,
          }}
        />
      </div>
    </div>
  )
}

function hitRateTone(
  hr: number | null,
  total: number,
): { text: string; bar: string } {
  // Small samples → stay neutral. We don't want the UI screaming
  // "100% hit rate!" when n=1.
  if (hr === null || total < 5) {
    return { text: 'text-gold-700', bar: 'bg-gold-400' }
  }
  if (hr >= 0.6) return { text: 'text-emerald-700', bar: 'bg-emerald-500' }
  if (hr >= 0.5) return { text: 'text-gold-700', bar: 'bg-gold-500' }
  return { text: 'text-rose-700', bar: 'bg-rose-500' }
}

// ---------------------------------------------------------------------------

function PerDecisionTable({ data }: { data: TrackRecord }) {
  // Collect the union of decision keys actually present in the data
  // so we don't render rows for ratings nobody has called yet.
  const seen = new Set<string>()
  for (const h of HORIZON_ORDER) {
    const stats = data.horizons[h]
    if (!stats) continue
    for (const k of Object.keys(stats.by_decision)) {
      seen.add(k)
    }
  }
  // Render in the canonical bullish→bearish order.
  const rows = DECISION_ORDER.filter((d) => seen.has(d))
  if (rows.length === 0) return null

  return (
    <section>
      <h3 className="text-sm font-semibold text-gold-700 uppercase tracking-wide mb-3">
        By decision
      </h3>
      <div className="bg-white border border-gold-200 rounded-xl overflow-hidden shadow-sm">
        <table className="w-full text-sm">
          <thead className="bg-gold-50 text-gold-700">
            <tr>
              <th className="text-left px-4 py-2 font-semibold">Decision</th>
              {HORIZON_ORDER.map((h) => (
                <th
                  key={h}
                  className="text-right px-4 py-2 font-semibold whitespace-nowrap"
                >
                  {HORIZON_LABELS[h]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gold-100">
            {rows.map((decision) => (
              <tr key={decision} className="hover:bg-gold-50/40">
                <td className="px-4 py-2 font-medium text-gold-900">
                  {decision}
                </td>
                {HORIZON_ORDER.map((h) => (
                  <td
                    key={h}
                    className="px-4 py-2 text-right tabular-nums"
                  >
                    <DecisionCell
                      stats={data.horizons[h]?.by_decision[decision]}
                    />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function DecisionCell({ stats }: { stats: DecisionStats | undefined }) {
  if (!stats || stats.total === 0) {
    return <span className="text-gold-400">—</span>
  }
  const tone = hitRateTone(stats.hit_rate, stats.total)
  return (
    <span>
      <span className={`font-semibold ${tone.text}`}>
        {stats.hit_rate === null
          ? '—'
          : `${(stats.hit_rate * 100).toFixed(0)}%`}
      </span>
      <span className="text-gold-500 ml-1 text-xs">
        ({stats.correct}/{stats.total})
      </span>
    </span>
  )
}

// ---------------------------------------------------------------------------

function PerAnalysisTable({ analyses }: { analyses: AnalysisSummary[] }) {
  const completed = analyses.filter((a) => a.status === 'completed')
  if (completed.length === 0) {
    return (
      <section>
        <h3 className="text-sm font-semibold text-gold-700 uppercase tracking-wide mb-3">
          Per-analysis breakdown
        </h3>
        <div className="border border-dashed border-gold-200 rounded-lg p-8 text-center text-gold-600 bg-white">
          No completed analyses yet. Start one to begin building a track record.
        </div>
      </section>
    )
  }

  return (
    <section>
      <h3 className="text-sm font-semibold text-gold-700 uppercase tracking-wide mb-3">
        Per-analysis breakdown
      </h3>
      <div className="bg-white border border-gold-200 rounded-xl overflow-hidden shadow-sm">
        <table className="w-full text-sm">
          <thead className="bg-gold-50 text-gold-700">
            <tr>
              <th className="text-left px-4 py-2 font-semibold">Date</th>
              <th className="text-left px-4 py-2 font-semibold">Ticker</th>
              <th className="text-left px-4 py-2 font-semibold">Decision</th>
              {HORIZON_ORDER.map((h) => (
                <th
                  key={h}
                  className="text-center px-3 py-2 font-semibold whitespace-nowrap"
                >
                  {HORIZON_LABELS[h]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gold-100">
            {completed.map((a) => (
              <AnalysisRow key={a.id} analysis={a} />
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function AnalysisRow({ analysis }: { analysis: AnalysisSummary }) {
  // Pull the per-analysis outcome lazily so this table can render
  // immediately even on cold-cache (the aggregate query above will
  // already have warmed the cache for most rows).
  const o = useQuery({
    queryKey: ['outcome', analysis.id],
    queryFn: () => api.outcome(analysis.id),
    // Per-analysis outcomes change rarely after all horizons resolve —
    // a 60 s stale time is enough.
    staleTime: 60_000,
  })

  const byHorizon = new Map(
    (o.data?.horizons ?? []).map((h) => [h.horizon, h]),
  )

  return (
    <tr className="hover:bg-gold-50/40">
      <td className="px-4 py-2 font-mono text-xs text-gold-700">
        {analysis.analysis_date}
      </td>
      <td className="px-4 py-2 font-mono font-semibold text-gold-900">
        <Link
          to={`/analyses/${analysis.id}`}
          className="hover:underline"
        >
          {analysis.ticker}
        </Link>
      </td>
      <td className="px-4 py-2 text-gold-800">
        {analysis.final_decision ?? <span className="text-gold-400">—</span>}
      </td>
      {HORIZON_ORDER.map((h) => (
        <td key={h} className="px-3 py-2 text-center">
          <HorizonGlyph horizon={byHorizon.get(h)} />
        </td>
      ))}
    </tr>
  )
}

function HorizonGlyph({
  horizon,
}: {
  horizon:
    | {
        correct: boolean | null
        forward_return: number | null
        actual_direction: string
      }
    | undefined
}) {
  if (!horizon || horizon.correct === null) {
    return (
      <span
        className="text-gold-400 text-xs"
        title="Not enough time elapsed yet"
      >
        ⋯
      </span>
    )
  }
  const ret = horizon.forward_return ?? 0
  const pct = `${(ret * 100).toFixed(2)}%`
  if (horizon.correct) {
    return (
      <span
        className="text-emerald-700 font-semibold text-xs"
        title={`Correct (${horizon.actual_direction}, ${pct})`}
      >
        ✓ {pct}
      </span>
    )
  }
  return (
    <span
      className="text-rose-700 font-semibold text-xs"
      title={`Wrong (${horizon.actual_direction}, ${pct})`}
    >
      ✗ {pct}
    </span>
  )
}

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString()
  } catch {
    return iso
  }
}
