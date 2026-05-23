import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

import { api } from '../api'
import type {
  AnalysisDetail,
  AnalysisOutcome,
  AnalysisReports,
  HorizonOutcome,
} from '../types'
import { DecisionBadge, StatusBadge } from './StatusBadge'
import { ProgressBar } from './ProgressBar'

export function AnalysisDetailPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const qc = useQueryClient()
  const [confirming, setConfirming] = useState(false)

  const { data, isLoading, error } = useQuery({
    queryKey: ['analysis', id],
    enabled: Boolean(id),
    queryFn: () => api.get(id!),
    refetchInterval: (query) => {
      const a = query.state.data as AnalysisDetail | undefined
      const live =
        a?.status === 'pending' || a?.status === 'running'
      return live ? 3000 : false
    },
  })

  const remove = useMutation({
    mutationFn: () => api.remove(id!),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['analyses'] })
      navigate('/')
    },
  })

  if (isLoading) {
    return <div className="text-center py-12 text-gold-600">Loading…</div>
  }

  if (error) {
    return (
      <div className="border border-rose-200 bg-rose-50 text-rose-800 rounded-lg p-4">
        <div className="font-semibold mb-1">Could not load analysis</div>
        <div className="font-mono text-sm break-all">
          {(error as Error).message}
        </div>
        <Link
          to="/"
          className="inline-block mt-3 text-sm underline text-rose-700"
        >
          Back to dashboard
        </Link>
      </div>
    )
  }

  if (!data) {
    return null
  }

  return (
    <div className="space-y-6">
      <Link
        to="/"
        className="inline-flex items-center gap-1 text-sm text-gold-700 hover:text-gold-900"
      >
        ← Back to dashboard
      </Link>

      <Header a={data} />

      {data.status === 'failed' && data.error && (
        <ErrorBlock error={data.error} />
      )}

      <Section title="Pipeline progress">
        <ProgressBar progress={data.progress} />
      </Section>

      {data.status === 'completed' && <ForwardOutcomeSection id={data.id} />}

      <ReportsTabs reports={data.reports} status={data.status} />

      <DangerZone
        confirming={confirming}
        deleting={remove.isPending}
        onCancel={() => setConfirming(false)}
        onAsk={() => setConfirming(true)}
        onConfirm={() => remove.mutate()}
      />
    </div>
  )
}

function Header({ a }: { a: AnalysisDetail }) {
  return (
    <div className="bg-white border border-gold-200 rounded-xl p-5 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-3 flex-wrap">
            <h2 className="text-2xl font-bold font-mono text-gold-900">
              {a.ticker}
            </h2>
            <StatusBadge status={a.status} />
            <DecisionBadge decision={a.final_decision} />
          </div>
          <div className="text-sm text-gold-600 mt-1">
            <span className="capitalize">{a.asset_type}</span> · analysis date{' '}
            <span className="font-mono">{a.analysis_date}</span> ·{' '}
            language <span className="font-medium">{a.language}</span>
            {(a.max_debate_rounds || a.max_risk_discuss_rounds) && (
              <>
                {' '}
                · debate{' '}
                <span className="font-mono">
                  {a.max_debate_rounds ?? 1}/{a.max_risk_discuss_rounds ?? 1}
                </span>{' '}
                <span
                  className="text-gold-500"
                  title="Bull-Bear rounds / Risk-debate rounds"
                >
                  rounds
                </span>
              </>
            )}
          </div>
        </div>
        <div className="text-xs text-gold-500 text-right">
          <div>Created: {formatDate(a.created_at)}</div>
          {a.completed_at && (
            <div>Completed: {formatDate(a.completed_at)}</div>
          )}
        </div>
      </div>
    </div>
  )
}

function formatDate(iso: string): string {
  try {
    const d = new Date(iso)
    return d.toLocaleString()
  } catch {
    return iso
  }
}

function Section({
  title,
  children,
}: {
  title: string
  children: React.ReactNode
}) {
  return (
    <section>
      <h3 className="text-sm font-semibold text-gold-700 uppercase tracking-wide mb-3">
        {title}
      </h3>
      {children}
    </section>
  )
}

function ErrorBlock({ error }: { error: string }) {
  return (
    <div className="border border-rose-200 bg-rose-50 text-rose-800 rounded-lg p-4">
      <div className="font-semibold mb-1">Run failed</div>
      <pre className="text-xs font-mono whitespace-pre-wrap break-all max-h-64 overflow-auto">
        {error}
      </pre>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Reports tabs
// ---------------------------------------------------------------------------

interface TabSpec {
  key: string
  label: string
  // Function returns the markdown body or undefined when nothing to show.
  pick: (r: AnalysisReports) => string | undefined
}

const TABS: TabSpec[] = [
  { key: 'final', label: 'Final decision', pick: (r) => r.final_trade_decision },
  { key: 'trader', label: 'Trader', pick: (r) => r.trader_investment_plan },
  { key: 'plan', label: 'Research plan', pick: (r) => r.investment_plan },
  { key: 'market', label: 'Market', pick: (r) => r.market_report },
  { key: 'sentiment', label: 'Sentiment', pick: (r) => r.sentiment_report },
  { key: 'news', label: 'News', pick: (r) => r.news_report },
  { key: 'fundamentals', label: 'Fundamentals', pick: (r) => r.fundamentals_report },
  {
    key: 'invest_debate',
    label: 'Bull / Bear debate',
    pick: (r) => r.investment_debate_state?.history,
  },
  {
    key: 'risk_debate',
    label: 'Risk debate',
    pick: (r) => r.risk_debate_state?.history,
  },
]

function ReportsTabs({
  reports,
  status,
}: {
  reports: AnalysisReports
  status: string
}) {
  // Hide tabs whose section is empty so the user isn't presented with
  // a long list of "(empty)" panels — most useful for in-progress runs
  // and for crypto/commodity which skip fundamentals.
  const available = TABS.filter((t) => {
    const v = t.pick(reports)
    return Boolean(v && v.trim())
  })

  const [active, setActive] = useState<string>(
    available[0]?.key ?? TABS[0].key,
  )

  if (status !== 'completed' && available.length === 0) {
    return (
      <Section title="Reports">
        <div className="border border-dashed border-gold-200 rounded-lg p-8 text-center text-gold-600 bg-white">
          The pipeline is still running — reports will appear here as each
          agent finishes.
        </div>
      </Section>
    )
  }

  if (available.length === 0) {
    return (
      <Section title="Reports">
        <div className="border border-dashed border-gold-200 rounded-lg p-8 text-center text-gold-600 bg-white">
          No reports were produced.
        </div>
      </Section>
    )
  }

  const activeTab =
    available.find((t) => t.key === active) ?? available[0]
  const body = activeTab.pick(reports) ?? ''

  return (
    <Section title="Reports">
      <div className="bg-white border border-gold-200 rounded-xl shadow-sm overflow-hidden">
        <div className="flex flex-wrap gap-1 p-2 border-b border-gold-100 bg-gold-50">
          {available.map((t) => (
            <button
              key={t.key}
              type="button"
              onClick={() => setActive(t.key)}
              className={`px-3 py-1.5 text-sm rounded-md font-medium transition-colors ${
                t.key === activeTab.key
                  ? 'bg-gold-600 text-white shadow'
                  : 'text-gold-700 hover:bg-gold-100'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
        <div className="p-6 markdown">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{body}</ReactMarkdown>
        </div>
      </div>
    </Section>
  )
}

// ---------------------------------------------------------------------------
// Danger zone
// ---------------------------------------------------------------------------

function DangerZone({
  confirming,
  deleting,
  onAsk,
  onCancel,
  onConfirm,
}: {
  confirming: boolean
  deleting: boolean
  onAsk: () => void
  onCancel: () => void
  onConfirm: () => void
}) {
  return (
    <Section title="Danger zone">
      <div className="border border-rose-200 bg-rose-50 rounded-xl p-4 flex items-center justify-between flex-wrap gap-3">
        <div>
          <div className="text-sm font-semibold text-rose-900">
            Delete this analysis
          </div>
          <div className="text-xs text-rose-700">
            Removes the JSON file from disk. The decision will no longer be
            visible to future memory-log lookups for this ticker.
          </div>
        </div>
        {confirming ? (
          <span className="flex items-center gap-2">
            <button
              type="button"
              disabled={deleting}
              onClick={onConfirm}
              className="px-3 py-1.5 text-sm font-semibold bg-rose-600 hover:bg-rose-700 text-white rounded disabled:opacity-60"
            >
              {deleting ? 'Deleting…' : 'Yes, delete'}
            </button>
            <button
              type="button"
              onClick={onCancel}
              className="px-3 py-1.5 text-sm text-rose-700 hover:text-rose-900"
            >
              Cancel
            </button>
          </span>
        ) : (
          <button
            type="button"
            onClick={onAsk}
            className="px-3 py-1.5 text-sm font-semibold border border-rose-300 text-rose-700 hover:bg-rose-100 rounded"
          >
            Delete analysis
          </button>
        )}
      </div>
    </Section>
  )
}


// ---------------------------------------------------------------------------
// Forward outcome
// ---------------------------------------------------------------------------

const HORIZON_LABELS_DETAIL: Record<string, string> = {
  '1d': '1 day',
  '5d': '1 week',
  '21d': '1 month',
  '63d': '1 quarter',
}

function ForwardOutcomeSection({ id }: { id: string }) {
  const o = useQuery({
    queryKey: ['outcome', id],
    queryFn: () => api.outcome(id),
    // Cache for 60 s — outcomes rarely change once horizons resolve.
    staleTime: 60_000,
  })

  // Render a placeholder while loading rather than nothing — the
  // section heading should always be present so users notice it.
  return (
    <Section title="Forward outcome">
      <p className="text-xs text-gold-500 mb-3">
        How the call held up against actual price action. Horizons that
        haven't elapsed yet show "—" and fill in automatically over time.
      </p>
      {o.isLoading && (
        <div className="text-sm text-gold-600">Computing outcome…</div>
      )}
      {o.error && (
        <div className="text-sm text-rose-700">
          Could not load outcome:{' '}
          <span className="font-mono text-xs">
            {(o.error as Error).message}
          </span>
        </div>
      )}
      {o.data && <OutcomeGrid outcome={o.data} />}
    </Section>
  )
}

function OutcomeGrid({ outcome }: { outcome: AnalysisOutcome }) {
  return (
    <div className="bg-white border border-gold-200 rounded-xl p-4 shadow-sm">
      <div className="text-xs text-gold-600 mb-3">
        Decision <span className="font-semibold">{outcome.decision}</span> ·
        expected{' '}
        <span className="font-mono">
          {outcome.expected_direction ?? 'n/a'}
        </span>
        {outcome.start_close !== null && (
          <>
            {' '}
            · entry close{' '}
            <span className="font-mono">{outcome.start_close.toFixed(2)}</span>
          </>
        )}
      </div>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {outcome.horizons.map((h) => (
          <OutcomeCard key={h.horizon} horizon={h} />
        ))}
      </div>
    </div>
  )
}

function OutcomeCard({ horizon }: { horizon: HorizonOutcome }) {
  const pending = horizon.correct === null
  const tone = pending
    ? 'border-gold-200 bg-gold-50/50'
    : horizon.correct
      ? 'border-emerald-200 bg-emerald-50'
      : 'border-rose-200 bg-rose-50'

  const ret = horizon.forward_return
  const retText =
    ret === null ? '—' : `${ret >= 0 ? '+' : ''}${(ret * 100).toFixed(2)}%`

  return (
    <div className={`rounded-lg border ${tone} p-3`}>
      <div className="flex items-center justify-between text-xs">
        <span className="font-semibold uppercase tracking-wide text-gold-700">
          {HORIZON_LABELS_DETAIL[horizon.horizon] ?? horizon.horizon}
        </span>
        <Verdict correct={horizon.correct} />
      </div>
      <div className="mt-1.5 text-2xl font-bold tabular-nums">{retText}</div>
      <div className="text-xs text-gold-600 mt-0.5">
        actual{' '}
        <span className="font-mono">{horizon.actual_direction}</span>
        {horizon.end_close !== null && (
          <>
            {' '}· close{' '}
            <span className="font-mono">{horizon.end_close.toFixed(2)}</span>
          </>
        )}
      </div>
      <div className="text-[10px] text-gold-500 mt-1">
        target {horizon.target_date}
      </div>
    </div>
  )
}

function Verdict({ correct }: { correct: boolean | null }) {
  if (correct === null) {
    return (
      <span className="text-gold-500 text-[10px] uppercase">Pending</span>
    )
  }
  if (correct) {
    return (
      <span className="text-emerald-700 text-[10px] uppercase font-semibold">
        ✓ Correct
      </span>
    )
  }
  return (
    <span className="text-rose-700 text-[10px] uppercase font-semibold">
      ✗ Wrong
    </span>
  )
}
