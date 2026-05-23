import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { Link } from 'react-router-dom'

import { api } from '../api'
import type { AnalysisSummary } from '../types'
import { DecisionBadge, StatusBadge } from './StatusBadge'
import { ProgressBar } from './ProgressBar'

export function AnalysisListPage() {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['analyses'],
    queryFn: api.list,
    // Poll every 3s while any run is still in flight; idle when all
    // are terminal so the user's network isn't thrashed needlessly.
    refetchInterval: (query) => {
      const list = (query.state.data as AnalysisSummary[] | undefined) ?? []
      const live = list.some(
        (a) => a.status === 'pending' || a.status === 'running',
      )
      return live ? 3000 : false
    },
  })

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between flex-wrap gap-4">
        <div>
          <h2 className="text-2xl font-bold text-gold-900">Analyses</h2>
          <p className="text-sm text-gold-600 mt-1">
            Background runs of the multi-agent gold pipeline. Click any row
            to inspect its full report.
          </p>
        </div>
        <Link
          to="/new"
          className="px-4 py-2 text-sm font-semibold bg-gold-600 text-white hover:bg-gold-700 rounded-md shadow-sm"
        >
          + New analysis
        </Link>
      </div>

      {isLoading && (
        <div className="text-center py-12 text-gold-600">Loading…</div>
      )}

      {error && (
        <ErrorPanel
          message={(error as Error).message}
          onRetry={() => refetch()}
        />
      )}

      {data && data.length === 0 && <EmptyState />}

      {data && data.length > 0 && <AnalysisTable items={data} />}
    </div>
  )
}

function EmptyState() {
  return (
    <div className="border-2 border-dashed border-gold-200 rounded-xl p-12 text-center bg-white">
      <div className="w-16 h-16 mx-auto rounded-full bg-gold-100 flex items-center justify-center text-3xl mb-4">
        🪙
      </div>
      <h3 className="text-lg font-semibold text-gold-900 mb-1">
        No analyses yet
      </h3>
      <p className="text-sm text-gold-600 mb-4 max-w-sm mx-auto">
        Kick off your first run with a gold-complex ticker like{' '}
        <code className="font-mono bg-gold-100 px-1.5 py-0.5 rounded">
          GLD
        </code>
        , <code className="font-mono bg-gold-100 px-1.5 py-0.5 rounded">GC=F</code>
        , or{' '}
        <code className="font-mono bg-gold-100 px-1.5 py-0.5 rounded">
          XAUUSD=X
        </code>
        .
      </p>
      <Link
        to="/new"
        className="inline-block px-4 py-2 text-sm font-semibold bg-gold-600 text-white hover:bg-gold-700 rounded-md shadow-sm"
      >
        Start your first analysis
      </Link>
    </div>
  )
}

function ErrorPanel({
  message,
  onRetry,
}: {
  message: string
  onRetry: () => void
}) {
  return (
    <div className="border border-rose-200 bg-rose-50 text-rose-800 rounded-lg p-4">
      <div className="font-semibold mb-1">Could not load analyses</div>
      <div className="text-sm font-mono mb-3 break-all">{message}</div>
      <button
        type="button"
        onClick={onRetry}
        className="px-3 py-1.5 text-xs font-semibold bg-rose-600 text-white hover:bg-rose-700 rounded"
      >
        Retry
      </button>
    </div>
  )
}

function AnalysisTable({ items }: { items: AnalysisSummary[] }) {
  return (
    <div className="bg-white border border-gold-200 rounded-xl overflow-hidden shadow-sm">
      <table className="w-full text-sm">
        <thead className="bg-gold-50 text-gold-700 text-xs uppercase tracking-wide">
          <tr>
            <th className="px-4 py-3 text-left">Ticker</th>
            <th className="px-4 py-3 text-left">Date</th>
            <th className="px-4 py-3 text-left">Type</th>
            <th className="px-4 py-3 text-left">Status</th>
            <th className="px-4 py-3 text-left">Decision</th>
            <th className="px-4 py-3 text-left w-1/4">Progress</th>
            <th className="px-4 py-3 text-right" />
          </tr>
        </thead>
        <tbody className="divide-y divide-gold-100">
          {items.map((a) => (
            <AnalysisRow key={a.id} a={a} />
          ))}
        </tbody>
      </table>
    </div>
  )
}

function AnalysisRow({ a }: { a: AnalysisSummary }) {
  const qc = useQueryClient()
  const [confirming, setConfirming] = useState(false)
  const remove = useMutation({
    mutationFn: () => api.remove(a.id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['analyses'] }),
  })

  return (
    <tr className="hover:bg-gold-50 transition-colors">
      <td className="px-4 py-3 font-mono font-semibold text-gold-900">
        <Link
          to={`/analyses/${a.id}`}
          className="hover:underline hover:text-gold-700"
        >
          {a.ticker}
        </Link>
      </td>
      <td className="px-4 py-3 text-gold-700">{a.analysis_date}</td>
      <td className="px-4 py-3 text-gold-600 capitalize text-xs">
        {a.asset_type}
      </td>
      <td className="px-4 py-3">
        <StatusBadge status={a.status} />
        {a.error && (
          <div className="text-xs text-rose-600 mt-1 max-w-xs truncate" title={a.error}>
            {a.error.split('\n')[0]}
          </div>
        )}
      </td>
      <td className="px-4 py-3">
        <DecisionBadge decision={a.final_decision} />
      </td>
      <td className="px-4 py-3">
        <ProgressBar progress={a.progress} compact />
      </td>
      <td className="px-4 py-3 text-right space-x-2">
        <Link
          to={`/analyses/${a.id}`}
          className="inline-block px-3 py-1.5 text-xs font-semibold border border-gold-300 hover:bg-gold-100 rounded"
        >
          View
        </Link>
        {confirming ? (
          <span className="inline-flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => remove.mutate()}
              disabled={remove.isPending}
              className="px-3 py-1.5 text-xs font-semibold bg-rose-600 hover:bg-rose-700 text-white rounded disabled:opacity-60"
            >
              {remove.isPending ? 'Deleting…' : 'Confirm'}
            </button>
            <button
              type="button"
              onClick={() => setConfirming(false)}
              className="px-2 py-1.5 text-xs text-gold-600 hover:text-gold-900"
            >
              Cancel
            </button>
          </span>
        ) : (
          <button
            type="button"
            onClick={() => setConfirming(true)}
            className="inline-block px-3 py-1.5 text-xs font-semibold border border-rose-200 text-rose-700 hover:bg-rose-50 rounded"
          >
            Delete
          </button>
        )}
      </td>
    </tr>
  )
}
