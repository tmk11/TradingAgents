import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'

import { api } from '../api'
import type { CreateAnalysisRequest } from '../types'

// Keep the example list in sync with cli/utils.GOLD_TICKERS.
const TICKER_PRESETS = [
  { ticker: 'GLD', label: 'SPDR Gold ETF' },
  { ticker: 'IAU', label: 'iShares Gold Trust' },
  { ticker: 'GC=F', label: 'COMEX Gold Futures' },
  { ticker: 'XAUUSD=X', label: 'Spot XAU/USD' },
  { ticker: 'GDX', label: 'Gold Miners ETF' },
  { ticker: 'GDXJ', label: 'Junior Gold Miners ETF' },
]

const LANGUAGES = [
  'English',
  'Vietnamese',
  'Chinese',
  'Japanese',
  'Korean',
  'French',
  'Spanish',
  'German',
]

function todayIsoDate(): string {
  const d = new Date()
  return d.toISOString().slice(0, 10)
}

export function NewAnalysisPage() {
  const navigate = useNavigate()
  const qc = useQueryClient()
  const [form, setForm] = useState<CreateAnalysisRequest>({
    ticker: 'GLD',
    analysis_date: todayIsoDate(),
    language: 'English',
  })
  const create = useMutation({
    mutationFn: (payload: CreateAnalysisRequest) => api.create(payload),
    onSuccess: (rec) => {
      qc.invalidateQueries({ queryKey: ['analyses'] })
      navigate(`/analyses/${rec.id}`)
    },
  })

  function update<K extends keyof CreateAnalysisRequest>(
    key: K,
    value: CreateAnalysisRequest[K],
  ) {
    setForm((f) => ({ ...f, [key]: value }))
  }

  return (
    <div className="max-w-2xl mx-auto">
      <Link
        to="/"
        className="inline-flex items-center gap-1 text-sm text-gold-700 hover:text-gold-900 mb-4"
      >
        ← Back to dashboard
      </Link>
      <h2 className="text-2xl font-bold text-gold-900 mb-1">
        New analysis
      </h2>
      <p className="text-sm text-gold-600 mb-6">
        The pipeline runs in the background — you can leave this page once
        it's queued. Most analyses complete in 3–10 minutes depending on
        your LLM provider.
      </p>

      <form
        className="bg-white border border-gold-200 rounded-xl p-6 space-y-5 shadow-sm"
        onSubmit={(e) => {
          e.preventDefault()
          create.mutate(form)
        }}
      >
        {/* Ticker */}
        <div>
          <label
            htmlFor="ticker"
            className="block text-sm font-semibold text-gold-800 mb-1"
          >
            Ticker
          </label>
          <input
            id="ticker"
            type="text"
            required
            autoFocus
            value={form.ticker}
            onChange={(e) => update('ticker', e.target.value.toUpperCase())}
            className="w-full px-3 py-2 border border-gold-300 rounded-md font-mono uppercase focus:outline-none focus:ring-2 focus:ring-gold-500 focus:border-gold-500"
            placeholder="GLD"
            maxLength={32}
          />
          <p className="text-xs text-gold-500 mt-1">
            Gold-complex tickers auto-detected; stock and crypto tickers
            still work.
          </p>
          <div className="flex flex-wrap gap-1.5 mt-2">
            {TICKER_PRESETS.map((p) => (
              <button
                key={p.ticker}
                type="button"
                onClick={() => update('ticker', p.ticker)}
                title={p.label}
                className={`text-xs font-mono px-2 py-1 rounded border transition-colors ${
                  form.ticker === p.ticker
                    ? 'bg-gold-600 text-white border-gold-600'
                    : 'border-gold-200 hover:bg-gold-100'
                }`}
              >
                {p.ticker}
              </button>
            ))}
          </div>
        </div>

        {/* Date */}
        <div>
          <label
            htmlFor="analysis_date"
            className="block text-sm font-semibold text-gold-800 mb-1"
          >
            Analysis date
          </label>
          <input
            id="analysis_date"
            type="date"
            required
            value={form.analysis_date}
            max={todayIsoDate()}
            onChange={(e) => update('analysis_date', e.target.value)}
            className="w-full px-3 py-2 border border-gold-300 rounded-md focus:outline-none focus:ring-2 focus:ring-gold-500 focus:border-gold-500"
          />
        </div>

        {/* Language */}
        <div>
          <label
            htmlFor="language"
            className="block text-sm font-semibold text-gold-800 mb-1"
          >
            Output language
          </label>
          <select
            id="language"
            value={form.language ?? 'English'}
            onChange={(e) => update('language', e.target.value)}
            className="w-full px-3 py-2 border border-gold-300 rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-gold-500 focus:border-gold-500"
          >
            {LANGUAGES.map((l) => (
              <option key={l} value={l}>
                {l}
              </option>
            ))}
          </select>
          <p className="text-xs text-gold-500 mt-1">
            Internal agent debate stays in English for reasoning quality;
            this controls the user-facing report language.
          </p>
        </div>

        {/* Errors */}
        {create.error && (
          <div className="border border-rose-200 bg-rose-50 text-rose-800 rounded-md p-3 text-sm">
            <div className="font-semibold mb-0.5">Failed to start</div>
            <div className="font-mono text-xs break-all">
              {(create.error as Error).message}
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center justify-end gap-2 pt-2 border-t border-gold-100">
          <Link
            to="/"
            className="px-4 py-2 text-sm text-gold-700 hover:text-gold-900"
          >
            Cancel
          </Link>
          <button
            type="submit"
            disabled={create.isPending}
            className="px-5 py-2 text-sm font-semibold bg-gold-600 text-white hover:bg-gold-700 rounded-md shadow-sm disabled:opacity-60"
          >
            {create.isPending ? 'Queuing…' : 'Start analysis'}
          </button>
        </div>
      </form>
    </div>
  )
}
