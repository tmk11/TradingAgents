import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { Link } from 'react-router-dom'

import { api } from '../api'
import type {
  CreateScheduleRequest,
  Schedule,
  ScheduleKind,
} from '../types'

// Mirrors server.schedules.SCHEDULE_KINDS — keep these in sync.
const KIND_OPTIONS: ReadonlyArray<{
  value: ScheduleKind
  label: string
  blurb: string
}> = [
  {
    value: 'daily_after_close',
    label: 'Daily after US close',
    blurb:
      'Fires once per weekday at 21:30 UTC (≈ 16:30 ET / 04:30 VN next morning).',
  },
  {
    value: 'volatility_trigger',
    label: 'Volatility trigger',
    blurb:
      'Fires when |intraday move| ≥ threshold vs prior close. Throttled to one fire per N hours.',
  },
]

export function SchedulesPage() {
  const qc = useQueryClient()
  const list = useQuery({
    queryKey: ['schedules'],
    queryFn: () => api.listSchedules(),
    refetchInterval: 30_000,
  })

  const seed = useMutation({
    mutationFn: () => api.seedRecommendedSchedules('GLD', 'English'),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['schedules'] }),
  })

  const empty = !list.isLoading && (list.data?.length ?? 0) === 0

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
          Schedules
        </h2>
        <p className="text-sm text-gold-600 max-w-2xl">
          Auto-create analyses on a cadence so you build a track record
          without remembering to click "New analysis" daily. Two kinds
          today: a daily run after US close, and a volatility-triggered
          run for big intraday gold moves.
        </p>
      </div>

      <ManualEventReminder />

      {list.isLoading && (
        <div className="text-center py-12 text-gold-600">Loading…</div>
      )}

      {list.error && (
        <div className="border border-rose-200 bg-rose-50 text-rose-800 rounded-lg p-4">
          <div className="font-semibold mb-1">Could not load schedules</div>
          <div className="font-mono text-xs break-all">
            {(list.error as Error).message}
          </div>
        </div>
      )}

      {empty && (
        <EmptyState
          onSeed={() => seed.mutate()}
          loading={seed.isPending}
          error={seed.error as Error | null}
        />
      )}

      {!empty && list.data && (
        <>
          <ScheduleList schedules={list.data} />
          <CreateForm />
        </>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------

function ManualEventReminder() {
  return (
    <div className="border border-amber-200 bg-amber-50 rounded-lg p-4 text-sm text-amber-900">
      <div className="font-semibold mb-1">FOMC / CPI / NFP events</div>
      <p className="leading-relaxed">
        We don't auto-detect economic releases (no good free calendar
        API). After a FOMC decision, CPI / NFP print, or any other
        major event, click <span className="font-mono">Run now</span>
        {' '}on a schedule below to fire an extra analysis on demand.
      </p>
    </div>
  )
}

// ---------------------------------------------------------------------------

function EmptyState({
  onSeed,
  loading,
  error,
}: {
  onSeed: () => void
  loading: boolean
  error: Error | null
}) {
  return (
    <div className="bg-white border border-gold-200 rounded-xl p-8 shadow-sm text-center">
      <div className="text-lg font-semibold text-gold-900 mb-1">
        No schedules yet
      </div>
      <p className="text-sm text-gold-600 mb-4 max-w-md mx-auto">
        Get going with the recommended setup: <strong>daily after US close</strong>
        {' '}plus a <strong>≥1.5% volatility trigger</strong>, both on GLD at
        Medium research depth.
      </p>
      <button
        type="button"
        onClick={onSeed}
        disabled={loading}
        className="px-5 py-2 text-sm font-semibold bg-gold-600 text-white hover:bg-gold-700 rounded-md shadow-sm disabled:opacity-60"
      >
        {loading ? 'Creating…' : 'Apply recommended defaults'}
      </button>
      {error && (
        <div className="text-xs text-rose-700 mt-3 font-mono break-all">
          {error.message}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------

function ScheduleList({ schedules }: { schedules: Schedule[] }) {
  return (
    <section>
      <h3 className="text-sm font-semibold text-gold-700 uppercase tracking-wide mb-3">
        Active schedules
      </h3>
      <div className="space-y-3">
        {schedules.map((s) => (
          <ScheduleCard key={s.id} schedule={s} />
        ))}
      </div>
    </section>
  )
}

function ScheduleCard({ schedule }: { schedule: Schedule }) {
  const qc = useQueryClient()

  const toggle = useMutation({
    mutationFn: () =>
      api.updateSchedule(schedule.id, { enabled: !schedule.enabled }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['schedules'] }),
  })

  const remove = useMutation({
    mutationFn: () => api.deleteSchedule(schedule.id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['schedules'] }),
  })

  const runNow = useMutation({
    mutationFn: () => api.runScheduleNow(schedule.id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['schedules'] })
      qc.invalidateQueries({ queryKey: ['analyses'] })
    },
  })

  return (
    <div
      className={`bg-white border rounded-xl p-4 shadow-sm ${
        schedule.enabled ? 'border-gold-200' : 'border-gold-100 opacity-70'
      }`}
    >
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-mono font-semibold text-gold-900">
              {schedule.ticker}
            </span>
            <KindBadge kind={schedule.kind} />
            {!schedule.enabled && (
              <span className="text-xs px-1.5 py-0.5 rounded bg-gold-100 text-gold-700">
                Disabled
              </span>
            )}
          </div>
          <div className="text-sm text-gold-700 mt-0.5">{schedule.name}</div>
          <div className="text-xs text-gold-500 mt-1">
            <ScheduleSummary schedule={schedule} />
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => runNow.mutate()}
            disabled={runNow.isPending}
            className="px-2.5 py-1.5 text-xs font-semibold bg-gold-600 text-white hover:bg-gold-700 rounded disabled:opacity-60"
            title="Manually trigger one analysis right now"
          >
            {runNow.isPending ? 'Queuing…' : 'Run now'}
          </button>
          <button
            type="button"
            onClick={() => toggle.mutate()}
            disabled={toggle.isPending}
            className="px-2.5 py-1.5 text-xs border border-gold-200 hover:bg-gold-50 rounded disabled:opacity-60"
          >
            {schedule.enabled ? 'Disable' : 'Enable'}
          </button>
          <button
            type="button"
            onClick={() => {
              if (confirm(`Delete schedule "${schedule.name}"?`)) {
                remove.mutate()
              }
            }}
            disabled={remove.isPending}
            className="px-2.5 py-1.5 text-xs border border-rose-200 text-rose-700 hover:bg-rose-50 rounded disabled:opacity-60"
          >
            Delete
          </button>
        </div>
      </div>

      {(toggle.error || remove.error || runNow.error) && (
        <div className="text-xs text-rose-700 mt-2 font-mono break-all">
          {(
            (toggle.error || remove.error || runNow.error) as Error
          )?.message}
        </div>
      )}

      {runNow.data && (
        <div className="text-xs text-emerald-700 mt-2">
          ✓ Queued —{' '}
          <Link
            to={`/analyses/${runNow.data.id}`}
            className="underline font-mono"
          >
            view progress
          </Link>
        </div>
      )}
    </div>
  )
}

function KindBadge({ kind }: { kind: ScheduleKind }) {
  const label =
    kind === 'daily_after_close' ? 'Daily' : 'Volatility'
  const tone =
    kind === 'daily_after_close'
      ? 'bg-blue-100 text-blue-800'
      : 'bg-purple-100 text-purple-800'
  return (
    <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${tone}`}>
      {label}
    </span>
  )
}

function ScheduleSummary({ schedule }: { schedule: Schedule }) {
  const lastRun = schedule.last_run_at
    ? `last ran ${formatRelative(schedule.last_run_at)}`
    : 'never run yet'

  if (schedule.kind === 'daily_after_close') {
    const h = String(schedule.params.fire_hour_utc ?? 21).padStart(2, '0')
    const m = String(schedule.params.fire_minute_utc ?? 30).padStart(2, '0')
    const days = schedule.params.weekdays_only ? 'Mon–Fri' : 'every day'
    return (
      <>
        Fires {days} at {h}:{m} UTC · depth {schedule.max_debate_rounds}
        {' '}rounds · {lastRun}
      </>
    )
  }
  if (schedule.kind === 'volatility_trigger') {
    return (
      <>
        Fires when |move| ≥ {schedule.params.threshold_pct ?? 1.5}% · throttle{' '}
        {schedule.params.throttle_hours ?? 6}h · poll every{' '}
        {schedule.params.check_interval_minutes ?? 15}m · {lastRun}
      </>
    )
  }
  return null
}

function formatRelative(iso: string): string {
  try {
    const d = new Date(iso)
    const diffMs = Date.now() - d.getTime()
    const mins = Math.floor(diffMs / 60_000)
    if (mins < 1) return 'just now'
    if (mins < 60) return `${mins}m ago`
    const hours = Math.floor(mins / 60)
    if (hours < 24) return `${hours}h ago`
    const days = Math.floor(hours / 24)
    return `${days}d ago`
  } catch {
    return iso
  }
}

// ---------------------------------------------------------------------------
// Create form (collapsed by default to keep the page calm)
// ---------------------------------------------------------------------------

function CreateForm() {
  const qc = useQueryClient()
  const [open, setOpen] = useState(false)
  const [form, setForm] = useState<CreateScheduleRequest>({
    ticker: 'GLD',
    kind: 'daily_after_close',
    max_debate_rounds: 3,
    max_risk_discuss_rounds: 3,
    params: {},
  })

  const create = useMutation({
    mutationFn: (payload: CreateScheduleRequest) => api.createSchedule(payload),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['schedules'] })
      setOpen(false)
      setForm({
        ticker: 'GLD',
        kind: 'daily_after_close',
        max_debate_rounds: 3,
        max_risk_discuss_rounds: 3,
        params: {},
      })
    },
  })

  if (!open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="w-full py-3 border-2 border-dashed border-gold-200 rounded-xl text-sm text-gold-700 hover:bg-gold-50 hover:border-gold-300"
      >
        + Add another schedule
      </button>
    )
  }

  return (
    <form
      className="bg-white border border-gold-200 rounded-xl p-5 shadow-sm space-y-4"
      onSubmit={(e) => {
        e.preventDefault()
        create.mutate(form)
      }}
    >
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gold-900 uppercase tracking-wide">
          New schedule
        </h3>
        <button
          type="button"
          onClick={() => setOpen(false)}
          className="text-xs text-gold-500 hover:text-gold-800"
        >
          Cancel
        </button>
      </div>

      {/* Ticker */}
      <div>
        <label className="block text-xs font-semibold text-gold-700 mb-1">
          Ticker
        </label>
        <input
          type="text"
          required
          value={form.ticker}
          onChange={(e) =>
            setForm({ ...form, ticker: e.target.value.toUpperCase() })
          }
          className="w-full px-3 py-2 border border-gold-300 rounded-md font-mono uppercase focus:outline-none focus:ring-2 focus:ring-gold-500"
          maxLength={32}
        />
      </div>

      {/* Kind */}
      <div className="space-y-2">
        <label className="block text-xs font-semibold text-gold-700">
          Schedule kind
        </label>
        {KIND_OPTIONS.map((opt) => (
          <label
            key={opt.value}
            className={`flex items-start gap-3 p-3 border rounded-md cursor-pointer ${
              form.kind === opt.value
                ? 'border-gold-500 bg-gold-50'
                : 'border-gold-200 hover:bg-gold-50/50'
            }`}
          >
            <input
              type="radio"
              name="kind"
              value={opt.value}
              checked={form.kind === opt.value}
              onChange={() => setForm({ ...form, kind: opt.value, params: {} })}
              className="mt-0.5"
            />
            <span>
              <span className="block font-medium text-sm text-gold-900">
                {opt.label}
              </span>
              <span className="block text-xs text-gold-600">{opt.blurb}</span>
            </span>
          </label>
        ))}
      </div>

      {/* Volatility-specific params */}
      {form.kind === 'volatility_trigger' && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <NumField
            label="Threshold (%)"
            min={0.1}
            max={10}
            step={0.1}
            value={form.params?.threshold_pct ?? 1.5}
            onChange={(v) =>
              setForm({
                ...form,
                params: { ...form.params, threshold_pct: v },
              })
            }
          />
          <NumField
            label="Throttle (hours)"
            min={1}
            max={48}
            step={1}
            value={form.params?.throttle_hours ?? 6}
            onChange={(v) =>
              setForm({
                ...form,
                params: { ...form.params, throttle_hours: v },
              })
            }
          />
          <NumField
            label="Poll every (min)"
            min={5}
            max={120}
            step={5}
            value={form.params?.check_interval_minutes ?? 15}
            onChange={(v) =>
              setForm({
                ...form,
                params: { ...form.params, check_interval_minutes: v },
              })
            }
          />
        </div>
      )}

      {/* Research depth */}
      <div className="grid grid-cols-2 gap-3">
        <NumField
          label="Bull/Bear rounds"
          min={1}
          max={10}
          step={1}
          value={form.max_debate_rounds ?? 3}
          onChange={(v) => setForm({ ...form, max_debate_rounds: v })}
        />
        <NumField
          label="Risk-debate rounds"
          min={1}
          max={10}
          step={1}
          value={form.max_risk_discuss_rounds ?? 3}
          onChange={(v) => setForm({ ...form, max_risk_discuss_rounds: v })}
        />
      </div>

      {create.error && (
        <div className="border border-rose-200 bg-rose-50 text-rose-800 rounded-md p-3 text-sm">
          <div className="font-semibold mb-0.5">Failed to create</div>
          <div className="font-mono text-xs break-all">
            {(create.error as Error).message}
          </div>
        </div>
      )}

      <div className="flex justify-end">
        <button
          type="submit"
          disabled={create.isPending}
          className="px-5 py-2 text-sm font-semibold bg-gold-600 text-white hover:bg-gold-700 rounded-md shadow-sm disabled:opacity-60"
        >
          {create.isPending ? 'Creating…' : 'Create schedule'}
        </button>
      </div>
    </form>
  )
}

function NumField({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
}) {
  return (
    <label className="block">
      <span className="block text-xs font-semibold text-gold-700 mb-1">
        {label}
      </span>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => {
          const n = Number(e.target.value)
          if (Number.isFinite(n)) onChange(n)
        }}
        className="w-full px-3 py-2 border border-gold-300 rounded-md focus:outline-none focus:ring-2 focus:ring-gold-500"
      />
    </label>
  )
}
