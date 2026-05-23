// Shape mirrors server.api Pydantic models. Update both sides if the
// backend response changes.

export type AssetType = 'stock' | 'crypto' | 'commodity'

export type AnalysisStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'

/** Step keys come from server.storage.PIPELINE_STEPS. */
export type PipelineStep =
  | 'market_analyst'
  | 'sentiment_analyst'
  | 'news_analyst'
  | 'fundamentals_analyst'
  | 'bull_researcher'
  | 'bear_researcher'
  | 'research_manager'
  | 'trader'
  | 'risk_aggressive'
  | 'risk_conservative'
  | 'risk_neutral'
  | 'portfolio_manager'

export type StepStatus = 'pending' | 'running' | 'completed' | 'skipped'

export interface AnalysisSummary {
  id: string
  ticker: string
  asset_type: AssetType
  analysis_date: string
  language: string
  status: AnalysisStatus
  progress: Record<string, StepStatus>
  final_decision: string | null
  error: string | null
  created_at: string
  completed_at: string | null
  // Per-analysis debate-round configuration. Optional because old
  // records persisted before the feature shipped don't have them.
  max_debate_rounds?: number | null
  max_risk_discuss_rounds?: number | null
}

export interface InvestmentDebateState {
  bull_history?: string
  bear_history?: string
  history?: string
  judge_decision?: string
  current_response?: string
  count?: number
}

export interface RiskDebateState {
  aggressive_history?: string
  conservative_history?: string
  neutral_history?: string
  history?: string
  judge_decision?: string
  latest_speaker?: string
  count?: number
}

export interface AnalysisReports {
  market_report?: string
  sentiment_report?: string
  news_report?: string
  fundamentals_report?: string
  investment_plan?: string
  trader_investment_plan?: string
  final_trade_decision?: string
  investment_debate_state?: InvestmentDebateState
  risk_debate_state?: RiskDebateState
}

export interface AnalysisDetail extends AnalysisSummary {
  reports: AnalysisReports
}

// ---------------------------------------------------------------------------
// Backtest / track-record
// ---------------------------------------------------------------------------

export type ForwardDirection = 'up' | 'down' | 'flat' | 'unknown'

export interface HorizonOutcome {
  horizon: string         // "1d" / "5d" / "21d" / "63d"
  days: number
  target_date: string     // YYYY-MM-DD calendar date
  end_close: number | null
  forward_return: number | null  // decimal, e.g. 0.0123 = +1.23%
  actual_direction: ForwardDirection
  correct: boolean | null  // null until the target date elapses
}

export interface AnalysisOutcome {
  decision: string
  expected_direction: 'up' | 'down' | 'flat' | null
  start_close: number | null
  computed_at: string
  horizons: HorizonOutcome[]
}

/** Per-decision counters within one horizon's aggregate. */
export interface DecisionStats {
  total: number
  correct: number
  hit_rate: number | null
}

/** One horizon's aggregate row in the track-record summary. */
export interface HorizonStats {
  total: number
  correct: number
  hit_rate: number | null
  by_decision: Record<string, DecisionStats>
}

export interface TrackRecord {
  total_completed: number
  total_with_outcomes: number
  horizons: Record<string, HorizonStats>
  computed_at: string
}

export interface CreateAnalysisRequest {
  ticker: string
  analysis_date: string
  language?: string
  /** Bull/Bear debate rounds. Server clamps to 1-10; default 1. */
  max_debate_rounds?: number
  /** Aggressive/Conservative/Neutral risk-debate rounds. 1-10; default 1. */
  max_risk_discuss_rounds?: number
}

// ---------------------------------------------------------------------------
// Schedules — auto-fire analyses on a cadence
// ---------------------------------------------------------------------------

export type ScheduleKind = 'daily_after_close' | 'volatility_trigger'

/** Params block — shape varies by kind, kept as a loose record so
 * the form UI can pass through whatever the user typed. The backend
 * clamps + validates on every read. */
export interface ScheduleParams {
  // daily_after_close
  fire_hour_utc?: number
  fire_minute_utc?: number
  weekdays_only?: boolean
  // volatility_trigger
  threshold_pct?: number
  throttle_hours?: number
  check_interval_minutes?: number
}

export interface Schedule {
  id: string
  name: string
  ticker: string
  asset_type: AssetType
  kind: ScheduleKind
  params: ScheduleParams
  language: string
  max_debate_rounds: number
  max_risk_discuss_rounds: number
  enabled: boolean
  last_run_at: string | null
  last_run_analysis_id: string | null
  last_check_at: string | null
  created_at: string
}

export interface CreateScheduleRequest {
  ticker: string
  kind: ScheduleKind
  name?: string
  language?: string
  max_debate_rounds?: number
  max_risk_discuss_rounds?: number
  params?: ScheduleParams
  enabled?: boolean
}

export interface UpdateScheduleRequest {
  name?: string
  enabled?: boolean
  language?: string
  max_debate_rounds?: number
  max_risk_discuss_rounds?: number
  params?: ScheduleParams
}
