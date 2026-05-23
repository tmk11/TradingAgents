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

export interface CreateAnalysisRequest {
  ticker: string
  analysis_date: string
  language?: string
  /** Bull/Bear debate rounds. Server clamps to 1-10; default 1. */
  max_debate_rounds?: number
  /** Aggressive/Conservative/Neutral risk-debate rounds. 1-10; default 1. */
  max_risk_discuss_rounds?: number
}
