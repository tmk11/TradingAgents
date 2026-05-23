// Thin API client for the FastAPI backend. All endpoints live under
// ``/api`` so the same code works in dev (Vite proxy) and prod
// (FastAPI serving the built bundle on the same origin).

import type {
  AnalysisDetail,
  AnalysisSummary,
  CreateAnalysisRequest,
} from './types'

const BASE = '/api'

async function request<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(init.headers ?? {}),
    },
    ...init,
  })

  if (!res.ok) {
    // Surface FastAPI/Pydantic validation errors verbatim so the form
    // can show "ticker must be ..." style messages instead of a
    // generic "request failed".
    let detail: string
    try {
      const body = await res.json()
      detail =
        typeof body.detail === 'string'
          ? body.detail
          : JSON.stringify(body.detail ?? body)
    } catch {
      detail = await res.text()
    }
    throw new Error(`${res.status}: ${detail}`)
  }

  if (res.status === 204) {
    return undefined as T
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => request<{ status: string }>('/health'),

  list: () => request<AnalysisSummary[]>('/analyses'),

  get: (id: string) => request<AnalysisDetail>(`/analyses/${id}`),

  create: (payload: CreateAnalysisRequest) =>
    request<AnalysisSummary>('/analyses', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  remove: (id: string) =>
    request<void>(`/analyses/${id}`, { method: 'DELETE' }),
}
