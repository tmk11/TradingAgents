import { Link, Route, Routes } from 'react-router-dom'

import { AnalysisDetailPage } from './components/AnalysisDetail'
import { AnalysisListPage } from './components/AnalysisList'
import { NewAnalysisPage } from './components/AnalysisForm'
import { TrackRecordPage } from './components/TrackRecord'

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1 max-w-6xl mx-auto w-full px-4 sm:px-6 py-8">
        <Routes>
          <Route path="/" element={<AnalysisListPage />} />
          <Route path="/new" element={<NewAnalysisPage />} />
          <Route path="/track-record" element={<TrackRecordPage />} />
          <Route path="/analyses/:id" element={<AnalysisDetailPage />} />
          <Route
            path="*"
            element={
              <div className="text-center py-20">
                <h2 className="text-2xl font-bold mb-2">Page not found</h2>
                <Link to="/" className="text-gold-700 underline">
                  Back to dashboard
                </Link>
              </div>
            }
          />
        </Routes>
      </main>
      <Footer />
    </div>
  )
}

function Header() {
  return (
    <header className="border-b border-gold-200 bg-white shadow-sm">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-4 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-gold-300 to-gold-600 flex items-center justify-center text-gold-900 font-bold text-lg shadow-inner">
            Au
          </div>
          <div>
            <h1 className="text-xl font-bold text-gold-900 leading-tight">
              TradingAgents
            </h1>
            <p className="text-xs text-gold-600 -mt-0.5">
              Gold Edition · Multi-agent commodity analysis
            </p>
          </div>
        </Link>
        <nav className="flex items-center gap-2">
          <Link
            to="/"
            className="px-3 py-2 text-sm font-medium text-gold-700 hover:text-gold-900 hover:bg-gold-50 rounded-md"
          >
            Dashboard
          </Link>
          <Link
            to="/track-record"
            className="px-3 py-2 text-sm font-medium text-gold-700 hover:text-gold-900 hover:bg-gold-50 rounded-md"
          >
            Track record
          </Link>
          <Link
            to="/new"
            className="px-3 py-2 text-sm font-semibold bg-gold-600 text-white hover:bg-gold-700 rounded-md shadow-sm"
          >
            + New analysis
          </Link>
        </nav>
      </div>
    </header>
  )
}

function Footer() {
  return (
    <footer className="border-t border-gold-200 py-4 mt-8">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 text-xs text-gold-600 flex flex-wrap items-center justify-between gap-2">
        <div>
          Built on{' '}
          <a
            href="https://github.com/TauricResearch/TradingAgents"
            className="underline hover:text-gold-800"
            target="_blank"
            rel="noreferrer"
          >
            TradingAgents
          </a>{' '}
          · Gold-complex fork
        </div>
        <div className="italic">
          Research tool — not investment advice.
        </div>
      </div>
    </footer>
  )
}
