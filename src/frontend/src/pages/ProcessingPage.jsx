// Processing Page - Real-time Status Polling
import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import './ProcessingPage.css'

const PHASES = [
    { id: 'PHASE1', label: 'Audio Processing', description: 'Extracting audio and transcribing speech' },
    { id: 'PHASE2', label: 'Visual Analysis', description: 'Detecting and tracking faces' },
    { id: 'PHASE3', label: 'Speaker Matching', description: 'Associating voices with faces' },
    { id: 'PHASE4', label: 'Intelligence', description: 'Generating summary and action items' },
]

function ProcessingPage() {
    const { id } = useParams()
    const navigate = useNavigate()
    const [status, setStatus] = useState(null)
    const [error, setError] = useState(null)

    useEffect(() => {
        const pollStatus = async () => {
            try {
                const res = await fetch(`/api/meetings/${id}/status`)
                if (!res.ok) throw new Error('Failed to fetch status')

                const data = await res.json()
                setStatus(data)

                if (data.status === 'READY') {
                    navigate(`/meeting/${id}`)
                } else if (data.status === 'FAILED') {
                    setError(data.error_message || 'Processing failed')
                }
            } catch (err) {
                setError('Unable to fetch status')
            }
        }

        pollStatus()
        const interval = setInterval(pollStatus, 2000)
        return () => clearInterval(interval)
    }, [id, navigate])

    const getCurrentPhaseIndex = () => {
        if (!status?.phase) return 0
        const idx = PHASES.findIndex(p => p.id === status.phase)
        return idx >= 0 ? idx : 0
    }

    const currentPhaseIndex = getCurrentPhaseIndex()
    const progress = status?.progress || 0

    return (
        <div className="processing">
            <header className="processing-header">
                <div className="logo">
                    <div className="logo-mark">M</div>
                    <span className="logo-text">MeetingMind</span>
                </div>
            </header>

            <main className="processing-main">
                <div className="processing-card">
                    {error ? (
                        <div className="processing-error">
                            <div className="error-icon">
                                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                    <circle cx="12" cy="12" r="10" />
                                    <line x1="15" y1="9" x2="9" y2="15" />
                                    <line x1="9" y1="9" x2="15" y2="15" />
                                </svg>
                            </div>
                            <h2>Processing Failed</h2>
                            <p>{error}</p>
                            <button className="retry-button" onClick={() => navigate('/')}>
                                Try Again
                            </button>
                        </div>
                    ) : (
                        <>
                            <div className="processing-header-content">
                                <h1>Analyzing your meeting</h1>
                                <p>This may take a few minutes depending on the video length</p>
                            </div>

                            <div className="progress-bar-container">
                                <div className="progress-bar">
                                    <div
                                        className="progress-fill"
                                        style={{ width: `${Math.max(5, progress * 100)}%` }}
                                    />
                                </div>
                                <span className="progress-text">{Math.round(progress * 100)}%</span>
                            </div>

                            <div className="phases">
                                {PHASES.map((phase, index) => {
                                    let phaseStatus = 'pending'
                                    if (index < currentPhaseIndex) phaseStatus = 'complete'
                                    else if (index === currentPhaseIndex) phaseStatus = 'active'

                                    return (
                                        <div key={phase.id} className={`phase ${phaseStatus}`}>
                                            <div className="phase-indicator">
                                                {phaseStatus === 'complete' ? (
                                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                        <polyline points="20 6 9 17 4 12" />
                                                    </svg>
                                                ) : phaseStatus === 'active' ? (
                                                    <div className="phase-spinner" />
                                                ) : (
                                                    <span>{index + 1}</span>
                                                )}
                                            </div>
                                            <div className="phase-content">
                                                <h3>{phase.label}</h3>
                                                <p>{phase.description}</p>
                                            </div>
                                        </div>
                                    )
                                })}
                            </div>

                            {status?.message && (
                                <div className="status-message">
                                    {status.message}
                                </div>
                            )}
                        </>
                    )}
                </div>
            </main>
        </div>
    )
}

export default ProcessingPage
