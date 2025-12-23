// Landing Page - Professional Upload Interface
import { useState, useCallback } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import './LandingPage.css'

// Professional SVG Icons
const UploadIcon = () => (
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="17 8 12 3 7 8" />
        <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
)

const MicIcon = () => (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
        <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
        <line x1="12" y1="19" x2="12" y2="23" />
        <line x1="8" y1="23" x2="16" y2="23" />
    </svg>
)

const UserIcon = () => (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
        <circle cx="12" cy="7" r="4" />
    </svg>
)

const FileTextIcon = () => (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
        <polyline points="14 2 14 8 20 8" />
        <line x1="16" y1="13" x2="8" y2="13" />
        <line x1="16" y1="17" x2="8" y2="17" />
        <polyline points="10 9 9 9 8 9" />
    </svg>
)

function LandingPage() {
    const [isDragging, setIsDragging] = useState(false)
    const [isUploading, setIsUploading] = useState(false)
    const [error, setError] = useState(null)
    const navigate = useNavigate()

    const handleDragOver = useCallback((e) => {
        e.preventDefault()
        setIsDragging(true)
    }, [])

    const handleDragLeave = useCallback((e) => {
        e.preventDefault()
        setIsDragging(false)
    }, [])

    const handleDrop = useCallback((e) => {
        e.preventDefault()
        setIsDragging(false)
        const file = e.dataTransfer.files[0]
        if (file) handleUpload(file)
    }, [])

    const handleFileSelect = (e) => {
        const file = e.target.files[0]
        if (file) handleUpload(file)
    }

    const handleUpload = async (file) => {
        if (!file.name.toLowerCase().endsWith('.mp4')) {
            setError('Please upload an MP4 file')
            return
        }

        setIsUploading(true)
        setError(null)

        const formData = new FormData()
        formData.append('file', file)

        try {
            const res = await fetch('/api/meetings', {
                method: 'POST',
                body: formData,
            })

            if (!res.ok) throw new Error('Upload failed')

            const data = await res.json()
            navigate(`/meeting/${data.meeting_id}/processing`)
        } catch (err) {
            setError('Upload failed. Please try again.')
            setIsUploading(false)
        }
    }

    return (
        <div className="landing">
            <header className="landing-header">
                <div className="logo">
                    <div className="logo-mark">M</div>
                    <span className="logo-text">MeetingMind</span>
                </div>
                <Link to="/dashboard" className="nav-link">My Meetings</Link>
            </header>

            <main className="landing-main">
                <div className="hero">
                    <h1 className="hero-title">
                        Transform meetings into
                        <span className="gradient-text"> actionable insights</span>
                    </h1>
                    <p className="hero-subtitle">
                        AI-powered transcription, speaker identification, and intelligent summaries.
                        Upload your meeting recording to get started.
                    </p>
                </div>

                <div
                    className={`dropzone ${isDragging ? 'dragging' : ''} ${isUploading ? 'uploading' : ''}`}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                >
                    {isUploading ? (
                        <div className="upload-progress">
                            <div className="spinner"></div>
                            <p>Processing upload...</p>
                        </div>
                    ) : (
                        <>
                            <div className="dropzone-icon">
                                <UploadIcon />
                            </div>
                            <p className="dropzone-text">
                                Drag and drop your meeting video
                            </p>
                            <p className="dropzone-hint">or</p>
                            <label className="upload-button">
                                Choose File
                                <input
                                    type="file"
                                    accept=".mp4,video/mp4"
                                    onChange={handleFileSelect}
                                    hidden
                                />
                            </label>
                            <p className="dropzone-format">MP4 format supported</p>
                        </>
                    )}
                </div>

                {error && <p className="error-message">{error}</p>}

                <div className="features">
                    <div className="feature">
                        <div className="feature-icon">
                            <MicIcon />
                        </div>
                        <h3>Speaker Detection</h3>
                        <p>Automatically identify and attribute speech to individual participants</p>
                    </div>
                    <div className="feature">
                        <div className="feature-icon">
                            <UserIcon />
                        </div>
                        <h3>Face Recognition</h3>
                        <p>Visual identification and tracking of meeting participants</p>
                    </div>
                    <div className="feature">
                        <div className="feature-icon">
                            <FileTextIcon />
                        </div>
                        <h3>Intelligent Summary</h3>
                        <p>Extract key decisions, action items, and discussion points</p>
                    </div>
                </div>
            </main>

            <footer className="landing-footer">
                <p>MeetingMind AI — Intelligent Meeting Analysis</p>
            </footer>
        </div>
    )
}

export default LandingPage
