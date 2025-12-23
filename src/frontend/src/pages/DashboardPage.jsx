// Dashboard Page - List all meetings
import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import './DashboardPage.css'

function DashboardPage() {
    const [meetings, setMeetings] = useState([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        const fetchMeetings = async () => {
            try {
                const res = await fetch('/api/meetings')
                if (res.ok) {
                    const data = await res.json()
                    setMeetings(data.meetings || [])
                }
                setLoading(false)
            } catch (err) {
                setLoading(false)
            }
        }
        fetchMeetings()
    }, [])

    const getStatusColor = (status) => {
        switch (status) {
            case 'READY': return 'status-ready'
            case 'PROCESSING': return 'status-processing'
            case 'FAILED': return 'status-failed'
            default: return ''
        }
    }

    const formatDate = (dateStr) => {
        if (!dateStr) return ''
        return new Date(dateStr).toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        })
    }

    return (
        <div className="dashboard">
            <header className="dashboard-header">
                <Link to="/" className="logo">
                    <div className="logo-mark">M</div>
                    <span className="logo-text">MeetingMind</span>
                </Link>
                <Link to="/" className="upload-btn">Upload New</Link>
            </header>

            <main className="dashboard-main">
                <h1>Your Meetings</h1>

                {loading ? (
                    <div className="loading">Loading meetings...</div>
                ) : meetings.length === 0 ? (
                    <div className="empty-state">
                        <p>No meetings yet.</p>
                        <Link to="/" className="upload-link">Upload your first meeting</Link>
                    </div>
                ) : (
                    <div className="meetings-grid">
                        {meetings.map((meeting) => (
                            <Link
                                key={meeting.meeting_id}
                                to={meeting.status === 'READY'
                                    ? `/meeting/${meeting.meeting_id}`
                                    : `/meeting/${meeting.meeting_id}/processing`}
                                className="meeting-card"
                            >
                                <div className="meeting-info">
                                    <h3>{meeting.title || 'Untitled Meeting'}</h3>
                                    <span className="meeting-date">{formatDate(meeting.created_at)}</span>
                                </div>
                                <div className={`meeting-status ${getStatusColor(meeting.status)}`}>
                                    {meeting.status === 'PROCESSING' && (
                                        <span className="progress">{Math.round((meeting.progress || 0) * 100)}%</span>
                                    )}
                                    <span className="status-label">{meeting.status}</span>
                                </div>
                            </Link>
                        ))}
                    </div>
                )}
            </main>
        </div>
    )
}

export default DashboardPage
