// Results Page - Meeting Analysis Display with Labeling & Export
import { useState, useEffect, useRef } from 'react'
import { useParams, Link } from 'react-router-dom'
import './ResultsPage.css'

// Icons
const PlayIcon = () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
        <polygon points="5 3 19 12 5 21 5 3" />
    </svg>
)

const PauseIcon = () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
        <rect x="6" y="4" width="4" height="16" />
        <rect x="14" y="4" width="4" height="16" />
    </svg>
)

const DownloadIcon = () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="7 10 12 15 17 10" />
        <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
)

function ResultsPage() {
    const { id } = useParams()
    const videoRef = useRef(null)
    const transcriptRef = useRef(null)
    const activeSegmentRef = useRef(null)
    const [isPlaying, setIsPlaying] = useState(false)
    const [currentTime, setCurrentTime] = useState(0)
    const [duration, setDuration] = useState(0)
    const [activeSpeaker, setActiveSpeaker] = useState(null)

    const [transcript, setTranscript] = useState(null)
    const [notes, setNotes] = useState(null)
    const [faces, setFaces] = useState(null)
    const [visualInsights, setVisualInsights] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)

    // Speaker labeling state
    const [speakerLabels, setSpeakerLabels] = useState({})
    const [speakerToFace, setSpeakerToFace] = useState({})
    const [faceToSpeaker, setFaceToSpeaker] = useState({})
    const [showLabelModal, setShowLabelModal] = useState(false)
    const [editingSpeaker, setEditingSpeaker] = useState(null)
    const [editingFace, setEditingFace] = useState(null)
    const [labelInput, setLabelInput] = useState('')
    const [savingLabel, setSavingLabel] = useState(false)
    const [showToast, setShowToast] = useState(false)
    const [toastMessage, setToastMessage] = useState('')
    const [showAllVisuals, setShowAllVisuals] = useState(false)

    // Chat/RAG state
    const [chatMessages, setChatMessages] = useState([])
    const [chatInput, setChatInput] = useState('')
    const [chatLoading, setChatLoading] = useState(false)
    const [showChat, setShowChat] = useState(false)
    const chatEndRef = useRef(null)

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [transcriptRes, notesRes, facesRes, labelsRes] = await Promise.all([
                    fetch(`/api/meetings/${id}/transcript`),
                    fetch(`/api/meetings/${id}/notes`),
                    fetch(`/api/meetings/${id}/faces`),
                    fetch(`/api/meetings/${id}/speaker-labels`),
                ])

                let existingLabels = {}
                let existingSpeakerToFace = {}

                if (labelsRes.ok) {
                    const labelsData = await labelsRes.json()
                    existingLabels = labelsData.labels || {}
                    existingSpeakerToFace = labelsData.speaker_to_face || {}
                }

                // Also fetch Phase 3 auto-detected suggestions
                const suggestionsRes = await fetch(`/api/meetings/${id}/speaker-suggestions`)
                if (suggestionsRes.ok) {
                    const suggestionsData = await suggestionsRes.json()
                    // Merge: user labels override auto-suggestions
                    existingSpeakerToFace = {
                        ...suggestionsData.speaker_to_face,
                        ...existingSpeakerToFace
                    }
                }

                setSpeakerToFace(existingSpeakerToFace)
                // Build reverse mapping (face -> speaker)
                const reversemap = {}
                Object.entries(existingSpeakerToFace).forEach(([sp, face]) => {
                    reversemap[face] = sp
                })
                setFaceToSpeaker(reversemap)

                if (transcriptRes.ok) {
                    const transcriptData = await transcriptRes.json()
                    setTranscript(transcriptData)
                    // Extract unique speakers and merge with existing labels
                    const speakers = [...new Set(transcriptData?.segments?.map(s => s.speaker) || [])]
                    const labels = {}
                    speakers.forEach(s => {
                        labels[s] = existingLabels[s] || s
                    })
                    setSpeakerLabels(labels)
                }
                if (notesRes.ok) {
                    const notesData = await notesRes.json()
                    // Normalize keys to lowercase (LLM may output Title vs title)
                    const normalizedNotes = {}
                    Object.entries(notesData).forEach(([key, value]) => {
                        normalizedNotes[key.toLowerCase()] = value
                    })
                    setNotes(normalizedNotes)
                }
                if (facesRes.ok) setFaces(await facesRes.json())

                // Fetch visual insights (Phase 5)
                try {
                    const visualRes = await fetch(`/api/meetings/${id}/visual-insights`)
                    if (visualRes.ok) {
                        const visualData = await visualRes.json()
                        if (visualData.keyframes_analyzed > 0) {
                            setVisualInsights(visualData)
                        }
                    }
                } catch (e) {
                    console.log('No visual insights available')
                }

                setLoading(false)
            } catch (err) {
                setError('Failed to load meeting data')
                setLoading(false)
            }
        }

        fetchData()
    }, [id])

    const formatTime = (seconds) => {
        if (typeof seconds !== 'number' || isNaN(seconds)) return '0:00'
        const mins = Math.floor(seconds / 60)
        const secs = Math.floor(seconds % 60)
        return `${mins}:${secs.toString().padStart(2, '0')}`
    }

    const parseTime = (timeStr) => {
        if (typeof timeStr === 'number') return timeStr
        if (!timeStr) return 0
        const parts = timeStr.split(':').map(Number)
        if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if (parts.length === 2) return parts[0] * 60 + parts[1]
        return 0
    }

    const handlePlayPause = () => {
        if (videoRef.current) {
            if (isPlaying) {
                videoRef.current.pause()
            } else {
                videoRef.current.play()
            }
            setIsPlaying(!isPlaying)
        }
    }

    const handleTimeUpdate = () => {
        if (videoRef.current) {
            const time = videoRef.current.currentTime
            setCurrentTime(time)

            // Find active segment and speaker
            if (transcript?.segments) {
                const activeIdx = transcript.segments.findIndex(seg => {
                    const start = parseTime(seg.start)
                    const end = parseTime(seg.end)
                    return time >= start && time < end
                })

                if (activeIdx >= 0) {
                    const seg = transcript.segments[activeIdx]
                    setActiveSpeaker(seg.speaker)

                    // Auto-scroll to active segment
                    if (isPlaying && activeSegmentRef.current) {
                        activeSegmentRef.current.scrollIntoView({
                            behavior: 'smooth',
                            block: 'center'
                        })
                    }
                } else {
                    setActiveSpeaker(null)
                }
            }
        }
    }

    const handleLoadedMetadata = () => {
        if (videoRef.current) {
            setDuration(videoRef.current.duration)
        }
    }

    const seekTo = (time) => {
        const t = parseTime(time)
        if (videoRef.current) {
            videoRef.current.currentTime = t
            setCurrentTime(t)
        }
    }

    const handleProgressClick = (e) => {
        const rect = e.currentTarget.getBoundingClientRect()
        const percent = (e.clientX - rect.left) / rect.width
        seekTo(percent * duration)
    }

    // Speaker labeling functions
    const openLabelModal = (speaker, faceTrackId = null) => {
        setEditingSpeaker(speaker)
        setEditingFace(faceTrackId)
        setLabelInput(speakerLabels[speaker] || speaker)
        setShowLabelModal(true)
    }

    const openFaceLabelModal = (trackId) => {
        // Find which speaker is linked to this face, or use track_id as key
        const linkedSpeaker = Object.entries(speakerToFace).find(([sp, face]) => face === trackId)?.[0]
        setEditingSpeaker(linkedSpeaker || trackId)
        setEditingFace(trackId)
        setLabelInput(speakerLabels[linkedSpeaker || trackId] || trackId.replace('track_', 'Person '))
        setShowLabelModal(true)
    }

    const saveLabel = async () => {
        if (!labelInput.trim()) return

        const key = editingSpeaker || editingFace
        if (!key) return

        setSavingLabel(true)

        // Update local state
        const newLabels = {
            ...speakerLabels,
            [key]: labelInput.trim()
        }
        setSpeakerLabels(newLabels)

        // Persist to backend
        try {
            await fetch(`/api/meetings/${id}/speaker-labels`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ labels: { [key]: labelInput.trim() } })
            })
        } catch (err) {
            console.error('Failed to save label:', err)
        }

        setSavingLabel(false)
        setShowLabelModal(false)
        setEditingSpeaker(null)
        setEditingFace(null)

        // Show toast
        setToastMessage('Label saved!')
        setShowToast(true)
        setTimeout(() => setShowToast(false), 2000)
    }

    const getDisplaySpeaker = (speaker) => {
        return speakerLabels[speaker] || speaker
    }

    // Calculate speaking time for a speaker
    const getSpeakingTime = (speakerId) => {
        if (!transcript?.segments) return 0
        return transcript.segments
            .filter(s => s.speaker === speakerId)
            .reduce((total, seg) => {
                const start = parseTime(seg.start)
                const end = parseTime(seg.end)
                return total + (end - start)
            }, 0)
    }

    // Get display name for a face track
    const getFaceDisplayName = (trackId) => {
        // Check if this track is linked to a speaker with a custom label
        const linkedSpeaker = Object.entries(speakerToFace).find(([sp, face]) => face === trackId)?.[0]
        if (linkedSpeaker && speakerLabels[linkedSpeaker]) {
            return speakerLabels[linkedSpeaker]
        }
        // Check if track itself has a label
        if (speakerLabels[trackId]) {
            return speakerLabels[trackId]
        }
        return trackId.replace('track_', 'Person ')
    }

    // Export functions
    const exportTranscript = () => {
        if (!transcript?.segments) return

        let text = `Meeting Transcript\n${'='.repeat(50)}\n\n`
        transcript.segments.forEach(seg => {
            const speaker = getDisplaySpeaker(seg.speaker)
            const time = formatTime(parseTime(seg.start))
            text += `[${time}] ${speaker}:\n${seg.text}\n\n`
        })

        downloadFile(text, `transcript_${id.slice(0, 8)}.txt`, 'text/plain')
    }

    const exportSummary = () => {
        if (!notes) return

        let text = `Meeting Summary\n${'='.repeat(50)}\n\n`
        if (notes.title) text += `Title: ${notes.title}\n\n`
        if (notes.summary) text += `Summary:\n${notes.summary}\n\n`
        if (notes.decisions?.length) {
            text += `Key Decisions:\n`
            notes.decisions.forEach((d, i) => { text += `${i + 1}. ${d}\n` })
            text += '\n'
        }
        if (notes.action_items?.length) {
            text += `Action Items:\n`
            notes.action_items.forEach((a, i) => { text += `${i + 1}. [${a.owner}] ${a.item}\n` })
        }

        downloadFile(text, `summary_${id.slice(0, 8)}.txt`, 'text/plain')
    }

    const downloadFile = (content, filename, type) => {
        const blob = new Blob([content], { type })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = filename
        a.click()
        URL.revokeObjectURL(url)
    }

    // Chat/RAG functions
    const sendMessage = async (e) => {
        e.preventDefault()
        if (!chatInput.trim() || chatLoading) return

        const userMessage = chatInput.trim()
        setChatInput('')
        setChatMessages(prev => [...prev, { role: 'user', content: userMessage }])
        setChatLoading(true)

        try {
            const response = await fetch(`/api/meetings/${id}/ask`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: userMessage })
            })

            if (!response.ok) throw new Error('Failed to get response')

            const data = await response.json()
            setChatMessages(prev => [...prev, {
                role: 'assistant',
                content: data.answer,
                sources: data.sources || []
            }])
        } catch (err) {
            setChatMessages(prev => [...prev, {
                role: 'assistant',
                content: 'Sorry, I encountered an error. Please try again.',
                error: true
            }])
        } finally {
            setChatLoading(false)
            // Scroll to bottom
            setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 100)
        }
    }

    const clearChat = async () => {
        try {
            await fetch(`/api/meetings/${id}/chat`, { method: 'DELETE' })
        } catch (e) { /* ignore */ }
        setChatMessages([])
    }

    const formatTimestamp = (timestamp) => {
        // Handle string timestamps (e.g., "02:08.76" from RAG sources)
        if (typeof timestamp === 'string') {
            // If it's already formatted like "MM:SS.ss", extract just MM:SS
            const match = timestamp.match(/^(\d+):(\d+)/)
            if (match) return `${match[1]}:${match[2].padStart(2, '0')}`
            return timestamp
        }
        // Handle numeric seconds
        if (typeof timestamp !== 'number' || isNaN(timestamp)) return '0:00'
        const mins = Math.floor(timestamp / 60)
        const secs = Math.floor(timestamp % 60)
        return `${mins}:${secs.toString().padStart(2, '0')}`
    }

    if (loading) {
        return (
            <div className="results loading-state">
                <div className="spinner-large"></div>
                <p>Loading meeting data...</p>
            </div>
        )
    }

    if (error) {
        return (
            <div className="results error-state">
                <p>{error}</p>
                <Link to="/" className="back-link">Return home</Link>
            </div>
        )
    }

    return (
        <div className="results">
            <header className="results-header">
                <Link to="/" className="logo">
                    <div className="logo-mark">M</div>
                    <span className="logo-text">MeetingMind</span>
                </Link>
                <div className="header-actions">
                    <Link to="/dashboard" className="nav-link">All Meetings</Link>
                    <div className="export-buttons">
                        <button onClick={exportTranscript} className="export-btn" title="Export Transcript">
                            <DownloadIcon /> Transcript
                        </button>
                        <button onClick={exportSummary} className="export-btn" title="Export Summary">
                            <DownloadIcon /> Summary
                        </button>
                    </div>
                </div>
            </header>

            <main className="results-main">
                {/* Left Panel - Video */}
                <section className="panel video-panel">
                    <div className="video-container">
                        <video
                            ref={videoRef}
                            src={`/data/meetings/${id}/original.mp4`}
                            onTimeUpdate={handleTimeUpdate}
                            onLoadedMetadata={handleLoadedMetadata}
                            onEnded={() => setIsPlaying(false)}
                        />
                    </div>
                    <div className="video-controls">
                        <button className="play-button" onClick={handlePlayPause}>
                            {isPlaying ? <PauseIcon /> : <PlayIcon />}
                        </button>
                        <span className="time-current">{formatTime(currentTime)}</span>
                        <div className="progress-track" onClick={handleProgressClick}>
                            <div
                                className="progress-fill"
                                style={{ width: `${duration ? (currentTime / duration) * 100 : 0}%` }}
                            />
                        </div>
                        <span className="time-duration">{formatTime(duration)}</span>
                    </div>

                    {/* Face Gallery with Labeling */}
                    {faces?.tracks && faces.tracks.length > 0 && (
                        <div className="faces-section">
                            <h3>Participants ({faces.tracks.length})</h3>
                            <div className="faces-grid">
                                {faces.tracks.slice(0, 6).map((track) => {
                                    const displayName = getFaceDisplayName(track.track_id)
                                    const isNamed = displayName !== track.track_id.replace('track_', 'Person ')
                                    const linkedSpeaker = faceToSpeaker[track.track_id]
                                    const isActiveSpeaker = linkedSpeaker && activeSpeaker === linkedSpeaker
                                    return (
                                        <div
                                            key={track.track_id}
                                            className={`face-card ${isNamed ? 'named' : 'unnamed'} ${isActiveSpeaker ? 'speaking' : ''}`}
                                            onClick={() => openFaceLabelModal(track.track_id)}
                                            title="Click to name this participant"
                                        >
                                            <div className="face-image-wrapper">
                                                <img
                                                    src={`/data/meetings/${id}/${track.keyframe?.path}`}
                                                    alt={track.track_id}
                                                    onError={(e) => e.target.style.display = 'none'}
                                                />
                                                <div className="face-edit-overlay">
                                                    <span>Edit</span>
                                                </div>
                                                {isActiveSpeaker && (
                                                    <div className="speaking-indicator">
                                                        <span className="pulse"></span>
                                                    </div>
                                                )}
                                            </div>
                                            <span className="face-name">{displayName}</span>
                                            <span className="face-appearances">
                                                {track.num_appearances} appearances
                                            </span>
                                        </div>
                                    )
                                })}
                            </div>
                        </div>
                    )}

                    {/* Speaker Labels */}
                    {Object.keys(speakerLabels).length > 0 && (
                        <div className="speakers-section">
                            <h3>Speakers</h3>
                            <div className="speakers-list">
                                {Object.entries(speakerLabels).map(([original, label]) => (
                                    <button
                                        key={original}
                                        className="speaker-tag"
                                        onClick={() => openLabelModal(original)}
                                    >
                                        {label}
                                        <span className="edit-hint">Edit</span>
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}
                </section>

                {/* Center Panel - Transcript */}
                <section className="panel transcript-panel">
                    <h2>Transcript</h2>
                    <div className="transcript-content" ref={transcriptRef}>
                        {transcript?.segments?.map((segment, index) => {
                            const startSec = parseTime(segment.start)
                            const endSec = parseTime(segment.end)
                            const isActive = currentTime >= startSec && currentTime < endSec
                            const faceTrackId = speakerToFace[segment.speaker]
                            const faceKeyframe = faceTrackId && faces?.tracks?.find(t => t.track_id === faceTrackId)?.keyframe?.path

                            return (
                                <div
                                    key={index}
                                    ref={isActive ? activeSegmentRef : null}
                                    className={`segment ${isActive ? 'active' : ''}`}
                                    onClick={() => seekTo(startSec)}
                                >
                                    <div className="segment-meta">
                                        <div className="speaker-with-face">
                                            {faceKeyframe && (
                                                <img
                                                    className="speaker-face-thumb"
                                                    src={`/data/meetings/${id}/${faceKeyframe}`}
                                                    alt=""
                                                    onError={(e) => e.target.style.display = 'none'}
                                                />
                                            )}
                                            <span className="speaker" onClick={(e) => { e.stopPropagation(); openLabelModal(segment.speaker) }}>
                                                {getDisplaySpeaker(segment.speaker)}
                                            </span>
                                        </div>
                                        <span className="timestamp">{formatTime(startSec)}</span>
                                    </div>
                                    <p className="segment-text">{segment.text}</p>
                                </div>
                            )
                        })}
                    </div>
                </section>

                {/* Right Panel - Summary */}
                <section className="panel summary-panel">
                    <h2>Meeting Intelligence</h2>

                    {notes?.title && (
                        <div className="summary-section">
                            <h3>{typeof notes.title === 'string' ? notes.title : 'Meeting Notes'}</h3>
                        </div>
                    )}

                    {notes?.summary && (
                        <div className="summary-section">
                            <h4>Summary</h4>
                            {typeof notes.summary === 'string' ? (
                                <p>{notes.summary}</p>
                            ) : (
                                <>
                                    {/* Overview */}
                                    {notes.summary.overview && (
                                        <div className="summary-subsection overview-section">
                                            <p className="overview-text">{notes.summary.overview}</p>
                                        </div>
                                    )}

                                    {/* Visual Content Summary */}
                                    {notes.summary.visual_content_summary && (
                                        <div className="summary-subsection">
                                            <h5>🖥️ Presentations & Visual Content</h5>
                                            <p>{notes.summary.visual_content_summary}</p>
                                        </div>
                                    )}

                                    {/* Discussion Topics */}
                                    {notes.summary.discussion_topics && notes.summary.discussion_topics.length > 0 && (
                                        <div className="summary-subsection">
                                            <h5>📋 Discussion Topics</h5>
                                            <ul>
                                                {notes.summary.discussion_topics.map((topic, i) => (
                                                    <li key={i}>{topic}</li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                    {/* Key Points */}
                                    {notes.summary.key_points && notes.summary.key_points.length > 0 && (
                                        <div className="summary-subsection">
                                            <h5>💡 Key Points</h5>
                                            <ul>
                                                {notes.summary.key_points.map((point, i) => (
                                                    <li key={i}>{typeof point === 'object' ? point.point || JSON.stringify(point) : point}</li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                    {/* Decisions Made */}
                                    {notes.summary.decisions_made && notes.summary.decisions_made.length > 0 && (
                                        <div className="summary-subsection">
                                            <h5>✅ Decisions Made</h5>
                                            <ul>
                                                {notes.summary.decisions_made.map((decision, i) => (
                                                    <li key={i}>{typeof decision === 'object' ? decision.decision || JSON.stringify(decision) : decision}</li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                    {/* Action Items from summary */}
                                    {notes.summary.action_items && notes.summary.action_items.length > 0 && (
                                        <div className="summary-subsection">
                                            <h5>📝 Action Items</h5>
                                            <ul>
                                                {notes.summary.action_items.map((item, i) => (
                                                    <li key={i}>
                                                        {typeof item === 'object'
                                                            ? `${item.owner ? `[${item.owner}] ` : ''}${item.item || item.action || JSON.stringify(item)}`
                                                            : item}
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                    {/* Risks Identified */}
                                    {notes.summary.risks_identified && notes.summary.risks_identified.length > 0 && (
                                        <div className="summary-subsection">
                                            <h5>⚠️ Risks Identified</h5>
                                            <ul className="risks-list">
                                                {notes.summary.risks_identified.map((risk, i) => (
                                                    <li key={i}>
                                                        {typeof risk === 'object'
                                                            ? risk.risk || risk.description || JSON.stringify(risk)
                                                            : risk}
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                    {/* Open Questions */}
                                    {notes.summary.open_questions && notes.summary.open_questions.length > 0 && (
                                        <div className="summary-subsection">
                                            <h5>❓ Open Questions</h5>
                                            <ul>
                                                {notes.summary.open_questions.map((question, i) => (
                                                    <li key={i}>
                                                        {typeof question === 'object'
                                                            ? question.question || JSON.stringify(question)
                                                            : question}
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                    {/* Next Steps */}
                                    {notes.summary.next_steps && (
                                        <div className="summary-subsection next-steps-section">
                                            <h5>➡️ Next Steps</h5>
                                            <p>{notes.summary.next_steps}</p>
                                        </div>
                                    )}
                                    {/* Fallback for keypoints format */}
                                    {(notes.summary.keypoints || notes.summary.KeyPoints) && (
                                        <ul>
                                            {(notes.summary.keypoints || notes.summary.KeyPoints).map((point, i) => (
                                                <li key={i}>{typeof point === 'string' ? point : JSON.stringify(point)}</li>
                                            ))}
                                        </ul>
                                    )}
                                </>
                            )}
                        </div>
                    )}

                    {notes?.decisions && notes.decisions.length > 0 && (
                        <div className="summary-section">
                            <h4>Key Decisions</h4>
                            <ul>
                                {notes.decisions.map((decision, i) => (
                                    <li key={i}>
                                        {typeof decision === 'object'
                                            ? decision.decision || decision.item || decision.text || JSON.stringify(decision)
                                            : decision}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {notes?.action_items && notes.action_items.length > 0 && (
                        <div className="summary-section">
                            <h4>Action Items</h4>
                            <div className="action-items">
                                {notes.action_items.map((item, i) => (
                                    <div key={i} className="action-item">
                                        <span className="owner">
                                            {typeof item === 'object' ? item.owner || 'Unassigned' : ''}
                                        </span>
                                        <p>
                                            {typeof item === 'object'
                                                ? item.item || item.action || item.text || JSON.stringify(item)
                                                : item}
                                        </p>
                                        {typeof item === 'object' && item.due_date && (
                                            <span className="due-date">Due: {item.due_date}</span>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {notes?.risks && notes.risks.length > 0 && (
                        <div className="summary-section">
                            <h4>Risks</h4>
                            <ul className="risks-list">
                                {notes.risks.map((risk, i) => (
                                    <li key={i}>
                                        {typeof risk === 'object'
                                            ? risk.risk || risk.description || JSON.stringify(risk)
                                            : risk}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {/* Visual Insights from Phase 5 */}
                    {visualInsights && visualInsights.insights && visualInsights.insights.length > 0 && (() => {
                        // Filter out camera_only and duplicate frames (skipped by content filter)
                        const contentFrames = visualInsights.insights.filter(
                            insight => !['camera_only', 'duplicate', 'unknown'].includes(insight.content_type) && !insight.skipped_vlm
                        );
                        const skippedCount = visualInsights.insights.length - contentFrames.length;

                        return contentFrames.length > 0 ? (
                            <div className="summary-section visual-insights-section">
                                <div className="visual-insights-header">
                                    <h4>📊 Visual Content ({contentFrames.length} slides/charts)</h4>
                                    {skippedCount > 0 && (
                                        <span className="skipped-count" title="Camera-only frames were filtered out">
                                            {skippedCount} camera frames filtered
                                        </span>
                                    )}
                                    <button
                                        className="expand-btn"
                                        onClick={() => setShowAllVisuals(!showAllVisuals)}
                                    >
                                        {showAllVisuals ? 'Show less' : 'Show all'}
                                    </button>
                                </div>
                                <div className="visual-insights-grid">
                                    {contentFrames
                                        .slice(0, showAllVisuals ? 25 : 6)
                                        .map((insight, i) => (
                                            <div
                                                key={i}
                                                className={`visual-insight-card ${insight.content_type}`}
                                                onClick={() => seekTo(insight.timestamp)}
                                                title={insight.description}
                                            >
                                                <img
                                                    src={`/data/meetings/${id}/${insight.frame_path}`}
                                                    alt={insight.description}
                                                    onError={(e) => e.target.style.display = 'none'}
                                                />
                                                <div className="visual-insight-info">
                                                    <span className="visual-time">{insight.timestamp_formatted}</span>
                                                    <span className={`visual-type ${insight.content_type}`}>{insight.content_type}</span>
                                                </div>
                                                {insight.chart_analysis && (
                                                    <span className="chart-badge">📊</span>
                                                )}
                                                {insight.extracted_text && Array.isArray(insight.extracted_text) && insight.extracted_text.length > 0 && (
                                                    <div className="extracted-text-hint">
                                                        {insight.extracted_text.slice(0, 2).join(', ')}
                                                    </div>
                                                )}
                                            </div>
                                        ))}
                                </div>
                            </div>
                        ) : null;
                    })()}
                </section>
            </main>

            {/* Label Modal */}
            {showLabelModal && (
                <div className="modal-overlay" onClick={() => setShowLabelModal(false)}>
                    <div className="modal" onClick={e => e.stopPropagation()}>
                        <h3>Rename Speaker</h3>
                        <p className="modal-subtitle">Original: {editingSpeaker}</p>
                        <input
                            type="text"
                            value={labelInput}
                            onChange={e => setLabelInput(e.target.value)}
                            placeholder="Enter speaker name"
                            autoFocus
                            onKeyDown={e => e.key === 'Enter' && saveLabel()}
                        />
                        <div className="modal-actions">
                            <button onClick={() => setShowLabelModal(false)} className="btn-cancel">Cancel</button>
                            <button onClick={saveLabel} className="btn-save" disabled={savingLabel}>
                                {savingLabel ? 'Saving...' : 'Save'}
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Chat Panel */}
            <div className={`chat-panel ${showChat ? 'open' : ''}`}>
                <div className="chat-header" onClick={() => setShowChat(!showChat)}>
                    <span>💬 Ask about this meeting</span>
                    <button className="chat-toggle">{showChat ? '▼' : '▲'}</button>
                </div>
                {showChat && (
                    <div className="chat-body">
                        <div className="chat-messages">
                            {chatMessages.length === 0 && (
                                <div className="chat-empty">
                                    Ask any question about this meeting. Examples:
                                    <ul>
                                        <li>"What were the main decisions?"</li>
                                        <li>"Who mentioned the budget?"</li>
                                        <li>"What was shown on the slides?"</li>
                                    </ul>
                                </div>
                            )}
                            {chatMessages.map((msg, i) => (
                                <div key={i} className={`chat-message ${msg.role}`}>
                                    <div className="message-content">{msg.content}</div>
                                    {msg.sources && msg.sources.length > 0 && (
                                        <div className="message-sources">
                                            <span className="sources-label">Sources:</span>
                                            {msg.sources.map((src, j) => (
                                                <span key={j} className="source-tag"
                                                    onClick={() => src.timestamp && seekTo(src.timestamp)}>
                                                    {src.type === 'transcript' && src.timestamp !== undefined
                                                        ? `📝 ${formatTimestamp(src.timestamp)}`
                                                        : src.type === 'visual' && src.timestamp !== undefined
                                                            ? `🖼️ ${formatTimestamp(src.timestamp)}`
                                                            : `📄 ${src.section || src.type}`}
                                                </span>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            ))}
                            {chatLoading && (
                                <div className="chat-message assistant loading">
                                    <div className="typing-indicator">
                                        <span></span><span></span><span></span>
                                    </div>
                                </div>
                            )}
                            <div ref={chatEndRef} />
                        </div>
                        <form className="chat-input-form" onSubmit={sendMessage}>
                            <input
                                type="text"
                                value={chatInput}
                                onChange={e => setChatInput(e.target.value)}
                                placeholder="Ask a question..."
                                disabled={chatLoading}
                            />
                            <button type="submit" disabled={chatLoading || !chatInput.trim()}>
                                Send
                            </button>
                            {chatMessages.length > 0 && (
                                <button type="button" className="clear-btn" onClick={clearChat}>
                                    Clear
                                </button>
                            )}
                        </form>
                    </div>
                )}
            </div>

            {/* Toast Notification */}
            {showToast && (
                <div className="toast">
                    {toastMessage}
                </div>
            )}
        </div>
    )
}

export default ResultsPage
