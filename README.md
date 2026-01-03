# MeetingMind AI

**Automated Meeting Intelligence System** - A full-stack application that transforms recorded meeting videos into structured, searchable notes with speaker attribution, topic summaries, action items, and visual insights using multimodal AI.

---

## Overview

MeetingMind AI processes meeting recordings through a sophisticated 5-phase ML pipeline to extract comprehensive meeting intelligence:

1. **Audio Processing** - Speaker diarization (who spoke when) and speech-to-text transcription
2. **Face Detection & Tracking** - Detecting and tracking participants throughout the video
3. **Speaker-Face Matching** - Linking audio speakers to visual faces for accurate attribution
4. **AI Summarization** - Generating structured meeting notes with topics, action items, and decisions
5. **Visual Intelligence** - Extracting content from slides, charts, screen shares, and whiteboards

The system also includes a **RAG-powered Q&A interface** that allows users to ask natural language questions about their meeting content.

---

## Architecture

```
Frontend (React/Vite)  -->  Backend (FastAPI)  -->  ML Pipeline (5 Phases)
                                  |                        |
                                  v                        v
                            SQLite Database          Vertex AI Gemini
```

### Processing Pipeline

| Phase | Purpose | Technology |
|-------|---------|------------|
| Phase 1 | Audio transcription + speaker diarization | Pyannote.audio 3.1, OpenAI Whisper |
| Phase 2 | Face detection and tracking | YOLOv11m-face, InsightFace (ArcFace) |
| Phase 3 | Speaker-face association | Custom temporal matching algorithm |
| Phase 4 | Meeting summarization | Vertex AI Gemini 2.0 Flash |
| Phase 5 | Visual content extraction | EasyOCR, Gemini Vision |

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| Frontend | React 18, Vite 5, React Router 6 |
| Backend | FastAPI, Uvicorn, SQLAlchemy |
| Audio ML | OpenAI Whisper, Pyannote.audio 3.1 |
| Vision ML | YOLOv11m-face, InsightFace (ArcFace), EasyOCR |
| LLM/VLM | Vertex AI Gemini 2.0 Flash |
| RAG | LangChain, ChromaDB, Vertex AI Embeddings |
| Cloud | Google Cloud Run, Artifact Registry, Vertex AI |
| DevOps | Docker, Docker Compose, Nginx |

---

## Features

- **Speaker Diarization** - Automatically identify and label different speakers
- **Timestamped Transcripts** - Full transcription with clickable timestamps
- **Face-to-Speaker Matching** - Visual identification of who is speaking
- **AI-Generated Summaries** - Executive summary, topics, action items, and key decisions
- **Visual Content Extraction** - OCR and description of slides, charts, and screen shares
- **Conversational Q&A** - Ask questions about meeting content with source citations
- **Speaker Labeling** - Rename speakers and regenerate transcripts
- **Export** - Download transcripts and summaries

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/meetings` | POST | Upload video |
| `/api/meetings` | GET | List meetings |
| `/api/meetings/{id}/status` | GET | Processing status |
| `/api/meetings/{id}/transcript` | GET | Get transcript |
| `/api/meetings/{id}/notes` | GET | Get summary |
| `/api/meetings/{id}/faces` | GET | Get detected faces |
| `/api/meetings/{id}/visual-insights` | GET | Get visual analysis |
| `/api/meetings/{id}/ask` | POST | RAG Q&A |

---

## Project Structure

```
MeetingsAI/
├── src/
│   ├── app/                    # FastAPI application
│   │   ├── main.py             # API endpoints
│   │   ├── rag.py              # RAG Q&A system
│   │   ├── gemini_client.py    # Vertex AI client
│   │   └── settings.py         # Configuration
│   ├── pipeline/               # ML processing phases
│   │   ├── phase1_audio_processing.py
│   │   ├── phase2_face_tracking.py
│   │   ├── phase3_matching.py
│   │   ├── phase4.py
│   │   └── phase5_visual.py
│   ├── frontend/               # React application
│   └── worker/                 # Background task processor
├── docker/                     # Docker configurations
└── requirements.txt
```

---

## Author

**Sankalp Rajeev**
