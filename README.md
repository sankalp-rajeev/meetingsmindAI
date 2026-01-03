# 🎯 MeetingMind AI

**Automated Meeting Intelligence System** - Transform recorded meeting videos into structured, searchable notes with speaker attribution, topic summaries, action items, and visual insights using multimodal AI.

[![Live Demo](https://img.shields.io/badge/Live-Demo-green)](https://meetingmind-frontend-669045652498.us-central1.run.app)
[![Backend](https://img.shields.io/badge/API-Healthy-brightgreen)](https://meetingmind-backend-669045652498.us-central1.run.app/health)

---

## ✨ Features

- **🎤 Speaker Diarization** - Automatically identify who spoke when using Pyannote.audio
- **📝 Speech-to-Text** - Transcribe audio using OpenAI Whisper
- **👤 Face Detection & Tracking** - Detect and track faces with YOLOv11 + InsightFace
- **🔗 Speaker-Face Matching** - Link audio speakers to visual faces
- **📊 Visual Intelligence** - Extract content from slides, charts, and screen shares
- **📋 AI Summarization** - Generate meeting notes with Gemini 2.0 Flash
- **💬 RAG Q&A** - Ask questions about your meeting with conversational AI
- **📤 Export** - Download transcripts and summaries

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   React/Vite    │────▶│    FastAPI      │────▶│   ML Pipeline   │
│    Frontend     │     │    Backend      │     │   (5 Phases)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │                        │
                               ▼                        ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │     SQLite      │     │   Vertex AI     │
                        │    Database     │     │    Gemini       │
                        └─────────────────┘     └─────────────────┘
```

### 5-Phase Processing Pipeline

| Phase | Purpose | Technology |
|-------|---------|------------|
| **Phase 1** | Audio Processing | Pyannote 3.1 + Whisper |
| **Phase 2** | Face Detection | YOLOv11m-face + InsightFace |
| **Phase 3** | Speaker-Face Matching | Custom temporal algorithm |
| **Phase 4** | Summarization | Gemini 2.0 Flash |
| **Phase 5** | Visual Intelligence | EasyOCR + Gemini Vision |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (optional)
- Google Cloud account (for Vertex AI)

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/MeetingsAI.git
cd MeetingsAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your HUGGINGFACE_TOKEN and GCP credentials

# Start the backend
uvicorn src.app.main:app --reload --port 8000

# In another terminal, start the frontend
cd src/frontend
npm install
npm run dev
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose -f docker/docker-compose.yml up --build
```

---

## 🔧 Configuration

### Required Environment Variables

```env
# HuggingFace (for Pyannote models)
HUGGINGFACE_TOKEN=hf_xxxxx

# Google Cloud (for Vertex AI Gemini)
GCP_PROJECT=your-project-id
GCP_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

---

## 📁 Project Structure

```
MeetingsAI/
├── src/
│   ├── app/               # FastAPI application
│   │   ├── main.py        # API endpoints
│   │   ├── rag.py         # RAG Q&A system
│   │   ├── gemini_client.py  # Vertex AI client
│   │   └── settings.py    # Configuration
│   ├── pipeline/          # ML processing phases
│   │   ├── phase1_audio_processing.py
│   │   ├── phase2_face_tracking.py
│   │   ├── phase3_matching.py
│   │   ├── phase4.py
│   │   └── phase5_visual.py
│   ├── frontend/          # React application
│   │   └── src/
│   │       ├── pages/     # LandingPage, ProcessingPage, ResultsPage
│   │       └── components/
│   └── worker/            # Background task processor
├── docker/                # Docker configurations
│   ├── Dockerfile.cloudrun.backend
│   ├── Dockerfile.cloudrun.frontend
│   └── docker-compose.yml
├── requirements.txt
└── PROJECT_SUMMARY.md     # Detailed technical documentation
```

---

## 🌐 Cloud Deployment

### Google Cloud Run

```bash
# Build the backend image
docker build -f docker/Dockerfile.cloudrun.backend -t meetingsmind-backend:cpu .

# Push to Artifact Registry
docker tag meetingsmind-backend:cpu us-central1-docker.pkg.dev/PROJECT/REPO/meetingsmind-backend:cpu
docker push us-central1-docker.pkg.dev/PROJECT/REPO/meetingsmind-backend:cpu

# Deploy to Cloud Run
gcloud run deploy meetingmind-backend \
  --image=us-central1-docker.pkg.dev/PROJECT/REPO/meetingsmind-backend:cpu \
  --region=us-central1 \
  --memory=8Gi --cpu=4 --timeout=3600
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Frontend** | React 18, Vite 5, React Router 6 |
| **Backend** | FastAPI, Uvicorn, SQLAlchemy |
| **Audio ML** | OpenAI Whisper, Pyannote.audio 3.1 |
| **Vision ML** | YOLOv11m-face, InsightFace, EasyOCR |
| **LLM/VLM** | Vertex AI Gemini 2.0 Flash |
| **RAG** | LangChain, ChromaDB |
| **Cloud** | Google Cloud Run, Artifact Registry, Vertex AI |

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/meetings` | POST | Upload video |
| `/api/meetings` | GET | List meetings |
| `/api/meetings/{id}/status` | GET | Processing status |
| `/api/meetings/{id}/transcript` | GET | Get transcript |
| `/api/meetings/{id}/notes` | GET | Get summary |
| `/api/meetings/{id}/ask` | POST | RAG Q&A |

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

**Sankalp Rajeev**

---

*Built with ❤️ using multimodal AI*
