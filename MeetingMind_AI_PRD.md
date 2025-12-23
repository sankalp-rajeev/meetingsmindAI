# 📋 Product Requirements Document (PRD)

## **MeetingMind AI - Automated Meeting Intelligence System**

---

## 📑 Document Overview

| Field | Value |
|-------|-------|
| **Product** | MeetingMind AI |
| **Version** | 1.0 MVP |
| **Author** | Sankalp |
| **Date** | December 2024 |
| **Timeline** | 8 weeks (MVP) |

---

## 🎯 Product Vision

Transform recorded meeting videos into structured, searchable notes with speaker attribution, topic summaries, and action items using multimodal AI.

---

## 📊 Core Features (MVP)

1. **Audio Processing**: Speaker diarization + speech-to-text
2. **Speaker Identification**: Face detection and manual labeling
3. **Meeting Summarization**: Topic extraction and action items
4. **Structured Output**: Timestamped transcript with speaker names

---

## 🏗️ Development Phases

---

## **PHASE 1: Audio Pipeline** (Week 1-2)

### **Goal**
Extract and transcribe audio with speaker segmentation.

### **Features**
- Extract audio from video
- Identify "who spoke when" (speaker diarization)
- Convert speech to text with timestamps
- Merge diarization + transcription

### **Input/Output**
```
Input:  meeting_video.mp4 (1 hour)
Output: transcript.json
{
  "segments": [
    {
      "start": "00:00",
      "end": "02:46",
      "speaker": "SPEAKER_0",
      "text": "Thanks everyone for joining..."
    },
    {
      "start": "02:47",
      "end": "05:10",
      "speaker": "SPEAKER_1",
      "text": "The model accuracy improved to 92%..."
    }
  ]
}
```

### **Tech Stack**
| Component | Technology | Why |
|-----------|-----------|-----|
| Audio extraction | FFmpeg | Fast, standard tool |
| Speaker diarization | pyannote.audio 3.0 | Best open-source option |
| Speech-to-text | OpenAI Whisper large-v3 | High accuracy, multilingual |
| Backend | Python 3.10+ | ML ecosystem |

### **Models Required**
| Model | Size | Download |
|-------|------|----------|
| pyannote/speaker-diarization-3.1 | 300 MB | HuggingFace |
| openai/whisper-large-v3 | 3 GB | HuggingFace |

### **Dataset/Credentials**
- **pyannote**: Requires HuggingFace token (free, requires accepting terms)
- **Whisper**: No credentials needed

### **Installation**
```bash
pip install pyannote.audio openai-whisper ffmpeg-python torch
```

### **Processing Time (1hr meeting)**
- Audio extraction: 10 seconds
- Diarization: 2-3 minutes
- Transcription: 3-5 minutes
- **Total: ~5-8 minutes**

### **Success Criteria**
- ✅ Accurate speaker segmentation (90%+ correct boundaries)
- ✅ Transcription WER < 10%
- ✅ Speaker labels consistent throughout meeting
- ✅ Timestamps accurate to ±1 second

### **Deliverable**
Working script that takes video, outputs JSON with speaker-attributed transcript.

---

## **PHASE 2: Visual Pipeline** (Week 3-4)

### **Goal**
Detect and track faces throughout the meeting.

### **Features**
- Extract frames from video
- Detect faces in frames
- Track same person across frames
- Extract representative face images per person
- Generate face embeddings

### **Input/Output**
```
Input:  meeting_video.mp4
Output: faces.json
{
  "tracks": [
    {
      "track_id": "track_1",
      "appearances": [
        {"timestamp": "00:00", "frame": 0, "bbox": [x,y,w,h]},
        {"timestamp": "00:03", "frame": 90, "bbox": [x,y,w,h]}
      ],
      "keyframe": "track_1_keyframe.jpg",
      "embedding": [512-dim vector]
    },
    {
      "track_id": "track_2",
      ...
    }
  ]
}
```

### **Tech Stack**
| Component | Technology | Why |
|-----------|-----------|-----|
| Face detection | YOLOv8-face or RetinaFace | Fast, accurate |
| Face tracking | ByteTrack | Robust tracking |
| Face embedding | InsightFace (ArcFace) | Industry standard |
| Frame extraction | OpenCV | Standard library |

### **Models Required**
| Model | Size | Download |
|-------|------|----------|
| yolov8n-face or RetinaFace | 50-100 MB | GitHub/HuggingFace |
| InsightFace buffalo_l | 400 MB | InsightFace |

### **Dataset/Credentials**
- No credentials required
- Models are open-source

### **Installation**
```bash
pip install insightface opencv-python ultralytics
```

### **Processing Time (1hr meeting)**
- Frame sampling (2 FPS): 10 seconds
- Face detection: 2-3 minutes
- Face tracking: 1-2 minutes
- Embedding extraction: 30 seconds
- **Total: ~4-6 minutes**

### **Success Criteria**
- ✅ Detect all visible faces (95%+ recall)
- ✅ Track consistency (same person = same track_id)
- ✅ Quality keyframes for each person
- ✅ Distinct embeddings for different people

### **Deliverable**
Script that outputs tracked faces with embeddings and keyframe images.

---

## **PHASE 3: Speaker-Face Association** (Week 4-5)

### **Goal**
Link speaker audio segments to face tracks and enable naming.

### **Features**
- Match speaker segments to face tracks by timestamp
- Manual labeling interface
- Face registry (optional: remember people for future)
- Generate final speaker-attributed transcript

### **Logic**
```
For each speaker segment:
  - Find which face tracks appear during that time
  - Assign most frequently appearing face to that speaker
  
SPEAKER_0 → track_1 → Manual label → "Alex Chen"
SPEAKER_1 → track_2 → Manual label → "Priya Patel"
```

### **Input/Output**
```
Input:  transcript.json (from Phase 1)
        faces.json (from Phase 2)
Output: labeled_transcript.json
{
  "segments": [
    {
      "start": "00:00",
      "end": "02:46",
      "speaker": "Alex Chen",
      "face_track_id": "track_1",
      "text": "Thanks everyone for joining..."
    }
  ],
  "participants": [
    {
      "name": "Alex Chen",
      "face_track_id": "track_1",
      "speaking_time": "15:30",
      "keyframe": "alex_keyframe.jpg"
    }
  ]
}
```

### **Tech Stack**
| Component | Technology | Why |
|-----------|-----------|-----|
| Matching logic | Python (pandas) | Data manipulation |
| Storage | SQLite | Lightweight DB |
| UI (optional) | Streamlit | Quick prototyping |

### **Processing Time**
- Matching: 5-10 seconds
- Manual labeling: User-dependent (1-2 minutes)
- **Total: ~1-2 minutes**

### **Success Criteria**
- ✅ Correct speaker-face matching (90%+ accuracy)
- ✅ Simple UI for labeling
- ✅ Export labeled transcript

### **Deliverable**
Complete speaker-attributed transcript with names and faces.

---

## **PHASE 4: Content Summarization** (Week 5-6)

### **Goal**
Generate structured meeting notes using LLM.

### **Features**
- Topic-wise summary
- Action items with owners
- Decisions made
- Timeline of discussion phases
- Key points extraction

### **Input/Output**
```
Input:  labeled_transcript.json
Output: meeting_notes.json
{
  "summary": "The team discussed Q3 budget allocation...",
  "topics": [
    {
      "title": "Q3 Budget Review",
      "timestamp": "00:00-10:30",
      "summary": "Alex presented budget proposal...",
      "participants": ["Alex Chen", "Priya Patel"]
    }
  ],
  "action_items": [
    {
      "task": "Review API latency metrics",
      "owner": "Priya Patel",
      "due_date": "Next week",
      "timestamp": "15:30"
    }
  ],
  "decisions": [
    {
      "decision": "Approved $50K budget increase",
      "timestamp": "08:45"
    }
  ]
}
```

### **Tech Stack**
| Component | Technology | Why |
|-----------|-----------|-----|
| LLM | GPT-4o or LLaMA-3-70B | Best reasoning |
| Prompt framework | LangChain | Structured prompts |
| API | OpenAI API or Ollama | Inference |

### **Models Required**
| Model | Option 1 (Cloud) | Option 2 (Local) |
|-------|------------------|------------------|
| LLM | GPT-4o via API | LLaMA-3-70B |
| Cost | $0.10-0.30/meeting | Free (requires GPU) |

### **Credentials**
- **Option 1**: OpenAI API key (paid)
- **Option 2**: None (self-hosted)

### **Installation**
```bash
# Option 1
pip install openai langchain

# Option 2
pip install ollama langchain
ollama pull llama3:70b
```

### **Processing Time (1hr meeting)**
- GPT-4o API: 30-60 seconds
- LLaMA-3 local: 2-3 minutes
- **Total: 0.5-3 minutes**

### **Success Criteria**
- ✅ Accurate topic identification
- ✅ All action items captured
- ✅ Correct owner attribution
- ✅ Coherent summary

### **Deliverable**
Structured meeting notes with topics, action items, and decisions.

---

## **PHASE 5: Integration & Output** (Week 7-8)

### **Goal**
End-to-end pipeline with user interface and export.

### **Features**
- Upload video interface
- Processing status tracking
- Display results (web UI)
- Export to PDF/DOCX
- Search across meetings (optional)

### **Tech Stack**
| Component | Technology | Why |
|-----------|-----------|-----|
| Backend API | FastAPI | Fast, async |
| Task queue | Celery + Redis | Background processing |
| Frontend | Streamlit or React | Quick UI |
| Export | python-docx, reportlab | Document generation |
| Storage | MinIO or local filesystem | Video storage |

### **Installation**
```bash
pip install fastapi celery redis streamlit python-docx reportlab
```

### **Architecture**
```
User uploads video
    ↓
FastAPI endpoint receives file
    ↓
Store in filesystem/MinIO
    ↓
Create Celery task
    ↓
Background worker runs:
    - Phase 1: Audio processing
    - Phase 2: Face detection
    - Phase 3: Speaker matching
    - Phase 4: Summarization
    ↓
Update status in Redis
    ↓
Return results to frontend
    ↓
User reviews & exports
```

### **Success Criteria**
- ✅ Video upload works (<2GB files)
- ✅ Processing runs in background
- ✅ Status updates in real-time
- ✅ Results display correctly
- ✅ Export generates clean documents

### **Deliverable**
Complete working application with UI.

---

## 📊 Overall System Requirements

### **Hardware (Minimum)**
- CPU: 8+ cores
- RAM: 16 GB
- GPU: NVIDIA GPU with 8GB VRAM (for faster processing)
- Storage: 50 GB free space

### **Hardware (Recommended)**
- CPU: 16+ cores
- RAM: 32 GB
- GPU: NVIDIA RTX 3090/4090 or A100
- Storage: 500 GB SSD

### **Cloud Alternative**
- AWS EC2 g4dn.xlarge or g5.xlarge
- Cost: ~$0.50-1.00 per meeting processed

---

## 📈 Performance Targets (1hr Meeting)

| Phase | Target Time | Acceptable Time |
|-------|-------------|-----------------|
| Phase 1 (Audio) | 5 min | 8 min |
| Phase 2 (Faces) | 4 min | 6 min |
| Phase 3 (Matching) | 1 min | 2 min |
| Phase 4 (Summary) | 2 min | 5 min |
| **Total** | **12 min** | **20 min** |

---

## 🎯 MVP Success Metrics

1. **Accuracy**
   - Transcription WER < 10%
   - Speaker diarization > 90% correct
   - Action item extraction > 85% recall

2. **Performance**
   - Process 1hr meeting in < 20 minutes
   - Support videos up to 2GB

3. **Usability**
   - Complete processing without manual intervention (except face labeling)
   - Export in < 10 seconds

---

## 🚀 Development Timeline

| Week | Phase | Deliverable |
|------|-------|------------|
| 1-2 | Phase 1 | Audio pipeline working |
| 3-4 | Phase 2 | Face detection working |
| 4-5 | Phase 3 | Speaker-face matching |
| 5-6 | Phase 4 | LLM summarization |
| 7 | Phase 5 | Integration |
| 8 | Phase 5 | Polish & testing |

---

## 📦 Deployment Options

### **Option 1: Local Application**
- Package with Docker
- User runs on their machine
- Good for: Demo, personal use

### **Option 2: Web Application**
- Deploy on AWS/Azure
- Users upload via web interface
- Good for: Portfolio, multiple users

### **Recommended for MVP**: Option 1 (Docker)

---

## 🔒 Privacy Considerations

- All processing done locally (no data leaves machine in Option 1)
- Face embeddings stored (not raw images)
- Optional: Add data deletion after processing

---

## 💰 Cost Estimate (Per Meeting)

| Component | Local (GPU) | Cloud (API) |
|-----------|-------------|-------------|
| Processing | Free | $0.50-1.00 |
| LLM | Free (local) | $0.10-0.30 |
| Storage | Local disk | $0.01 |
| **Total** | **Free** | **$0.61-1.31** |

---

## 🎓 Learning Outcomes

After completing this project, you will have demonstrated:
- Multimodal AI system design
- ML model integration (audio, vision, language)
- Production pipeline development
- Async processing with task queues
- Full-stack application development

---

## 📚 Additional Resources

### **Documentation**
- [Pyannote Audio Documentation](https://github.com/pyannote/pyannote-audio)
- [Whisper Documentation](https://github.com/openai/whisper)
- [InsightFace Documentation](https://github.com/deepinsight/insightface)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

### **Model Cards**
- [Pyannote Speaker Diarization 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [Whisper Large V3](https://huggingface.co/openai/whisper-large-v3)
- [InsightFace Buffalo L](https://github.com/deepinsight/insightface/tree/master/model_zoo)

### **Tutorials**
- Whisper + Pyannote alignment: [GitHub Issue](https://github.com/openai/whisper/discussions/264)
- Face tracking with ByteTrack: [Official Repo](https://github.com/ifzhang/ByteTrack)
- LangChain for summarization: [LangChain Docs](https://python.langchain.com/docs/use_cases/summarization)

---

## 🔄 Future Enhancements (Post-MVP)

### **Phase 6: Advanced Features** (Optional)
- **Real-time processing**: Process meetings as they happen
- **Screen content OCR**: Extract text from slides/screen shares
- **VLM integration**: Multimodal understanding (LLaVA for visual context)
- **Multi-meeting search**: Search across multiple meetings
- **Analytics dashboard**: Meeting insights over time
- **Calendar integration**: Auto-capture scheduled meetings

### **Technical Improvements**
- Quantized models for faster inference
- Streaming transcription
- Multi-language support
- Custom vocabulary for domain-specific terms
- Speaker embedding clustering (auto-identify recurring participants)

---

## ✅ Ready to Start?

### **Step 1: Environment Setup**
```bash
# Create project directory
mkdir meetingmind-ai
cd meetingmind-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Phase 1 dependencies
pip install pyannote.audio openai-whisper ffmpeg-python torch
```

### **Step 2: Get HuggingFace Token**
1. Create account at https://huggingface.co
2. Accept pyannote terms: https://huggingface.co/pyannote/speaker-diarization-3.1
3. Generate token: https://huggingface.co/settings/tokens
4. Save token for use in scripts

### **Step 3: Test Installation**
```bash
# Test Whisper
python -c "import whisper; print('Whisper OK')"

# Test Pyannote
python -c "from pyannote.audio import Pipeline; print('Pyannote OK')"
```

### **Step 4: Begin Phase 1**
Start with audio processing pipeline. Download a sample meeting video or use your own recorded meeting for testing.

---

## 📞 Support & Questions

For technical issues or questions:
- Check model documentation links above
- Review GitHub issues for each library
- Test with short video clips first (5-10 minutes)
- Monitor GPU memory usage during processing

---

**Document Version**: 1.0  
**Last Updated**: December 10, 2024  
**Status**: Ready for Development

---

Good luck with your project! 🚀
