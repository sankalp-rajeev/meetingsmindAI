from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from uuid import uuid4
from pathlib import Path
import shutil
import json
from redis import Redis
from rq import Queue
import os
from src.app.db import engine
from src.app.settings import settings
from src.app.storage import meeting_root, ensure_meeting_dirs, new_manifest, write_manifest
from uuid import UUID
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Dict, Optional

app = FastAPI(title="MeetingsAI API", version="0.1.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for meeting artifacts
data_root_path = Path(settings.DATA_ROOT)
data_root_path.mkdir(parents=True, exist_ok=True)
app.mount("/data/meetings", StaticFiles(directory=str(data_root_path)), name="meetings")


@app.on_event("startup")
def create_tables():
    """Create database tables on startup if they don't exist."""
    with engine.begin() as conn:
        # Detect database type and use appropriate syntax
        dialect = engine.dialect.name
        
        if dialect == "postgresql":
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS meetings (
                    id SERIAL PRIMARY KEY,
                    meeting_id TEXT UNIQUE NOT NULL,
                    title TEXT,
                    status TEXT DEFAULT 'PROCESSING',
                    phase TEXT DEFAULT 'PHASE1',
                    progress REAL DEFAULT 0.0,
                    message TEXT,
                    artifact_root TEXT,
                    error_phase TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
        else:
            # SQLite
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS meetings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meeting_id TEXT UNIQUE NOT NULL,
                    title TEXT,
                    status TEXT DEFAULT 'PROCESSING',
                    phase TEXT DEFAULT 'PHASE1',
                    progress REAL DEFAULT 0.0,
                    message TEXT,
                    artifact_root TEXT,
                    error_phase TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
    print("✓ Database tables initialized")

@app.get("/health")
def health():
    return {"ok": True}


@app.get("/db-check")
def db_check():
    with engine.connect() as conn:
        v = conn.execute(text("SELECT 1")).scalar_one()
    return {"db_ok": True, "result": v}


@app.post("/api/meetings")
def upload_meeting(file: UploadFile = File(...)):
    meeting_id = str(uuid4())

    root = meeting_root(settings.DATA_ROOT, meeting_id)
    ensure_meeting_dirs(root)

    # Save original video
    video_path = root / "original.mp4"
    with video_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Write manifest
    manifest = new_manifest(meeting_id, file.filename or "uploaded.mp4")
    write_manifest(root, manifest)

    # Insert meeting row
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO meetings (meeting_id, title, status, phase, progress, message, artifact_root)
                VALUES (:meeting_id, :title, :status, :phase, :progress, :message, :artifact_root)
            """),
            {
                "meeting_id": meeting_id,
                "title": file.filename,
                "status": "PROCESSING",
                "phase": "PHASE1",
                "progress": 0.01,
                "message": "Uploaded. Starting Phase 1 and Phase 2 soon.",
                "artifact_root": str(root),
            },
        )
    redis = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    q = Queue("meetings", connection=redis)
    q.enqueue("src.worker.tasks.process_meeting", meeting_id, job_timeout=3600)
    print("ENQUEUED JOB for meeting_id =", meeting_id)
    return {"meeting_id": str(meeting_id)}



@app.get("/api/meetings")
def list_meetings():
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT
                    meeting_id,
                    title,
                    status,
                    phase,
                    progress,
                    created_at
                FROM meetings
                ORDER BY created_at DESC
            """)
        ).mappings().all()

    return {
        "meetings": [dict(row) for row in rows]
    }
    
@app.get("/api/meetings/{meeting_id}/status")
def meeting_status(meeting_id: UUID):
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT
                    meeting_id,
                    status,
                    phase,
                    progress,
                    message,
                    error_phase,
                    error_message
                FROM meetings
                WHERE meeting_id = :meeting_id
            """),
            {"meeting_id": str(meeting_id)}
        ).mappings().first()

    if not row:
        raise HTTPException(status_code=404, detail="Meeting not found")

    return dict(row)

@app.get("/api/meetings/{meeting_id}/manifest")
def get_manifest(meeting_id: UUID):
    root = Path(settings.DATA_ROOT) / str(meeting_id)
    manifest_path = root / "manifest.json"

    if not manifest_path.exists():
        return {"error": "Manifest not found"}

    return manifest_path.read_text(encoding="utf-8")


@app.get("/api/meetings/{meeting_id}/transcript")
def get_transcript(meeting_id: UUID):
    root = Path(settings.DATA_ROOT) / str(meeting_id)
    
    # Prefer labeled transcript (with speaker names) if available
    labeled_path = root / "phase3" / "labeled_transcript.json"
    fallback_path = root / "phase1" / "transcript.json"
    
    if labeled_path.exists():
        return json.loads(labeled_path.read_text(encoding="utf-8"))
    elif fallback_path.exists():
        return json.loads(fallback_path.read_text(encoding="utf-8"))
    else:
        raise HTTPException(status_code=404, detail="Transcript not found")


@app.get("/api/meetings/{meeting_id}/notes")
def get_notes(meeting_id: UUID):
    root = Path(settings.DATA_ROOT) / str(meeting_id)
    path = root / "phase4" / "meeting_notes.json"
    
    if not path.exists():
        raise HTTPException(status_code=404, detail="Notes not found")
    
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/meetings/{meeting_id}/faces")
def get_faces(meeting_id: UUID):
    root = Path(settings.DATA_ROOT) / str(meeting_id)
    path = root / "phase2" / "faces.json"
    
    if not path.exists():
        raise HTTPException(status_code=404, detail="Faces not found")
    
    return json.loads(path.read_text(encoding="utf-8"))


# Pydantic model for speaker labels
class SpeakerLabelsUpdate(BaseModel):
    labels: Dict[str, str]  # {"SPEAKER_00": "Alex Chen", "track_1": "Alex Chen"}


@app.get("/api/meetings/{meeting_id}/speaker-labels")
def get_speaker_labels(meeting_id: UUID):
    """Get existing speaker-face label mappings"""
    root = Path(settings.DATA_ROOT) / str(meeting_id)
    map_path = root / "phase3" / "speaker_face_map.json"
    
    if not map_path.exists():
        return {"labels": {}, "speaker_to_face": {}}
    
    data = json.loads(map_path.read_text(encoding="utf-8"))
    
    # Check if data has old format (labels dict) or new format (mappings array)
    if "mappings" in data:
        # Convert mappings array to labels dict and speaker_to_face dict
        labels = {}
        speaker_to_face = {}
        for mapping in data.get("mappings", []):
            speaker_id = mapping.get("speaker_id", "")
            name = mapping.get("name", speaker_id)
            face_track_id = mapping.get("face_track_id")
            
            if speaker_id:
                labels[speaker_id] = name
                if face_track_id:
                    speaker_to_face[speaker_id] = face_track_id
        
        return {"labels": labels, "speaker_to_face": speaker_to_face}
    else:
        # Old format already has labels dict
        return {"labels": data.get("labels", {}), "speaker_to_face": data.get("speaker_to_face", {})}


@app.patch("/api/meetings/{meeting_id}/speaker-labels")
def update_speaker_labels(meeting_id: UUID, update: SpeakerLabelsUpdate):
    """Save speaker-face label mappings and regenerate labeled transcript"""
    root = Path(settings.DATA_ROOT) / str(meeting_id)
    phase3_dir = root / "phase3"
    phase3_dir.mkdir(parents=True, exist_ok=True)
    
    map_path = phase3_dir / "speaker_face_map.json"
    
    # Load existing data or create new
    if map_path.exists():
        data = json.loads(map_path.read_text(encoding="utf-8"))
    else:
        data = {"labels": {}, "speaker_to_face": {}}
    
    # Update labels
    data["labels"] = {**data.get("labels", {}), **update.labels}
    
    # Save mapping
    map_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    
    # Regenerate labeled transcript if phase1 transcript exists
    transcript_path = root / "phase1" / "transcript.json"
    if transcript_path.exists():
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        
        # Apply labels to transcript
        labeled_segments = []
        for seg in transcript.get("segments", []):
            speaker = seg.get("speaker", "")
            labeled_seg = {**seg}
            # Check if we have a custom label for this speaker
            if speaker in data["labels"]:
                labeled_seg["speaker_label"] = data["labels"][speaker]
            else:
                labeled_seg["speaker_label"] = speaker
            labeled_segments.append(labeled_seg)
        
        # Build participants list with labels
        speakers = list(set(s.get("speaker", "") for s in transcript.get("segments", [])))
        participants = []
        for sp in speakers:
            participants.append({
                "speaker_id": sp,
                "name": data["labels"].get(sp, sp),
                "face_track_id": data.get("speaker_to_face", {}).get(sp)
            })
        
        labeled_transcript = {
            "segments": labeled_segments,
            "participants": participants
        }
        
        labeled_path = phase3_dir / "labeled_transcript.json"
        labeled_path.write_text(json.dumps(labeled_transcript, indent=2), encoding="utf-8")
    
    return {"success": True, "labels": data["labels"]}


@app.get("/api/meetings/{meeting_id}/speaker-suggestions")
def get_speaker_suggestions(meeting_id: UUID):
    """Get auto-detected speaker to face suggestions from Phase 3"""
    root = Path(settings.DATA_ROOT) / str(meeting_id)
    suggestions_path = root / "phase3" / "speaker_face_suggestions.json"
    
    if not suggestions_path.exists():
        return {"suggestions": [], "speaker_to_face": {}}
    
    data = json.loads(suggestions_path.read_text(encoding="utf-8"))
    
    # Build a simple speaker_to_face mapping from suggestions
    speaker_to_face = {}
    for suggestion in data.get("suggestions", []):
        speaker_id = suggestion.get("speaker_id")
        face_track_id = suggestion.get("suggested_face_track_id")
        match_rate = suggestion.get("match_rate", 0)
        
        # Only include high-confidence matches (>50% match rate)
        if speaker_id and face_track_id and match_rate > 0.5:
            speaker_to_face[speaker_id] = face_track_id
    
    return {
        "suggestions": data.get("suggestions", []),
        "speaker_to_face": speaker_to_face
    }


@app.get("/api/meetings/{meeting_id}/visual-insights")
def get_visual_insights(meeting_id: UUID):
    """Get visual insights from Phase 5 analysis"""
    root = Path(settings.DATA_ROOT) / str(meeting_id)
    insights_path = root / "phase5" / "visual_insights.json"
    
    if not insights_path.exists():
        return {"keyframes_analyzed": 0, "insights": [], "summary": "No visual analysis available"}
    
    data = json.loads(insights_path.read_text(encoding="utf-8"))
    return data


# ============================================================
# RAG Q&A Endpoints
# ============================================================

class QuestionRequest(BaseModel):
    question: str


class ChatClearResponse(BaseModel):
    success: bool
    message: str


@app.post("/api/meetings/{meeting_id}/ask")
def ask_meeting_question(meeting_id: UUID, request: QuestionRequest):
    """Ask a question about the meeting content using RAG."""
    from src.app.rag import get_meeting_rag
    
    # Verify meeting exists
    root = Path(settings.DATA_ROOT) / str(meeting_id)
    if not root.exists():
        raise HTTPException(status_code=404, detail="Meeting not found")
    
    try:
        rag = get_meeting_rag(str(meeting_id), settings.DATA_ROOT)
        result = rag.ask(request.question)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")


@app.delete("/api/meetings/{meeting_id}/chat")
def clear_chat_history(meeting_id: UUID):
    """Clear the conversation memory for a meeting."""
    from src.app.rag import get_meeting_rag, clear_rag_cache
    
    try:
        rag = get_meeting_rag(str(meeting_id), settings.DATA_ROOT)
        rag.clear_memory()
        return ChatClearResponse(success=True, message="Chat history cleared")
    except Exception as e:
        return ChatClearResponse(success=False, message=str(e))
