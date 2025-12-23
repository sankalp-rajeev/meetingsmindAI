import json
from pathlib import Path
from datetime import datetime

def meeting_root(data_root: str, meeting_id: str) -> Path:
    return Path(data_root) / meeting_id

def ensure_meeting_dirs(root: Path) -> None:
    (root / "phase1").mkdir(parents=True, exist_ok=True)
    (root / "phase2").mkdir(parents=True, exist_ok=True)
    (root / "phase3").mkdir(parents=True, exist_ok=True)
    (root / "phase4").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)

def write_manifest(root: Path, manifest: dict) -> None:
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

def new_manifest(meeting_id: str, filename: str) -> dict:
    return {
        "meeting_id": meeting_id,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "original_filename": filename,
        "status": "PROCESSING",
        "phase": "PHASE1",
        "artifacts": {
            "original_video": "original.mp4",
            "phase1": {"transcript": "phase1/transcript.json", "ready": False},
            "phase2": {"faces": "phase2/faces.json", "ready": False},
            "phase3": {"labeled_transcript": "phase3/labeled_transcript.json", "ready": False},
            "phase4": {"meeting_notes": "phase4/meeting_notes.json", "ready": False},
        },
        "warnings": [],
        "errors": None,
    }
