import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from sqlalchemy import text
from src.pipeline.phase4 import run_phase4
from datetime import timedelta
import json
from src.pipeline.phase1 import run_phase1
from src.pipeline.phase2 import run_phase2
from src.pipeline.phase3 import run_phase3
from src.pipeline.phase4 import run_phase4
from src.pipeline.phase5_visual import run_phase5



from src.app.db import engine
from src.app.settings import settings
from src.app.storage import write_manifest
from dotenv import load_dotenv
load_dotenv()


def update_db(meeting_id, **fields):
    if not fields:
        return
    sets = ", ".join(f"{k} = :{k}" for k in fields.keys())
    fields["meeting_id"] = meeting_id

    with engine.begin() as conn:
        conn.execute(
            text(f"UPDATE meetings SET {sets} WHERE meeting_id = :meeting_id"),
            fields,
        )


def load_manifest(meeting_id):
    path = Path(settings.DATA_ROOT) / meeting_id / "manifest.json"
    return path, path.read_text(encoding="utf-8")


def _fmt_time(seconds: float) -> str:
    td = timedelta(seconds=float(seconds))
    total = int(td.total_seconds())
    mm, ss = divmod(total, 60)
    hh, mm = divmod(mm, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

def phase1_stub(meeting_id: str):
    meeting_root = Path(settings.DATA_ROOT) / meeting_id
    out_path = meeting_root / "phase1" / "transcript.json"

    # Minimal transcript that Phase 4 can consume
    payload = {
        "segments": [
            {
                "start": _fmt_time(0),
                "end": _fmt_time(15),
                "speaker": "SPEAKER_00",
                "text": "Thanks everyone for joining. This is a stub transcript so Phase 4 can run."
            },
            {
                "start": _fmt_time(15),
                "end": _fmt_time(30),
                "speaker": "SPEAKER_01",
                "text": "Next steps: replace this stub with real Phase 1 diarization + ASR output."
            }
        ],
        "participants": [
            {"name": "SPEAKER_00"},
            {"name": "SPEAKER_01"}
        ]
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(out_path)

def phase1_real(meeting_id: str):
    meeting_root = Path(settings.DATA_ROOT) / meeting_id
    return run_phase1(meeting_root)

def phase2_real(meeting_id: str):
    meeting_root = Path(settings.DATA_ROOT) / meeting_id
    return run_phase2(meeting_root)

def phase2_stub(meeting_id):
    time.sleep(4)
    return "phase2 done"


def phase3_stub(meeting_id):
    time.sleep(2)


def process_meeting(meeting_id: str):
    try:
        meeting_root = Path(settings.DATA_ROOT) / meeting_id
        update_db(meeting_id, status="PROCESSING", phase="PHASE1", progress=0.05, message="Starting Phase 1 (real) + Phase 2 (real)")

        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(phase1_real, meeting_id)
            f2 = executor.submit(phase2_real, meeting_id)
            wait([f1, f2])

        phase1_out = f1.result()
        phase2_out = f2.result()
        transcript_path = Path(settings.DATA_ROOT) / meeting_id / "phase1" / "transcript.json"
        
        if not transcript_path.exists():
            raise FileNotFoundError(f"Phase 1 did not create transcript: {transcript_path}")

        update_db(meeting_id, message=f"Phase 1 & 2 OK")

        # Run Phase 5 (Visual Intelligence) - Analyzes slides, charts, whiteboards
        update_db(meeting_id, phase="PHASE5", progress=0.6, message="Running Phase 5 (Visual Intelligence)...")
        try:
            run_phase5(meeting_root, vlm_model="qwen2.5vl:latest", max_keyframes=50)
            update_db(meeting_id, progress=0.75, message="Phase 5 complete.")
        except Exception as e:
            print(f"Phase 5 failed (non-blocking): {e}")
            update_db(meeting_id, progress=0.75, message=f"Phase 5 skipped: {str(e)[:50]}")

        # Run Phase 3 (Matching) - Merges name suggestions with speaker-face mapping
        update_db(meeting_id, phase="PHASE3", progress=0.78, message="Running Phase 3 (Speaker-Face Matching)...")
        run_phase3(meeting_root)
        update_db(meeting_id, phase="PHASE3", progress=0.85, message="Phase 3 complete (Matching).")

        # Run Phase 4 (Summarization)
        update_db(meeting_id, phase="PHASE4", progress=0.9, message="Running Phase 4 (Summarization)...")
        meeting_root = Path(settings.DATA_ROOT) / meeting_id
        out_path = run_phase4(meeting_root, model="qwen2.5:14b")
        update_db(meeting_id, message=f"Phase 4 complete: {out_path.name}")

        update_db(
            meeting_id,
            status="READY",
            phase="DONE",
            progress=1.0,
            message="Processing complete",
        )

    except Exception as e:
        update_db(
            meeting_id,
            status="FAILED",
            error_phase="PIPELINE",
            error_message=str(e),
        )
        raise

