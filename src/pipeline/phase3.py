import subprocess
from pathlib import Path
import sys


def run_phase3(meeting_root: Path) -> Path:
    script = Path(__file__).resolve().parent / "phase3_matching.py"
    
    transcript_in = meeting_root / "phase1" / "transcript.json"
    faces_in = meeting_root / "phase2" / "faces.json"
    
    # Phase 3 outputs go to phase3/
    out_dir = meeting_root / "phase3"
    map_file = out_dir / "speaker_face_map.json"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not transcript_in.exists():
        raise FileNotFoundError(f"Phase 3 requires Phase 1 transcript: {transcript_in}")
    if not faces_in.exists():
         raise FileNotFoundError(f"Phase 3 requires Phase 2 faces: {faces_in}")

    cmd = [
        sys.executable,
        str(script),
        "--transcript", str(transcript_in),
        "--faces", str(faces_in),
        "--outdir", str(out_dir),
    ]
    
    if map_file.exists():
        cmd.extend(["--map", str(map_file)])

    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        raise RuntimeError(
            f"Phase 3 failed.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    out_json = out_dir / "prelabeled_transcript.json"
    if not out_json.exists():
        raise FileNotFoundError("Phase 3 completed but prelabeled_transcript.json was not created.")

    return out_json
