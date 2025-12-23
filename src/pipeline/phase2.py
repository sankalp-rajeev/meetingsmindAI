import subprocess
from pathlib import Path
import sys


def run_phase2(meeting_root: Path) -> Path:
    script = Path(__file__).resolve().parent / "phase2_face_tracking.py"
    video_in = meeting_root / "original.mp4"
    out_dir = meeting_root / "phase2"
    out_json = out_dir / "faces.json"

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script),
        "--in", str(video_in),
        "--outdir", str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        raise RuntimeError(
            f"Phase 2 failed.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    if not out_json.exists():
        raise FileNotFoundError("Phase 2 completed but faces.json was not created.")

    return out_json
