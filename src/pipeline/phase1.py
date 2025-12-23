import subprocess
from pathlib import Path
import sys


def run_phase1(meeting_root: Path) -> Path:
    script = Path(__file__).resolve().parent / "phase1_audio_processing.py"
    video_in = meeting_root / "original.mp4"
    out_json = meeting_root / "phase1" / "transcript.json"

    out_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
    sys.executable,
    str(script),
    "--in", str(video_in),
    "--out", str(out_json),
    "--outdir", str(out_json.parent),]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        raise RuntimeError(
            f"Phase 1 failed.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    if not out_json.exists():
        raise FileNotFoundError("Phase 1 completed but transcript.json was not created.")

    return out_json
