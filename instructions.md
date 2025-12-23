1) High-level architecture

PRD goal: upload meeting video → background pipeline runs → status updates + artifacts → UI review/export. 

MeetingMind_AI_PRD

Intended phases

Phase 1: audio extraction + diarization + transcription

Phase 2: face detection / tracking

Phase 3: speaker ↔ face association + labeling UI

Phase 4: structured meeting notes (LLM via local Ollama)

Phase 5: UI + export integration 

MeetingMind_AI_PRD

2) What each file does (shared files)
src/worker/worker.py

Runs the RQ worker consuming jobs from Redis queue "meetings".

You effectively have two variants:

Forking worker (Linux-ish)

Uses rq.Worker([q]) 

worker

On Windows this can blow up because RQ’s default execution path tries to fork.

Windows-safe worker

Uses rq.worker.SimpleWorker 

worker

This is why your later worker runs succeeded on Windows (no os.fork() dependency).

src/worker/tasks.py

Defines the background job process_meeting(meeting_id).

Responsibilities:

Updates DB status using SQLAlchemy and an UPDATE meetings SET ... query. 

tasks

Runs Phase 1 (real) and Phase 2 (stub) “in parallel” with ThreadPoolExecutor. 

tasks

Checks expected Phase 1 output exists at:
DATA_ROOT/<meeting_id>/phase1/transcript.json 

tasks

Calls Phase 4 summarization: run_phase4(meeting_root, model="qwen2.5:14b"). 

tasks

On exception: marks meeting FAILED and re-raises. 

tasks

Also includes a Phase 1 stub generator that writes a minimal phase1/transcript.json (useful to test Phase 4 without running diarization/whisper). 

tasks

src/pipeline/phase1.py

A wrapper that runs Phase 1 as a subprocess and returns the transcript path.

Resolves Phase 1 script path. 

phase1

Runs it like: python phase1_audio_processing.py --in <original.mp4> --out <phase1/transcript.json> 

phase1

Fails if subprocess non-zero or transcript.json missing. 

phase1

Critical mismatch: your Phase 1 script currently requires --outdir too, but this wrapper doesn’t pass it → argparse failure.

src/pipeline/phase1_audio_processing.py

The actual Phase 1 pipeline:

Extract audio from video via FFmpeg 

phase1_audio_processing

Speaker diarization (pyannote)

Whisper transcription

Merge speaker+text and write JSON outputs

Highlights:

FFmpeg extraction is explicitly invoked (ffmpeg ... -ar 16000 -ac 1). 

phase1_audio_processing

process_audio(..., output_dir=...) writes artifacts under output_dir. 

phase1_audio_processing

Prints Unicode symbols (like ✓) which can crash on Windows console encodings (cp1252). 

phase1_audio_processing

What your logs showed:

The work completed, but the job still failed because a later print used ✅/❌ and crashed with UnicodeEncodeError.

You also saw torchcodec/FFmpeg warnings (noisy, and may break decoding paths depending on how pyannote reads audio).

src/pipeline/phase4.py

Generates structured meeting notes using local Ollama.

Prefers phase3/labeled_transcript.json, otherwise uses phase1/transcript.json. 

phase4

Builds participants list from speaker IDs if needed. 

phase4

Calls http://localhost:11434/api/chat with a “JSON only” system prompt. 

phase4

Has robust JSON extraction logic (direct parse → code block → {...} span). 

phase4

Writes phase4/meeting_notes.json. 

phase4

Why you kept hitting No transcript found for Phase 4:

Neither phase3/labeled_transcript.json nor phase1/transcript.json existed where Phase 4 expects. 

phase4

phase3_label_ui.py (Streamlit labeling UI)

A labeling dashboard:

Video player on left + controls on right. 

phase3_label_ui

Auto-suggests speaker mapping based on match-rate threshold. 

phase3_label_ui

Saves mapping JSON. 

phase3_label_ui

Matches your requirement: “video playing side-by-side with labels + auto speaker info”.

MeetingMind_AI_PRD.md

Product/architecture spec. 

MeetingMind_AI_PRD

3) What’s completed vs what isn’t
✅ Working (observed)

Worker can run on Windows when using SimpleWorker. 

worker

Phase 4 pipeline exists and works once it has a transcript. 

phase4

Phase 1 algorithmically works (audio → diarize → whisper → merge), based on your stdout, but fails due to integration issues. 

phase1_audio_processing

🚧 In-progress / broken

Phase 2 and Phase 3 in the worker job path are currently stubs (sleep). 

tasks

Phase 1 integration is fragile:

wrapper missing required CLI args (--outdir)

subprocess may use wrong python interpreter (causing torch.amp missing earlier)

Unicode printing crashes after work completes 

phase1_audio_processing

4) Immediate next steps (fix order)
Step 1 — Make Phase 1 wrapper reliable (biggest blocker)

Do these first because everything else depends on Phase 1 output existing.

Use sys.executable instead of "python" in phase1.py
This prevents “subprocess used system Python” and missing torch modules.

Pass --outdir (Phase 1 script requires it per your logs)
Use meeting_root/phase1 as outdir.

Remove Unicode symbols from prints in Phase 1 script
Replace ✅❌✓ with ASCII ([OK], [ERROR]) so Windows console won’t crash.

Ensure Phase 1 writes exactly:

phase1/transcript.json (the file Phase 4 expects) 

tasks

Step 2 — Keep Phase 4 stable

Once Phase 1 reliably creates phase1/transcript.json, Phase 4 should stop throwing “No transcript found…” 

phase4

Step 3 — Phase 2/3 real implementation (later)

Only after Phase 1 + Phase 4 are stable, start filling:

Phase 2 face tracks

Phase 3 association + UI loop (Streamlit UI already exists)

5) “Why argparse/subprocess — is that production?”

Yes, it’s common to have CLI scripts + a worker calling them.

But “production best practice” is: don’t let it be fragile.
That means:

correct args always passed

consistent interpreter

no console encoding crashes

deterministic outputs

So your structure is fine for MVP, but hardening Phase 1 invocation is mandatory.

6) Operational runbook (what to run in terminals)
Terminal A — Redis

If using Docker Desktop:

Start Redis container (example): docker run -p 6379:6379 redis

(Or run Redis locally if already installed.)

Terminal B — API

From repo root (venv active):

uvicorn src.app.main:app --reload --port 8000

Then visit:

http://127.0.0.1:8000/docs

Terminal C — Worker

From repo root (venv active):

python -m src.worker.worker

Uploading

Yes — you can upload from Swagger UI at:

http://127.0.0.1:8000/docs#/default/upload_meeting_api_meetings_post

If you see 422, it’s request-shape mismatch (wrong field name / missing file).

Appendix: Expected artifact paths

Meeting root: DATA_ROOT/<meeting_id>/

original.mp4

phase1/transcript.json (required fallback for Phase 4) 

phase4

phase3/labeled_transcript.json (preferred for Phase 4) 

phase4

phase4/meeting_notes.json (output) 

phase4