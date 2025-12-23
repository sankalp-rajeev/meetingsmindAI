mplementation Plan by Blocks
Block 0 — Project foundation and contracts

Goal: You have a clean repo layout + agreed artifact contract + stable “meeting_id” concept.

Deliverables

Folder structure created (app/, worker/, pipeline/, data/meetings/)

A single documented artifact layout:

data/meetings/<meeting_id>/
  manifest.json
  original.mp4
  phase1/transcript.json
  phase2/faces.json
  phase3/labeled_transcript.json
  phase4/meeting_notes.json
  logs/pipeline.log


A simple “status state machine” definition:

UPLOADED → PROCESSING → READY

plus FAILED / NEEDS_REVIEW

Acceptance check

You can create a meeting folder by hand and see the structure.

Block 1 — Database setup (Postgres) + migrations

Goal: You can create a meetings table and read/write meeting rows.

Deliverables

Postgres installed + running locally

Database created (e.g. meetingsai)

meetings table with minimal columns:

meetings table (minimum)

meeting_id (UUID PK)

title (nullable)

created_at

status (enum-like text)

phase (text)

progress (float 0..1)

message (text)

artifact_root (text)

duration_seconds (nullable float)

model_text (default qwen2.5:14b)

error_phase / error_message (nullable)

Acceptance check

You can insert one row and query it from your app (even a tiny “hello DB” test).

Block 2 — FastAPI server skeleton

Goal: API server boots and can hit a health endpoint.

Deliverables

FastAPI app starts on localhost:8000

/health returns OK

Config system for:

DB connection string

data root path

Redis connection string

Acceptance check

You can start the server and hit /health in browser.

Block 3 — Upload endpoint with auto-start trigger (but no worker yet)

Goal: Upload creates meeting folder + DB row + manifest.

API

POST /api/meetings (multipart file upload)

What it must do

Generate meeting_id

Create /data/meetings/<meeting_id>/... folders

Save original.mp4

Create manifest.json with basic fields:

meeting_id

created_at

status

artifacts paths (ready=false initially)

Insert DB row:

status = UPLOADED (or PROCESSING if you want)

phase = PHASE1

progress ~0.01

Return { meeting_id }

Acceptance check

Upload from Postman / curl works

You see the MP4 saved + manifest created + DB row created

Auto-start will be “enqueue job” in the next block (once Redis/worker exists).

Block 4 — Status + list endpoints

Goal: Frontend can poll status and show meeting list.

Endpoints

GET /api/meetings → list (id/title/created_at/status/progress)

GET /api/meetings/{id}/status → status payload

GET /api/meetings/{id}/manifest → returns manifest.json

Acceptance check

After upload, you can query status and list and see the meeting.

Block 5 — Queue + Worker scaffolding (no pipeline yet)

Goal: Upload auto-enqueues a job and worker picks it up.

Deliverables

Redis running locally

Worker process started separately

Upload endpoint now:

sets DB status to PROCESSING

enqueues process_meeting(meeting_id)

Worker:

pulls job

writes logs + updates DB (just “started/finished” placeholders)

Acceptance check

Upload → job appears → worker runs → DB updates from worker

Block 6 — Pipeline orchestration (the important parallelism)

Goal: process_meeting() runs phases in the correct parallel/sequential order.

Execution rules (locked)

✅ Phase 1 and Phase 2 run in parallel
✅ Phase 3 waits for both
✅ Phase 4 waits for Phase 3

Orchestration behavior

Worker starts:

Phase 1 task (CPU)

Phase 2 task (GPU)

Barrier wait

Then Phase 3

Then Phase 4

Progress updates (so UI is real)

During Phase 1/2: progress moves 0.0 → 0.6 (combined)

Phase 3: 0.6 → 0.8

Phase 4: 0.8 → 1.0 (chunk-by-chunk)

Acceptance check

Even with dummy phase runners, you can see the state transitions happen in order.

Block 7 — Wire in Phase 1–4 “real runners”

Goal: Replace placeholders with your real scripts.

Phase 1 runner

input: original.mp4

output: phase1/transcript.json

Phase 2 runner

input: original.mp4

output: phase2/faces.json

Phase 3 runner

input: transcript + faces

output: phase3/labeled_transcript.json

Phase 4 runner

input: labeled_transcript

calls local Ollama

output: phase4/meeting_notes.json

plus: robust JSON handling + repair logs (like we discussed)

Acceptance check

Upload a real video → meeting becomes READY → you can download transcript + notes.

Block 8 — Artifact serving endpoints

Goal: UI can render the product.

Endpoints:

GET /api/meetings/{id}/transcript → labeled_transcript JSON

GET /api/meetings/{id}/notes → meeting notes JSON

GET /api/meetings/{id}/video → stream the MP4 (later optimize)

Acceptance check

Browser can fetch the JSON and video plays.

