# MeetingMind AI - Docker Deployment

## Quick Start

```bash
# 1. Make sure Ollama is running on host
ollama serve

# 2. Build and start containers
docker compose up --build

# 3. Access the app
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
```

## Architecture

```
┌─────────────────────────────────────────┐
│   Docker Compose                         │
│                                         │
│  ┌─────────┐  ┌─────────────────────┐  │
│  │Frontend │  │ Backend + Worker    │  │
│  │ (Nginx) │  │ (FastAPI + RQ)      │  │
│  │ :3000   │  │ :8000               │  │
│  └─────────┘  └─────────────────────┘  │
│                     │                   │
│              ┌──────▼──────┐           │
│              │    Redis    │           │
│              │    :6379    │           │
│              └─────────────┘           │
└─────────────────────────────────────────┘
         │
         ▼ (host network)
   ┌─────────────┐
   │   Ollama    │
   │  (GPU Host) │
   │   :11434    │
   └─────────────┘
```

## Files

- `Dockerfile.frontend` - React app served by Nginx
- `Dockerfile.backend` - FastAPI + RQ Worker + ML models
- `docker-compose.yml` - Orchestrates all services
- `.dockerignore` - Excludes unnecessary files

## Notes

- Ollama runs on the HOST (not in Docker) for GPU access
- Use `host.docker.internal:11434` to connect from containers
- Data persisted in `./src/data/` volume mount
