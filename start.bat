@echo off
echo Starting MeetingMind AI...
echo.
echo Make sure Ollama is running: ollama serve
echo.

REM Start Docker Compose
cd docker
docker compose up --build -d

echo.
echo ====================================
echo MeetingMind AI is starting up!
echo.
echo Frontend: http://localhost:3000
echo Backend:  http://localhost:8000
echo ====================================
echo.
echo To view logs: docker compose logs -f
echo To stop:      stop.bat
echo.

pause
