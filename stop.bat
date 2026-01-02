@echo off
echo Stopping MeetingMind AI...
echo.

cd docker
docker compose down

echo.
echo All services stopped.
pause
