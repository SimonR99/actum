#!/usr/bin/env bash
# Run Actum with monitoring dashboard (web server)
# Dashboard available at: http://localhost:8000
# Usage: ./docker/run-server.sh
# Stop with: Ctrl+C or docker compose down

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Starting Actum with monitoring dashboard..."
echo ""
echo "Dashboard: http://localhost:8000"
echo "Stop with: Ctrl+C"
echo ""

cd "$PROJECT_ROOT"

# Run with docker-compose
docker compose up actum

echo ""
echo "Dashboard stopped."
