#!/usr/bin/env bash
# Build Docker image for vlm-g1-agent
# Usage: ./docker/build.sh
# Or:    ./docker/build.sh --no-cache

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Building vlm-g1:latest Docker image from $PROJECT_ROOT..."
echo ""

cd "$PROJECT_ROOT"

# Check if Docker is running
if ! docker ps > /dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running. Please start Docker and try again."
    exit 1
fi

# Build with optional --no-cache flag
BUILD_ARGS=""
if [[ "$1" == "--no-cache" ]]; then
    BUILD_ARGS="--no-cache"
    echo "Building without cache (this may take a while)..."
fi

docker build \
    $BUILD_ARGS \
    -t vlm-g1:latest \
    -f Dockerfile \
    .

echo ""
echo "✓ Build complete: vlm-g1:latest"
echo ""
echo "Next steps:"
echo "  - Run with GPU support:  docker compose up"
echo "  - Or run shell:           ./docker/shell.sh"
echo "  - Or run headless mode:   ./docker/run-headless.sh"
