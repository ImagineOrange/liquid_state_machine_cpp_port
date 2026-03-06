#!/usr/bin/env bash
set -e

IMAGE_NAME="cls-sweep-cpp"
CONTAINER_NAME="cls-sweep-dev"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Build the Docker image
echo "Building Docker image '${IMAGE_NAME}'..."
docker build -t "${IMAGE_NAME}" "${SCRIPT_DIR}"

# Remove any existing container with the same name
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

# Run the container interactively
# Mount the project directory so edits persist on host
echo ""
echo "Entering dev container. The project is mounted at /workspace."
echo "The binary is pre-built at /workspace/cls_sweep"
echo ""
echo "Usage:"
echo "  ./cls_sweep --arms all --n-workers 8"
echo "  ./cls_sweep --arms original --n-workers 4 --output-dir /workspace/results/test"
echo "  make clean && make -j\$(nproc)    # rebuild"
echo ""

exec docker run -it --rm \
    --name "${CONTAINER_NAME}" \
    -v "${SCRIPT_DIR}:/workspace" \
    -w /workspace \
    "${IMAGE_NAME}" \
    /bin/bash
