#!/usr/bin/env bash
set -e

WORKSPACE_PATH=$(pwd)
IMAGE_NAME="cls-sweep-cpp"
CONTAINER_NAME="cls-sweep-dev"

# Build the Docker image
echo "Building Docker image '${IMAGE_NAME}'..."
docker build -f docker/Dockerfile -t "${IMAGE_NAME}" .

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
    -v "${WORKSPACE_PATH}:/workspace" \
    -w /workspace \
    "${IMAGE_NAME}" \
    /bin/bash
