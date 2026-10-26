#!/bin/bash
# Host side. One render node, so the container sees a single GPU.
#
#   bash scripts/radeon_start_container.sh /disk/ssd2/radeon/worldscore /dev/dri/renderD130
#
# GPU 0 (renderD128) and GPU 1 (renderD129) are already taken by other
# Radeon sessions. This default is GPU 2, PCI 0000:43:00.0.
# Workdir stays on /disk/ssd2: the root filesystem is nearly full.
# Core dumps are off. A crashed HIP process can write tens of GB.
set -euo pipefail
HOST_DIR=${1:?usage: radeon_start_container.sh HOST_DIR [RENDER_NODE]}
RENDER_NODE=${2:-/dev/dri/renderD130}
NAME=${NAME:-radeon-worldscore}
IMAGE=${IMAGE:-rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0}
VIDEO_GID=$(getent group video | cut -d: -f3)
RENDER_GID=$(getent group render | cut -d: -f3)
mkdir -p "$HOST_DIR/weights" "$HOST_DIR/data" "$HOST_DIR/results" "$HOST_DIR/logs" "$HOST_DIR/src"

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "container $NAME already exists"
  docker start "$NAME" >/dev/null
else
  docker run -d --name "$NAME" \
    --entrypoint bash \
    --device=/dev/kfd \
    --device="$RENDER_NODE" \
    --group-add "$VIDEO_GID" \
    --group-add "$RENDER_GID" \
    --ipc=host \
    --shm-size=32g \
    --ulimit core=0 \
    --security-opt seccomp=unconfined \
    -e HIP_VISIBLE_DEVICES=0 \
    -e ROCR_VISIBLE_DEVICES=0 \
    -e PYTORCH_ROCM_ARCH=gfx1100 \
    -v "$HOST_DIR":/workspace \
    -w /workspace \
    "$IMAGE" \
    -lc 'sleep infinity'
fi

docker exec "$NAME" /opt/venv/bin/python -c \
  'import torch; print("torch", torch.__version__, "hip", torch.version.hip, "devices", torch.cuda.device_count(), torch.cuda.get_device_name(0))'
