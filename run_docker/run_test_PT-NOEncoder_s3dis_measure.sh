#!/usr/bin/env bash
set -e

DEVICES="0"
OMP_NUM_THREADS=4
DOCKER_IMAGE="letatanu/pointcept1"

echo "Starting S3DIS Latency/Memory Measurement on Devices: $DEVICES"
EXP_NAME="semseg-pt-v3m1-0-base"
DATASET="s3dis"

docker run --ulimit nofile=1048576:1048576 --ipc=host \
  --shm-size=32g \
  --rm -ti \
  --gpus "\"device=${DEVICES}\"" \
  -w /working \
  -v /data/nhl224/code/semantic_3D/Pointcept/:/working \
  -e OMP_NUM_THREADS=${OMP_NUM_THREADS} \
  "${DOCKER_IMAGE}" \
  bash -lc "
    python tools/measure_inference.py \
      --config-file exp/${DATASET}/${EXP_NAME}/config.py \
      --checkpoint exp/${DATASET}/${EXP_NAME}/model/model_best.pth \
      --num-warmup 10 \
      --num-runs 50 \
      --save-results exp/${DATASET}/${EXP_NAME}/inference_metrics.json"