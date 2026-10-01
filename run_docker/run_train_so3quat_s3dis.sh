#!/usr/bin/env bash
set -e

DEVICES="6"
OMP_NUM_THREADS=4
DOCKER_IMAGE="letatanu/pointcept1"
MODEL_NAME="so3_quaternion_transformer"
EXP_NAME="so3quat_transformer_02"
DATASET="s3dis"

echo "Starting diagnostic training on devices: $DEVICES"
echo "Model Name: $MODEL_NAME"

docker run --init --ulimit nofile=1048576:1048576 --ipc=host \
  --rm -ti \
  --gpus "\"device=${DEVICES}\"" \
  -w /working \
  -v /data/nhl224/code/semantic_3D/Pointcept/:/working \
  -e OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  "${DOCKER_IMAGE}" bash -lc "
  sh scripts/train.sh \
      -p python \
      -d ${DATASET} \
      -c ${MODEL_NAME} \
      -n ${EXP_NAME}"