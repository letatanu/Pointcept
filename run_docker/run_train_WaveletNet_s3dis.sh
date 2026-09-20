#!/usr/bin/env bash
set -e

DEVICES="1,2,3,4,5,6,7"
OMP_NUM_THREADS=4
DOCKER_IMAGE="letatanu/pointcept1"
MODEL_NAME="semseg-aerial-wavelet-v1"
EXP_NAME="AerialWaveletNet_01"
DATASET="s3dis"

echo "Starting diagnostic training on devices: $DEVICES"
echo "Model Name: $MODEL_NAME"

docker run --ulimit nofile=1048576:1048576 --ipc=host \
  --rm -ti \
  --gpus "\"device=${DEVICES}\"" \
  -w /working \
  -v /data/nhl224/code/semantic_3D/Pointcept/:/working \
  -e OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  -e CUDA_LAUNCH_BLOCKING=1 \
  -e PTQWNO_DEBUG=1 \
  -e PYTHONUNBUFFERED=1 \
  "${DOCKER_IMAGE}" bash -lc "
  sh scripts/train.sh \
      -p python \
      -d ${DATASET} \
      -c ${MODEL_NAME} \
      -n ${EXP_NAME}"