#!/usr/bin/env bash
set -e

# Set available GPUs (adjust as needed, e.g., "0" or "0,1")
DEVICES="0,1"
# Calculate number of processes based on devices
OMP_NUM_THREADS=4

## --------------------------------------------------------- ##
# Ensure this matches your docker image name
DOCKER_IMAGE="letatanu/poincept1"

echo "Starting AeroRelief3D Training on Devices: $DEVICES"

MODEL_NAME="optnet"
EXP_NAME="optnet_s3dis_12"
## --------------------------------------------------------- ##
DATASET="s3dis"
echo "Model Name: $MODEL_NAME"
echo "Devices: $DEVICES"

docker run --ulimit nofile=1048576:1048576 --ipc=host \
  --rm -ti \
  --gpus "\"device=${DEVICES}\"" \
  -w /working \
  -v /media/volume/data/project/semantic_3d/Pointcept/:/working \
  -v /media/volume/data/project/semantic_3d/data/:/working/data \
  -e OMP_NUM_THREADS=${OMP_NUM_THREADS} \
  "${DOCKER_IMAGE}"  \
  bash -lc "
    sh scripts/train.sh \
      -p python \
      -d ${DATASET} \
      -c ${MODEL_NAME} \
      -n ${EXP_NAME}"