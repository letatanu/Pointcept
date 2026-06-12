#!/usr/bin/env bash
set -e

# Set available GPUs (adjust as needed, e.g., "0" or "0,1")
DEVICES="all"
# Calculate number of processes based on devices
OMP_NUM_THREADS=4

## --------------------------------------------------------- ##
# Ensure this matches your docker image name
DOCKER_IMAGE="letatanu/pointcept1"

echo "Starting dales Training on Devices: $DEVICES"

MODEL_NAME="dales"
EXP_NAME="semseg-pt-v3-no-v2-1-half_02"

DATASET="dales"
echo "Model Name: $MODEL_NAME"
echo "Devices: $DEVICES"

docker run --ulimit nofile=1048576:1048576 --ipc=host \
  --shm-size=16g \
  --rm -ti \
  --gpus "\"device=${DEVICES}\"" \
  -w /working \
  -v /data/nhl224/code/semantic_3D/Pointcept/:/working \
  -e OMP_NUM_THREADS=${OMP_NUM_THREADS} \
  "${DOCKER_IMAGE}"    \
  # bash -lc "
  #   sh scripts/test.sh -p python -d ${DATASET} -n ${EXP_NAME} -w model_best"