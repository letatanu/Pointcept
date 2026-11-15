#!/usr/bin/env bash
set -e

# Set available GPUs (adjust as needed, e.g., "0" or "0,1")
DEVICES="all"
# Calculate number of processes based on devices
OMP_NUM_THREADS=4

## --------------------------------------------------------- ##
# Ensure this matches your docker image name
DOCKER_IMAGE="letatanu/pointcept1"

echo "Starting Dales Training on Devices: $DEVICES"

EXP_NAME="dales_random_rot"

## --------------------------------------------------------- ##
DATASET="utonia"
echo "Devices: $DEVICES"

docker run --init --ulimit nofile=1048576:1048576 --ipc=host \
  --rm -ti \
  --gpus "\"device=${DEVICES}\"" \
  -w /working \
  -v /media/volume/Nhut_Data-1/Pointcept/:/working \
  -e OMP_NUM_THREADS=${OMP_NUM_THREADS} \
  "${DOCKER_IMAGE}"    \
  bash -lc "
    sh scripts/test.sh -p python -d ${DATASET} -n ${EXP_NAME} -w model_best"