#!/usr/bin/env bash
set -e

# Set available GPUs (adjust as needed, e.g., "0" or "0,1")
DEVICES="1,2,3,4,5,6,7"
# Calculate number of processes based on devices
OMP_NUM_THREADS=4

## --------------------------------------------------------- ##
# Ensure this matches your docker image name
DOCKER_IMAGE="letatanu/pointcept1"

echo "Starting S3DIS Training on Devices: $DEVICES"

EXP_NAME="semseg-pt-v3-no-encoderonly-v0_01"
CONFIG_PATH="semseg-pt-v3-no-encoderonly-v0"
## --------------------------------------------------------- ##
DATASET="s3dis"
echo "Devices: $DEVICES"

docker run --ulimit nofile=1048576:1048576 --ipc=host \
  --rm -ti \
  --gpus "\"device=${DEVICES}\"" \
  -w /working \
  -v /data/nhl224/code/semantic_3D/Pointcept/:/working \
  -e OMP_NUM_THREADS=${OMP_NUM_THREADS} \
  "${DOCKER_IMAGE}"    \
  bash -lc "
    sh scripts/train.sh \
      -p python \
      -d ${DATASET} \
      -c ${CONFIG_PATH} \
      -n ${EXP_NAME}"