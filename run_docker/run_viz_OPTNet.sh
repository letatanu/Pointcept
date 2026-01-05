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
EXP_NAME="aerorelief3d/optnet_5"
## --------------------------------------------------------- ##
DATASET="aerorelief3d"
echo "Model Name: $MODEL_NAME"
echo "Devices: $DEVICES"

docker run --ulimit nofile=1048576:1048576 --ipc=host \
  --rm -ti \
  --gpus "\"device=${DEVICES}\"" \
  -w /working \
  -v /media/volume/data_cvpr/project/semantic_3d/Pointcept/:/working \
  -e OMP_NUM_THREADS=${OMP_NUM_THREADS} \
  "${DOCKER_IMAGE}"  bash -lc "
    python -m pointcept.datasets.preprocessing.aerorelief3d.npy2ply --config-file configs/aerorelief3d/optnet.py --exp-name ${EXP_NAME}"