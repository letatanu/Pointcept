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

MODEL_NAME="semseg-swin3d-v1m1-1-large"

## --------------------------------------------------------- ##
DATASET="binalab"
echo "Model Name: $MODEL_NAME"
echo "Devices: $DEVICES"

docker run --rm -ti \
  -v /dev/shm:/dev/shm \
  --gpus "\"device=${DEVICES}\"" \
  -w /working \
  -v /media/volume/data_cvpr/project/semantic_3d/Pointcept/:/working \
  -e OMP_NUM_THREADS=${OMP_NUM_THREADS} \
  "${DOCKER_IMAGE}" 
#   bash -lc "
#   sh scripts/train.sh \
#       -p python \
#       -d ${DATASET} \
#       -c ${MODEL_NAME} \
#       -n ${MODEL_NAME}"
#   bash -lc "
#         torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
#         -m scripts.train_OPTNet --config_file datasets/config/AeroRelief3D_overfit.py
#       "