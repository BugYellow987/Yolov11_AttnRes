#!/usr/bin/env bash
# Train the YOLO11 6-CSAR + YOLA IIM instance-segmentation model.
#
# Usage:
#   nohup bash tools/train_yolo11_6csar_iim.sh /absolute/path/to/data.yaml > train_iim.log 2>&1 &
#
# Optional overrides:
#   EPOCHS=300 BATCH=2 DEVICE=0 RUN_NAME=iim-exp2 \
#     nohup bash tools/train_yolo11_6csar_iim.sh /absolute/path/to/data.yaml > train_iim.log 2>&1 &
#
# Dent hard-positive sampling is configured in data.yaml. The text file contains one reviewed image path/name/stem
# per line; every listed image must already contain a complete Dent polygon:
#   hard_positive:
#     class: Dent
#     images: dent-hard-positives.txt
#     repeat: 3
#     require_class: true

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DATA_YAML="${1:-}"
MODEL_YAML="${MODEL_YAML:-${PROJECT_ROOT}/ultralytics/cfg/models/11_myself/yolo11-6csar-iim.yaml}"
IMGSZ="${IMGSZ:-640}"
EPOCHS="${EPOCHS:-200}"
BATCH="${BATCH:-4}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-4}"
PATIENCE="${PATIENCE:-50}"
AMP="${AMP:-True}"
OUTPUT_PROJECT="${OUTPUT_PROJECT:-${PROJECT_ROOT}/runs/segment}"
RUN_NAME="${RUN_NAME:-yolo11-6csar-iim}"

if [[ -z "${DATA_YAML}" ]]; then
    echo "Error: missing dataset YAML."
    echo "Usage: bash tools/train_yolo11_6csar_iim.sh /absolute/path/to/data.yaml"
    exit 2
fi

if [[ ! -f "${DATA_YAML}" ]]; then
    echo "Error: dataset YAML not found: ${DATA_YAML}"
    exit 2
fi
DATA_YAML="$(cd -- "$(dirname -- "${DATA_YAML}")" && pwd)/$(basename -- "${DATA_YAML}")"

if [[ ! -f "${MODEL_YAML}" ]]; then
    echo "Error: model YAML not found: ${MODEL_YAML}"
    exit 2
fi

if ! command -v yolo >/dev/null 2>&1; then
    echo "Error: 'yolo' is not available in PATH."
    echo "Activate the training environment and run: python -m pip install -e '${PROJECT_ROOT}'"
    exit 127
fi

mkdir -p "${OUTPUT_PROJECT}"
cd "${PROJECT_ROOT}"

echo "Starting YOLO11 6-CSAR + IIM segmentation training"
echo "Model:   ${MODEL_YAML}"
echo "Data:    ${DATA_YAML}"
echo "Device:  ${DEVICE}"
echo "Run:     ${OUTPUT_PROJECT}/${RUN_NAME}"

exec yolo segment train \
    model="${MODEL_YAML}" \
    data="${DATA_YAML}" \
    imgsz="${IMGSZ}" \
    epochs="${EPOCHS}" \
    batch="${BATCH}" \
    device="${DEVICE}" \
    workers="${WORKERS}" \
    patience="${PATIENCE}" \
    amp="${AMP}" \
    project="${OUTPUT_PROJECT}" \
    name="${RUN_NAME}"
