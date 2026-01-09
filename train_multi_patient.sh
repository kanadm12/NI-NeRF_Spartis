#!/bin/bash
# Train on all patients together for N epochs
# Usage: ./train_multi_patient.sh

set -e

DATA_DIR="data"
CHECKPOINT_DIR="checkpoint"
LOG_DIR="log"
RESULT_DIR="result"

# Create directories
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$RESULT_DIR"

# Count patients
PICKLE_FILES=(${DATA_DIR}/*.pickle)
TOTAL_PATIENTS=${#PICKLE_FILES[@]}

echo "=========================================="
echo "NI-NeRF Multi-Patient Training"
echo "Training on ${TOTAL_PATIENTS} patients for 100 epochs"
echo "=========================================="

# Log file
LOG_FILE="training_multi_patient_$(date +%Y%m%d_%H%M%S).txt"

echo "Start Time: $(date)" | tee -a "$LOG_FILE"

# Train on all patients
python main_multi.py \
    --config config.json \
    --data_dir data \
    --max_patients ${TOTAL_PATIENTS} \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "End Time: $(date)" | tee -a "$LOG_FILE"
echo "=========================================="
echo ""
echo "Checkpoints: ${CHECKPOINT_DIR}/"
echo "Results: ${RESULT_DIR}/"
echo "Logs: ${LOG_DIR}/"
