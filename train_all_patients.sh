#!/bin/bash
# Batch training script for all patients
# Usage: ./train_all_patients.sh

set -e

DATA_DIR="data"
CHECKPOINT_DIR="checkpoint"
BACKUP_DIR="${CHECKPOINT_DIR}/backup"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Count actual pickle files
PICKLE_FILES=(${DATA_DIR}/*.pickle)
TOTAL_PATIENTS=${#PICKLE_FILES[@]}

echo "=========================================="
echo "NI-NeRF Batch Training"
echo "Found ${TOTAL_PATIENTS} patients to train"
echo "=========================================="

# Log file
LOG_FILE="training_log_$(date +%Y%m%d_%H%M%S).txt"

# Train each patient
PATIENT_NUM=0
for PATIENT_DATA in ${DATA_DIR}/*.pickle; do
    PATIENT_NUM=$((PATIENT_NUM + 1))
    
    # Extract patient ID from filename
    PATIENT_NAME=$(basename "$PATIENT_DATA" .pickle)
    
    echo ""
    echo "===========================================" | tee -a "$LOG_FILE"
    echo "Patient ${PATIENT_NUM} / ${TOTAL_PATIENTS}" | tee -a "$LOG_FILE"
    echo "ID: ${PATIENT_NAME}" | tee -a "$LOG_FILE"
    echo "Data: ${PATIENT_DATA}" | tee -a "$LOG_FILE"
    echo "Start Time: $(date)" | tee -a "$LOG_FILE"
    echo "===========================================" | tee -a "$LOG_FILE"
    
    # Check if data file exists
    if [ ! -f "$PATIENT_DATA" ]; then
        echo "WARNING: Data file not found: $PATIENT_DATA" | tee -a "$LOG_FILE"
        echo "Skipping patient" | tee -a "$LOG_FILE"
        continue
    fi
    
    # Train this patient with explicit data path
    python main.py \
        --config config.json \
        --data_path "${PATIENT_DATA}" \
        --name "${PATIENT_NAME}" \
        2>&1 | tee -a "$LOG_FILE"
    
    # Backup checkpoint
    if [ -f "${CHECKPOINT_DIR}/${PATIENT_NAME}.pkl" ]; then
        cp "${CHECKPOINT_DIR}/${PATIENT_NAME}.pkl" "${BACKUP_DIR}/${PATIENT_NAME}_final.pkl"
        echo "Checkpoint backed up successfully" | tee -a "$LOG_FILE"
    fi
    
    # Backup result images
    if [ -d "result" ]; then
        mkdir -p "${BACKUP_DIR}/results"
        cp result/${PATIENT_NAME}*.jpg "${BACKUP_DIR}/results/" 2>/dev/null || true
    fi
    
    echo "End Time: $(date)" | tee -a "$LOG_FILE"
    echo "Patient ${PATIENT_NAME} completed!" | tee -a "$LOG_FILE"
    
    # Optional: compress old checkpoints to save space every 10 patients
    if [ $((PATIENT_NUM % 10)) -eq 0 ]; then
        echo "Compressing old checkpoints..." | tee -a "$LOG_FILE"
        find ${CHECKPOINT_DIR} -name "*.pkl" -type f | head -n -10 | xargs gzip -f 2>/dev/null || true
    fi
done

echo ""
echo "=========================================="
echo "All ${TOTAL_PATIENTS} patients trained!"
echo "Total Time: See $LOG_FILE"
echo "=========================================="
echo ""
echo "Checkpoints: ${CHECKPOINT_DIR}/"
echo "Backups: ${BACKUP_DIR}/"
echo "Results: ${BACKUP_DIR}/results/"
