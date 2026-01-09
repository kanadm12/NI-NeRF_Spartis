#!/bin/bash
# Batch training script for 100 patients
# Usage: ./train_all_patients.sh

set -e

TOTAL_PATIENTS=100
DATA_DIR="data"
CHECKPOINT_DIR="checkpoint"
BACKUP_DIR="${CHECKPOINT_DIR}/backup"

# Create backup directory
mkdir -p "$BACKUP_DIR"

echo "=========================================="
echo "NI-NeRF Batch Training"
echo "Training ${TOTAL_PATIENTS} patients"
echo "=========================================="

# Log file
LOG_FILE="training_log_$(date +%Y%m%d_%H%M%S).txt"

for i in $(seq 1 $TOTAL_PATIENTS); do
    echo ""
    echo "===========================================" | tee -a "$LOG_FILE"
    echo "Patient ${i} / ${TOTAL_PATIENTS}" | tee -a "$LOG_FILE"
    echo "Start Time: $(date)" | tee -a "$LOG_FILE"
    echo "===========================================" | tee -a "$LOG_FILE"
    
    # Update config.json for this patient
    PATIENT_DATA="${DATA_DIR}/patient_${i}.pickle"
    PATIENT_NAME="patient_${i}"
    
    # Check if data file exists
    if [ ! -f "$PATIENT_DATA" ]; then
        echo "WARNING: Data file not found: $PATIENT_DATA" | tee -a "$LOG_FILE"
        echo "Skipping patient ${i}" | tee -a "$LOG_FILE"
        continue
    fi
    
    # Create temporary config
    cat config.json | \
        sed "s|\"in_dir\": \".*\"|\"in_dir\": \"${PATIENT_DATA}\"|" | \
        sed "s|\"name\": \".*\"|\"name\": \"${PATIENT_NAME}\"|" \
        > config_temp.json
    
    # Train this patient
    python main.py --config config_temp.json 2>&1 | tee -a "$LOG_FILE"
    
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
    echo "Patient ${i} completed!" | tee -a "$LOG_FILE"
    
    # Clean up temp config
    rm -f config_temp.json
    
    # Optional: compress old checkpoints to save space
    if [ $((i % 10)) -eq 0 ]; then
        echo "Compressing old checkpoints..." | tee -a "$LOG_FILE"
        gzip -f ${CHECKPOINT_DIR}/patient_$((i-9))_*.pkl 2>/dev/null || true
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
