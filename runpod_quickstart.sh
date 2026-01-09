#!/bin/bash
# Quick start script for RunPod with your specific data structure

set -e

echo "=========================================="
echo "NI-NeRF Training Setup for RunPod"
echo "=========================================="

# 1. Check data directory
if [ ! -d "/workspace/drr_patient_data" ]; then
    echo "❌ Error: /workspace/drr_patient_data not found"
    echo "Please ensure your data is uploaded to RunPod"
    exit 1
fi

# Count patients
PATIENT_COUNT=$(find /workspace/drr_patient_data -maxdepth 1 -type d | wc -l)
PATIENT_COUNT=$((PATIENT_COUNT - 1))  # Exclude parent dir
echo "Found ${PATIENT_COUNT} patient folders"

# 2. Clone repository if not exists
cd /workspace
if [ ! -d "NI-NeRF_Spartis" ]; then
    echo "Cloning repository..."
    git clone https://github.com/kanadm12/NI-NeRF_Spartis.git
fi

cd NI-NeRF_Spartis

# 3. Install dependencies
echo "Installing dependencies..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q numpy opencv-python SimpleITK commentjson tqdm tensorboard

# 4. Build hash encoder
echo "Building hash encoder..."
cd hashencoder
python setup.py build_ext --inplace 2>/dev/null || echo "Hash encoder build failed, will use PyTorch fallback"
cd ..

# 5. Create directories
mkdir -p data
mkdir -p checkpoint
mkdir -p result
mkdir -p log

# 6. Preprocess data
echo ""
echo "=========================================="
echo "Preprocessing patient data..."
echo "This will:"
echo "  - Load all patient CT volumes and DRRs"
echo "  - FLIP DRRs vertically"
echo "  - Convert to NI-NeRF format"
echo "=========================================="

python preprocess_data.py \
    --data_root /workspace/drr_patient_data \
    --output_dir /workspace/NI-NeRF_Spartis/data \
    --max_patients 100

# 7. Verify first patient
echo ""
echo "Verifying first preprocessed file..."
FIRST_FILE=$(ls data/*.pickle | head -n 1)
if [ -f "$FIRST_FILE" ]; then
    python verify_data.py "$FIRST_FILE"
else
    echo "No pickle files found after preprocessing"
fi

# 8. Update config for batch training
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Preprocessed data: $(ls data/*.pickle | wc -l) patients"
echo ""
echo "Next steps:"
echo ""
echo "Option 1 - Train single patient:"
echo "  1. Edit config.json:"
echo "     nano config.json"
echo "     Update 'in_dir' to: 'data/PATIENT_ID.pickle'"
echo "  2. Run: python main.py"
echo ""
echo "Option 2 - Train all patients (recommended):"
echo "  ./train_all_patients.sh"
echo ""
echo "Monitor with TensorBoard:"
echo "  tensorboard --logdir=log --bind_all --port=6006"
echo ""
