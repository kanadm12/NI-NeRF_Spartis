#!/bin/bash
# RunPod Setup Script for NI-NeRF Training
# Run this script when your RunPod instance starts

set -e

echo "=========================================="
echo "NI-NeRF RunPod Setup"
echo "=========================================="

# Update system
echo "Updating system packages..."
apt-get update -qq

# Install dependencies
echo "Installing system dependencies..."
apt-get install -y git wget unzip

# Install Python packages
echo "Installing Python packages..."
pip install --upgrade pip
pip install -r /workspace/NI-NeRF_Spartis/requirements.txt || \
    (pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
     pip install numpy opencv-python SimpleITK commentjson tqdm tensorboard pybind11 matplotlib)

# Clone repository
echo "Cloning NI-NeRF repository..."
cd /workspace
if [ ! -d "NI-NeRF_Spartis" ]; then
    git clone https://github.com/kanadm12/NI-NeRF_Spartis.git
    cd NI-NeRF_Spartis
else
    cd NI-NeRF_Spartis
    git pull
fi

# Build hash encoder
echo "Building hash encoder..."
cd hashencoder
python setup.py build_ext --inplace
cd ..

# Create directories
echo "Creating directories..."
mkdir -p checkpoint
mkdir -p result
mkdir -p data
mkdir -p model/pretrain
mkdir -p output/pretrain
mkdir -p log

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Ensure patient data is in /workspace/drr_patient_data/"
echo "   Each patient folder should contain:"
echo "   - PATIENT_ID.nii.gz"
echo "   - PATIENT_ID_lat_drr.png"
echo "   - PATIENT_ID_pa_drr.png"
echo ""
echo "2. Run preprocessing (includes vertical DRR flip):"
echo "   python preprocess_data.py --data_root /workspace/drr_patient_data --output_dir data"
echo ""
echo "3. Verify preprocessed data:"
echo "   python verify_data.py data/PATIENT_ID.pickle"
echo ""
echo "4. Train all patients:"
echo "   ./train_all_patients.sh"
echo ""
echo "5. Monitor with TensorBoard:"
echo "   tensorboard --logdir=log --bind_all"
echo ""
