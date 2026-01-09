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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy opencv-python SimpleITK commentjson tqdm tensorboard

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
echo "1. Upload your data to /workspace/NI-NeRF_Spartis/data/"
echo "2. Edit config.json with your data path"
echo "3. Optional: Run pre-training with: python pretrain.py"
echo "4. Start training with: python main.py"
echo ""
echo "Monitor training with TensorBoard:"
echo "  tensorboard --logdir=log --bind_all"
echo ""
