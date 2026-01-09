# NI-NeRF Training on RunPod

## Quick Start Guide

### 1. Create RunPod Instance

1. Go to [RunPod.io](https://www.runpod.io/)
2. Select **GPU Pod**
3. Choose a GPU: **RTX 3090** or **RTX 4090** (recommended)
4. Template: **PyTorch** or **CUDA** template
5. Disk: At least **50GB**
6. Start the pod

### 2. Connect to RunPod

Once your pod is running, click **Connect** → **Start Jupyter Lab** or **SSH**

### 3. Run Setup Script

```bash
# In the terminal
cd /workspace
wget https://raw.githubusercontent.com/kanadm12/NI-NeRF_Spartis/main/runpod_setup.sh
chmod +x runpod_setup.sh
./runpod_setup.sh
```

This will:
- Install all dependencies
- Clone your repository
- Build the hash encoder
- Create necessary directories

### 4. Upload Your Data

**Option A: Direct Upload (Small files)**
```bash
# Use RunPod's file browser to upload to:
/workspace/NI-NeRF_Spartis/data/
```

**Option B: Download from Cloud (Recommended)**
```bash
# Google Drive
pip install gdown
gdown YOUR_GDRIVE_LINK -O /workspace/NI-NeRF_Spartis/data/yourdata.pickle

# Or from your own server
wget YOUR_DATA_URL -O /workspace/NI-NeRF_Spartis/data/yourdata.pickle
```

### 5. Configure Training

Edit the config file:
```bash
cd /workspace/NI-NeRF_Spartis
nano config.json
```

Update these fields:
```json
{
  "file": {
    "in_dir": "data/yourdata.pickle",
    "model_dir": "checkpoint",
    "out_dir": "result",
    "name": "patient_case_1"
  },
  "train": {
    "gpu": 0,
    "epoch": 100,
    "num_sample_ray": 1024,
    "num_sample_point": 320
  }
}
```

### 6. Optional: Pre-training (MCI)

If you have a reference abdomen case for pre-training:

```bash
# Edit pre-training config
nano config_pretrain.json

# Update:
# "in_dir": "data/abdomen_reference.pickle"
# "epoch": 50

# Run pre-training
python pretrain.py
```

This creates initialization weights in `model/pretrain/`

### 7. Start Training

```bash
# Without pre-training
python main.py

# With pre-training weights
# Edit config.json first:
# "check_point": "model/pretrain/test.pkl"
python main.py
```

### 8. Monitor Training

**Option A: Watch terminal output**
```bash
# Training will show:
# - Epoch number
# - Loss value
# - Learning rate
```

**Option B: TensorBoard (Recommended)**

In a new terminal:
```bash
cd /workspace/NI-NeRF_Spartis
tensorboard --logdir=log --bind_all --port=6006
```

Then access via RunPod's port forwarding or copy the URL shown.

### 9. Download Results

After training completes:

**Checkpoints:**
```bash
/workspace/NI-NeRF_Spartis/checkpoint/
```

**Visualizations:**
```bash
/workspace/NI-NeRF_Spartis/result/
```

**Download via RunPod UI:**
- Use file browser to download files
- Or zip and download:
```bash
cd /workspace
zip -r results.zip NI-NeRF_Spartis/checkpoint NI-NeRF_Spartis/result
# Download results.zip via file browser
```

## Training for 100 Patients

### Batch Processing Approach

Since the official code trains one patient at a time, here's how to process 100 patients:

**Option 1: Sequential Training (Simplest)**

Create a batch script:
```bash
nano train_all_patients.sh
```

```bash
#!/bin/bash
# Train all 100 patients sequentially

for i in {1..100}; do
    echo "=========================================="
    echo "Training Patient $i / 100"
    echo "=========================================="
    
    # Update config for this patient
    sed -i "s/\"in_dir\": .*/\"in_dir\": \"data\/patient_${i}.pickle\",/" config.json
    sed -i "s/\"name\": .*/\"name\": \"patient_${i}\"/" config.json
    
    # Train
    python main.py
    
    # Backup checkpoint
    cp checkpoint/patient_${i}.pkl checkpoint/backup/patient_${i}_final.pkl
done

echo "All patients trained!"
```

Run it:
```bash
chmod +x train_all_patients.sh
./train_all_patients.sh
```

**Option 2: Parallel Training (Faster, needs more GPUs)**

If you have multiple GPUs on RunPod:
```bash
# Terminal 1 - GPU 0
CUDA_VISIBLE_DEVICES=0 python main.py --config config_patient1.json &

# Terminal 2 - GPU 1
CUDA_VISIBLE_DEVICES=1 python main.py --config config_patient2.json &

# etc...
```

## Cost Optimization Tips

### 1. Use Spot Instances
- 70% cheaper than on-demand
- May be interrupted (save checkpoints frequently!)

### 2. Auto-pause when idle
```bash
# Add to end of training script
shutdown -h now
```

### 3. Monitor GPU usage
```bash
watch -n 1 nvidia-smi
```

### 4. Compress checkpoints
```bash
# After training
cd checkpoint
gzip *.pkl
```

## Troubleshooting

### Hash Encoder Build Fails
```bash
# Install build tools
apt-get install -y build-essential
pip install pybind11

# Try manual build
cd hashencoder
python setup.py install
```

### CUDA Out of Memory
Edit `config.json`:
```json
"num_sample_ray": 512,    // Reduce from 1024
"num_sample_point": 160   // Reduce from 320
```

### Training Loss is NaN
- Check data normalization (projections should be positive)
- Reduce learning rate: `"lr": 5e-4`
- Enable gradient clipping in `main.py`

### Can't Access TensorBoard
```bash
# Use RunPod's HTTP port feature
# Set port 6006 to public in RunPod dashboard
tensorboard --logdir=log --host=0.0.0.0 --port=6006
```

## Expected Performance

**Single Patient (100 epochs):**
- Time: ~30-60 minutes on RTX 3090
- Checkpoint size: ~5-10 MB
- Memory usage: ~8 GB VRAM

**100 Patients Sequential:**
- Total time: ~50-100 hours
- Can pause/resume between patients
- Use spot instances + auto-save

## Advanced: Resume Training

If interrupted:
```python
# In main.py, add checkpoint loading:
if os.path.exists('checkpoint/latest.pth'):
    checkpoint = torch.load('checkpoint/latest.pth')
    NeRF.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
```

## Data Format Reminder

Your pickle files should contain:
```python
{
    'image': numpy array (W, H, L),
    'DSD': float (mm),
    'DSO': float (mm),
    'nDetector': [H, W],
    'dDetector': [dH, dW] (mm),
    'nVoxel': [W, H, L],
    'dVoxel': [dW, dH, dL] (mm),
    'train': {
        'angles': list of floats (radians),
        'projections': numpy array (N, H, W)
    }
}
```

## Support

If you encounter issues:
1. Check RunPod status and GPU availability
2. Verify data format with provided script
3. Monitor GPU memory with `nvidia-smi`
4. Check TensorBoard for training curves
