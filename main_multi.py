import torch
import torch.optim 
import cv2
import os
import numpy as np
import commentjson as json
import SimpleITK as sitk
import model
import util
from dataset_multi import MultiPatientDataset
from torch.utils import data, tensorboard
from hashencoder import HashEncoder
from level_mask import get_mask
from tqdm import tqdm
import argparse

# constrain CPU core
cpu_num = 4
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
parser.add_argument('--data_dir', type=str, required=True, help='Directory with patient pickle files')
parser.add_argument('--max_patients', type=int, default=None, help='Max patients to load')
args = parser.parse_args()

# import config
with open(args.config) as config_file:
    config = json.load(config_file)

# set GPU
gpu = config['train']['gpu']
torch.cuda.set_device(gpu)

# file root
output = config['file']['out_dir']
ckp = config['file']['model_dir']
name = 'multi_patient_model'

if not os.path.exists(ckp):
    os.makedirs(ckp)
if not os.path.exists(output):
    os.makedirs(output)

# NeRF hyperparameters
num_sample_ray = config['train']['num_sample_ray']
num_sample_point = config['train']['num_sample_point']
batch_size = config['train']['batch_size']
epochs = config['train']['epoch']
lr = config['train']['lr']
visualize = config['train']['visualize']

# init multi-patient dataset
print(f"\n{'='*60}")
print("Loading multi-patient dataset...")
print(f"{'='*60}")
train_dataset = MultiPatientDataset(args.data_dir, num_sample_point, num_sample_ray, 
                                   max_patients=args.max_patients)
TrainLoader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

print(f"\nDataset loaded:")
print(f"  Total patients: {len(train_dataset.patients)}")
print(f"  Total samples: {len(train_dataset)}")
print(f"  Batch size: {batch_size}")
print(f"  Epochs: {epochs}")
print(f"{'='*60}\n")

# construct NeRF
encoder = HashEncoder(input_dim=3, num_levels=10, level_dim=2, base_resolution=16, log2_hashmap_size=19).cuda()
NeRF = model.naf(out_size=1, hidden_dim=32, encoder=encoder)
mask = torch.ones((1,20)).cuda()

# init optimizer
optimizer = torch.optim.Adam([{'params': NeRF.parameters(), 'lr': lr},
                              {'params': encoder.parameters(), 'lr': lr}])

# load pretrain model
pre_train = config['file']['check_point']
if os.path.exists(pre_train):
    print(f"Loading pretrain weights from {pre_train}")
    pre_dict = torch.load(pre_train)
    NeRF.load_state_dict(pre_dict['NeRF'])
    encoder.load_state_dict(pre_dict['encoder'])
else:
    print("No pretrain weights found, training from scratch")

# tensorboard
tb_writer = tensorboard.SummaryWriter(log_dir='log/multi_patient')

# training loop
print("\nStarting training...")
print(f"{'='*60}")

global_step = 0
for epoch in range(epochs):
    NeRF.train()
    encoder.train()
    
    epoch_loss = 0.0
    pbar = tqdm(TrainLoader, desc=f'Epoch {epoch+1}/{epochs}')
    
    for rays, projections, coords in pbar:
        optimizer.zero_grad()
        
        # Forward pass
        B = rays.shape[0]
        rays = rays.reshape(-1, num_sample_point, 3)
        rays_input = rays.reshape(-1, 3)
        
        # Encode and predict
        encoded = encoder(rays_input, bound=1)
        density = NeRF(encoded, mask).squeeze(-1)
        density = density.reshape(B, -1, num_sample_point)
        
        # Render (Beer-Lambert law)
        rendered = util.render_density(density)
        
        # Loss
        projections = projections.reshape(B, -1)
        loss = torch.nn.functional.mse_loss(rendered, projections)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        global_step += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                         'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'})
        
        # Log to tensorboard
        if global_step % 10 == 0:
            tb_writer.add_scalar('loss/train', loss.item(), global_step)
    
    avg_loss = epoch_loss / len(TrainLoader)
    print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
    tb_writer.add_scalar('loss/epoch', avg_loss, epoch)
    
    # Learning rate decay
    if (epoch + 1) == config['train']['lr_decay_epoch']:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= config['train']['lr_decay_coefficient']
        print(f"Learning rate decayed to {optimizer.param_groups[0]['lr']:.6f}")
    
    # Save checkpoint
    if (epoch + 1) % config['train']['save_epoch'] == 0:
        checkpoint_path = os.path.join(ckp, f'{name}_epoch{epoch+1}.pkl')
        torch.save({
            'epoch': epoch + 1,
            'NeRF': NeRF.state_dict(),
            'encoder': encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"  ✓ Checkpoint saved: {checkpoint_path}")

# Save final model
final_path = os.path.join(ckp, f'{name}_final.pkl')
torch.save({
    'epoch': epochs,
    'NeRF': NeRF.state_dict(),
    'encoder': encoder.state_dict(),
    'optimizer': optimizer.state_dict(),
}, final_path)

print(f"\n{'='*60}")
print("Training completed!")
print(f"Final model saved: {final_path}")
print(f"{'='*60}")

tb_writer.close()
