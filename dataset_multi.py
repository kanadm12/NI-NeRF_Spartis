"""
Multi-patient dataset loader for NI-NeRF
Loads multiple patients for batch training
"""
from torch.utils import data
import pickle
import numpy as np
import torch
from pathlib import Path
from dataset import TrainData


class MultiPatientDataset(data.Dataset):
    """
    Dataset that loads multiple patients for joint training
    """
    def __init__(self, data_dir, num_samples, num_rays, scale_factor=1, max_patients=None):
        super().__init__()
        
        self.num_samples = num_samples
        self.num_rays = num_rays
        self.scale_factor = scale_factor
        
        # Load all patient pickle files
        data_dir = Path(data_dir)
        pickle_files = sorted(list(data_dir.glob("*.pickle")))
        
        if max_patients:
            pickle_files = pickle_files[:max_patients]
        
        print(f"Loading {len(pickle_files)} patients...")
        
        self.patients = []
        for pickle_file in pickle_files:
            try:
                with open(pickle_file, 'rb') as f:
                    patient_data = pickle.load(f)
                
                # Vertically flip DRRs during loading
                projections = patient_data['train']['projections']
                projections = np.flip(projections, axis=1).copy()
                patient_data['train']['projections'] = projections
                
                self.patients.append({
                    'name': pickle_file.stem,
                    'data': patient_data
                })
                print(f"  ✓ Loaded {pickle_file.stem}")
            except Exception as e:
                print(f"  ✗ Failed to load {pickle_file.stem}: {e}")
        
        print(f"Successfully loaded {len(self.patients)} patients")
        
        # Create index mapping: (patient_idx, view_idx)
        self.indices = []
        for p_idx, patient in enumerate(self.patients):
            num_views = patient['data']['numTrain']
            for v_idx in range(num_views):
                self.indices.append((p_idx, v_idx))
    
    def __getitem__(self, index):
        """Get a sample from a random patient and view"""
        patient_idx, view_idx = self.indices[index]
        patient = self.patients[patient_idx]
        data = patient['data']
        
        # Get geometry parameters
        DSD = data['DSD'] / 1000 * self.scale_factor
        DSO = data['DSO'] / 1000 * self.scale_factor
        num_detector = np.array(data['nDetector'])
        size_detector = np.array(data['dDetector']) / 1000 * self.scale_factor
        num_voxel = np.array(data["nVoxel"])
        size_voxel = np.array(data['dVoxel']) / 1000 * self.scale_factor
        
        # Get angle and projection
        angle = data['train']['angles'][view_idx]
        projection = torch.tensor(data['train']['projections'][view_idx]).cuda().reshape(-1)
        
        # Get detector coordinates
        coordinates = self._get_position(num_detector, DSD, angle).cuda()
        coordinate = coordinates.reshape(-1, 3)
        
        # Sample rays
        roi_index = torch.nonzero(projection > 0).squeeze()
        if len(roi_index) == 0:
            roi_index = torch.arange(len(projection))
        
        select_idx = np.random.choice(roi_index.cpu().numpy(), 
                                     size=min(self.num_rays, len(roi_index)), 
                                     replace=False)
        
        select_coordinates = coordinate[select_idx]
        select_projections = projection[select_idx]
        
        # Compute rays
        source = torch.tensor([np.cos(angle), np.sin(angle), 0.]).cuda()
        select_rays = self._get_rays(select_coordinates, source, 
                                     DSO, num_voxel, size_voxel)
        
        return select_rays, select_projections, select_coordinates
    
    def __len__(self):
        return len(self.indices)
    
    def _get_position(self, num_detector, DSD, angle):
        """Get detector pixel positions"""
        H, W = num_detector
        y, z = torch.meshgrid(torch.linspace(0, W - 1, W),
                             torch.linspace(0, H - 1, H), 
                             indexing="xy")
        x = -torch.ones_like(z)
        z = z - W/2 + 0.5
        y = y - H/2 + 0.5
        position = torch.stack((x, y/DSD/1000*self.scale_factor, 
                               -z/DSD/1000*self.scale_factor), dim=-1)
        
        # Rotate
        rotate_matrix = torch.tensor([[np.cos(angle), np.sin(angle), 0],
                                     [-np.sin(angle), np.cos(angle), 0],
                                     [0, 0, 1]], dtype=torch.float32)
        coordinate = torch.matmul(position, rotate_matrix)
        
        return coordinate
    
    def _get_rays(self, coordinate, source, DSO, num_voxel, size_voxel):
        """Generate rays from source through detector pixels"""
        source = source.expand(coordinate.shape)
        
        dis_np = np.array([num_voxel[0] * size_voxel[0] / 2, 
                          num_voxel[1] * size_voxel[1] / 2])
        dis = torch.tensor(np.linalg.norm(dis_np)).cuda()
        near = DSO - dis - 0.005
        far = dis + DSO + 0.005
        temp = torch.linspace(0., 1., self.num_samples).cuda()
        dis_samples = near * (1. - temp) + far * temp
        rays = coordinate[:,None,:] * dis_samples[None,:,None] + source[:,None,:]
        
        return rays
