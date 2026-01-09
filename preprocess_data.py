"""
Preprocess patient data from folder structure to NI-NeRF format
Handles vertical flipping of DRR images
"""

import os
import numpy as np
import SimpleITK as sitk
import cv2
import pickle
from pathlib import Path
from tqdm import tqdm


def load_nifti_volume(nifti_path):
    """Load NIfTI volume and convert to numpy array"""
    image = sitk.ReadImage(str(nifti_path))
    volume = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()
    
    # Convert to [W, H, L] format and normalize
    volume = np.transpose(volume, (2, 1, 0))
    
    return volume, spacing


def load_and_flip_drr(drr_path):
    """Load DRR image and flip vertically"""
    drr = cv2.imread(str(drr_path), cv2.IMREAD_GRAYSCALE)
    
    # Vertical flip (flip along axis 0)
    drr_flipped = np.flip(drr, axis=0).copy()
    
    return drr_flipped


def preprocess_patient(patient_folder, output_dir, 
                       DSD=1000.0, DSO=500.0,
                       num_views=50):
    """
    Preprocess single patient data to NI-NeRF format
    
    Args:
        patient_folder: Path to patient folder containing .nii.gz and DRRs
        output_dir: Output directory for pickle files
        DSD: Source-to-detector distance in mm
        DSO: Source-to-origin distance in mm
        num_views: Number of projection views to generate
    """
    patient_folder = Path(patient_folder)
    patient_id = patient_folder.name
    
    print(f"\nProcessing patient: {patient_id}")
    
    # Find files
    nifti_files = list(patient_folder.glob("*.nii.gz"))
    lat_drr = list(patient_folder.glob("*_lat_drr.png"))
    pa_drr = list(patient_folder.glob("*_pa_drr.png"))
    
    if not nifti_files:
        print(f"  ❌ No .nii.gz file found")
        return False
    
    if not lat_drr or not pa_drr:
        print(f"  ❌ DRR images not found")
        return False
    
    nifti_path = nifti_files[0]
    lat_drr_path = lat_drr[0]
    pa_drr_path = pa_drr[0]
    
    # Load and process volume
    print(f"  Loading volume: {nifti_path.name}")
    volume, spacing = load_nifti_volume(nifti_path)
    
    # Load and flip DRRs
    print(f"  Loading DRRs (with vertical flip)...")
    lat_projection = load_and_flip_drr(lat_drr_path)
    pa_projection = load_and_flip_drr(pa_drr_path)
    
    # Normalize projections to [0, 1]
    lat_projection = lat_projection.astype(np.float32) / 255.0
    pa_projection = pa_projection.astype(np.float32) / 255.0
    
    # Get dimensions
    nVoxel = list(volume.shape)  # [W, H, L]
    dVoxel = list(spacing)  # [dW, dH, dL] in mm
    
    nDetector = list(lat_projection.shape)  # [H, W]
    # Assume detector pixel size based on volume size
    dDetector = [1.0, 1.0]  # mm, adjust if you know actual values
    
    # Generate angles for projections
    # 2 views: lateral (0°) and PA (90°)
    angles = [0.0, np.pi/2]  # radians
    projections = np.stack([lat_projection, pa_projection], axis=0)
    
    # If you want to interpolate to more views, uncomment:
    if num_views > 2:
        angles_interp = np.linspace(0, 2*np.pi, num_views, endpoint=False)
        # For now, just repeat the two views
        # In production, you'd generate actual interpolated views
        angles = angles_interp.tolist()
        projections_interp = []
        for i in range(num_views):
            if i % 2 == 0:
                projections_interp.append(lat_projection)
            else:
                projections_interp.append(pa_projection)
        projections = np.stack(projections_interp, axis=0)
    
    # Create data dictionary
    data = {
        'image': volume,
        'DSD': DSD,  # mm
        'DSO': DSO,  # mm
        'nDetector': nDetector,
        'dDetector': dDetector,
        'nVoxel': nVoxel,
        'dVoxel': dVoxel,
        'train': {
            'angles': angles,
            'projections': projections
        },
        'numTrain': len(angles)
    }
    
    # Save as pickle
    output_path = Path(output_dir) / f"{patient_id}.pickle"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"  ✓ Saved: {output_path}")
    print(f"    Volume shape: {volume.shape}")
    print(f"    Projections: {len(angles)} views @ {nDetector}")
    print(f"    DRRs flipped vertically: ✓")
    
    return True


def preprocess_all_patients(data_root, output_dir, max_patients=None):
    """
    Preprocess all patients in directory
    
    Args:
        data_root: Root directory containing patient folders
        output_dir: Output directory for pickle files
        max_patients: Maximum number of patients to process (None = all)
    """
    data_root = Path(data_root)
    
    # Find all patient folders
    patient_folders = [f for f in data_root.iterdir() if f.is_dir()]
    
    if max_patients:
        patient_folders = patient_folders[:max_patients]
    
    print(f"Found {len(patient_folders)} patient folders")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    success_count = 0
    
    for patient_folder in tqdm(patient_folders, desc="Processing patients"):
        try:
            if preprocess_patient(patient_folder, output_dir):
                success_count += 1
        except Exception as e:
            print(f"  ❌ Error processing {patient_folder.name}: {e}")
    
    print("\n" + "="*60)
    print(f"Successfully processed: {success_count} / {len(patient_folders)} patients")
    print("="*60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess patient data for NI-NeRF')
    parser.add_argument('--data_root', type=str, 
                       default='/workspace/drr_patient_data',
                       help='Root directory with patient folders')
    parser.add_argument('--output_dir', type=str,
                       default='/workspace/NI-NeRF_Spartis/data',
                       help='Output directory for pickle files')
    parser.add_argument('--max_patients', type=int, default=None,
                       help='Maximum patients to process (default: all)')
    parser.add_argument('--test_single', type=str, default=None,
                       help='Test on single patient folder')
    
    args = parser.parse_args()
    
    if args.test_single:
        # Test single patient
        print(f"Testing single patient: {args.test_single}")
        preprocess_patient(args.test_single, args.output_dir)
    else:
        # Process all patients
        preprocess_all_patients(args.data_root, args.output_dir, args.max_patients)
    
    print("\nDone! You can now verify the data with:")
    print(f"  python verify_data.py {args.output_dir}/PATIENT_ID.pickle")
