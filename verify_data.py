"""
Verify data format for NI-NeRF training
Run this to check if your pickle files are correctly formatted
"""

import pickle
import numpy as np
import sys

def verify_pickle_file(filepath):
    """Verify pickle file has correct format for NI-NeRF"""
    
    print(f"\n{'='*60}")
    print(f"Verifying: {filepath}")
    print('='*60)
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False
    
    # Check required keys
    required_keys = ['image', 'DSD', 'DSO', 'nDetector', 'dDetector', 
                     'nVoxel', 'dVoxel', 'train']
    
    print("\n1. Checking required keys...")
    for key in required_keys:
        if key in data:
            print(f"   ✓ {key}")
        else:
            print(f"   ❌ Missing key: {key}")
            return False
    
    # Check train subkeys
    print("\n2. Checking train data...")
    train_keys = ['angles', 'projections']
    for key in train_keys:
        if key in data['train']:
            print(f"   ✓ train.{key}")
        else:
            print(f"   ❌ Missing key: train.{key}")
            return False
    
    # Check data types and shapes
    print("\n3. Checking data formats...")
    
    # Image
    if isinstance(data['image'], np.ndarray):
        print(f"   ✓ image: {data['image'].shape} {data['image'].dtype}")
        print(f"     Range: [{data['image'].min():.3f}, {data['image'].max():.3f}]")
    else:
        print(f"   ❌ image should be numpy array, got {type(data['image'])}")
        return False
    
    # Geometry parameters
    print(f"   ✓ DSD: {data['DSD']} mm")
    print(f"   ✓ DSO: {data['DSO']} mm")
    print(f"   ✓ nDetector: {data['nDetector']}")
    print(f"   ✓ dDetector: {data['dDetector']} mm")
    print(f"   ✓ nVoxel: {data['nVoxel']}")
    print(f"   ✓ dVoxel: {data['dVoxel']} mm")
    
    # Projections
    angles = data['train']['angles']
    projections = data['train']['projections']
    
    print(f"\n4. Checking projection data...")
    print(f"   ✓ Number of projections: {len(angles)}")
    print(f"   ✓ Angles range: [{min(angles):.3f}, {max(angles):.3f}] radians")
    
    if isinstance(projections, np.ndarray):
        print(f"   ✓ Projections shape: {projections.shape}")
        print(f"     Range: [{projections.min():.3f}, {projections.max():.3f}]")
        
        if projections.min() < 0:
            print(f"   ⚠️  WARNING: Negative projection values detected!")
            print(f"      X-ray projections should typically be positive")
    else:
        print(f"   ❌ projections should be numpy array")
        return False
    
    # Verify dimensions match
    print(f"\n5. Checking dimension consistency...")
    expected_proj_shape = (len(angles), data['nDetector'][0], data['nDetector'][1])
    if projections.shape == expected_proj_shape:
        print(f"   ✓ Projection dimensions match: {expected_proj_shape}")
    else:
        print(f"   ❌ Projection shape mismatch!")
        print(f"      Expected: {expected_proj_shape}")
        print(f"      Got: {projections.shape}")
        return False
    
    # Check for common issues
    print(f"\n6. Checking for common issues...")
    
    issues = []
    if np.any(np.isnan(data['image'])):
        issues.append("NaN values in image")
    if np.any(np.isnan(projections)):
        issues.append("NaN values in projections")
    if np.any(np.isinf(data['image'])):
        issues.append("Inf values in image")
    if np.any(np.isinf(projections)):
        issues.append("Inf values in projections")
    
    if issues:
        for issue in issues:
            print(f"   ⚠️  {issue}")
    else:
        print(f"   ✓ No NaN or Inf values detected")
    
    # Summary
    print(f"\n{'='*60}")
    if not issues:
        print("✅ File format is CORRECT and ready for training!")
    else:
        print("⚠️  File has some issues but may still work")
    print('='*60)
    
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python verify_data.py <path_to_pickle_file>")
        print("\nExample:")
        print("  python verify_data.py data/patient_1.pickle")
        sys.exit(1)
    
    filepath = sys.argv[1]
    verify_pickle_file(filepath)
