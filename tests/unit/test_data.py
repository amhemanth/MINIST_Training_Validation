"""
Unit tests for data loading functionality
"""

import torch
from src.data.mnist_data import get_data_loaders

def test_data_loading():
    """Test if data loaders are created correctly"""
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=32)
    
    # Check if data loaders are created
    assert train_loader is not None, "Training loader should be created"
    assert val_loader is not None, "Validation loader should be created"
    assert test_loader is not None, "Test loader should be created"
    
    # Check batch size
    for data, target in train_loader:
        assert data.shape[0] == 32, "Training batch size should be 32"
        break
        
    for data, target in val_loader:
        assert data.shape[0] <= 32, "Validation batch size should be <= 32"
        break
        
    for data, target in test_loader:
        assert data.shape[0] == 32, "Test batch size should be 32"
        break

def test_data_shape():
    """Test if data has correct shape and type"""
    train_loader, _, _ = get_data_loaders(batch_size=1)
    
    for data, target in train_loader:
        assert data.shape == (1, 1, 28, 28), "Data should be 28x28 grayscale images"
        assert data.dtype == torch.float32, "Data should be float32"
        assert target.dtype == torch.int64, "Target should be int64"
        break

def test_data_normalization():
    """Test if data is properly normalized"""
    train_loader, _, _ = get_data_loaders(batch_size=32)
    for data, _ in train_loader:
        min_val = (-0.1307) / 0.3081  # Normalized value for pixel value 0
        max_val = (1 - 0.1307) / 0.3081  # Normalized value for pixel value 1
        assert torch.all(data >= min_val - 1e-6) and torch.all(data <= max_val + 1e-6), \
            f"Data should be normalized to range [{min_val:.4f}, {max_val:.4f}]"
        break

def test_train_val_split():
    """Test if train/validation split is correct"""
    train_loader, val_loader, _ = get_data_loaders(batch_size=32, train_split=0.85)
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    total_size = train_size + val_size
    assert abs(train_size/total_size - 0.85) < 0.01, \
        f"Train split should be ~85%, got {train_size/total_size:.2%}"

def test_data_types():
    """Test if data and targets have correct types"""
    train_loader, _, _ = get_data_loaders(batch_size=32)
    for data, target in train_loader:
        assert data.dtype == torch.float32, f"Expected float32, got {data.dtype}"
        assert target.dtype == torch.long, f"Expected long, got {target.dtype}"
        break 