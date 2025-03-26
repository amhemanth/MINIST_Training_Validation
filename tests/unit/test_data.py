"""
Unit tests for data loading functionality
"""

import os
import sys
import torch
import pytest
import numpy as np

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.mnist_data import get_data_loaders

def test_data_loaders_creation():
    """Test if data loaders can be created"""
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=32)
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

def test_batch_shapes():
    """Test if batches have correct shapes"""
    train_loader, _, _ = get_data_loaders(batch_size=32)
    for data, target in train_loader:
        assert data.shape == (32, 1, 28, 28), f"Expected shape (32, 1, 28, 28), got {data.shape}"
        assert target.shape == (32,), f"Expected shape (32,), got {target.shape}"
        break

def test_data_normalization():
    """Test if data is properly normalized"""
    train_loader, _, _ = get_data_loaders(batch_size=32)
    for data, _ in train_loader:
        # For MNIST with mean=0.1307 and std=0.3081:
        # min_val = (0 - 0.1307) / 0.3081 ≈ -0.4242
        # max_val = (1 - 0.1307) / 0.3081 ≈ 2.8215
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