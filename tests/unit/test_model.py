"""
Unit tests for MNIST model architecture
"""

import os
import sys
import torch
import pytest
import numpy as np

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.mnist_model import Net

def test_model_creation():
    """Test if model can be created"""
    model = Net()
    assert isinstance(model, Net)

def test_model_parameters():
    """Test if model has less than 20k parameters"""
    model = Net()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20000, f"Model has {total_params} parameters, should be < 20000"

def test_model_forward():
    """Test model forward pass"""
    model = Net()
    batch_size = 64
    x = torch.randn(batch_size, 1, 28, 28)  # MNIST image size
    output = model(x)
    assert output.shape == (batch_size, 10), f"Expected shape (64, 10), got {output.shape}"

def test_batch_norm_layers():
    """Test presence of batch normalization layers"""
    model = Net()
    has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert has_bn, "Model should have batch normalization layers"

def test_dropout_layers():
    """Test presence of dropout layers"""
    model = Net()
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert has_dropout, "Model should have dropout layers"

def test_gap_layer():
    """Test presence of Global Average Pooling"""
    model = Net()
    has_gap = any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in model.modules())
    assert has_gap, "Model should have Global Average Pooling layer"

def test_model_training_mode():
    """Test model training mode behavior"""
    model = Net()
    model.train()
    assert model.training, "Model should be in training mode"
    model.eval()
    assert not model.training, "Model should be in evaluation mode" 