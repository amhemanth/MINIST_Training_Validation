"""
Integration tests for training functionality
"""

import os
import sys
import torch
import pytest
import numpy as np

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.mnist_data import get_data_loaders
from src.models.mnist_model import Net
from src.utils.training import train, test

@pytest.fixture
def model_and_data():
    """Fixture to provide model and data loaders"""
    device = torch.device("cpu")
    model = Net().to(device)
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=32)
    return model, train_loader, val_loader, test_loader, device

def test_training_step(model_and_data):
    """Test if model can perform one training step"""
    model, train_loader, _, _, device = model_and_data
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train_losses = []
    train_acc = []
    
    # Perform one training step
    train(model, device, train_loader, optimizer, epoch=1, 
          train_losses=train_losses, train_acc=train_acc)
    
    assert len(train_losses) > 0, "Training losses should be recorded"
    assert len(train_acc) > 0, "Training accuracy should be recorded"
    assert train_losses[-1] > 0, "Training loss should be positive"
    assert 0 <= train_acc[-1] <= 100, "Training accuracy should be between 0 and 100"

def test_validation_step(model_and_data):
    """Test if model can perform validation"""
    model, _, val_loader, _, device = model_and_data
    val_losses = []
    val_acc = []
    
    # Perform validation
    accuracy = test(model, device, val_loader, val_losses, val_acc)
    
    assert len(val_losses) > 0, "Validation losses should be recorded"
    assert len(val_acc) > 0, "Validation accuracy should be recorded"
    assert val_losses[-1] > 0, "Validation loss should be positive"
    assert 0 <= accuracy <= 100, "Validation accuracy should be between 0 and 100"

def test_model_save_load(model_and_data, tmp_path):
    """Test if model can be saved and loaded"""
    model, _, _, _, device = model_and_data
    save_path = tmp_path / "test_model.pt"
    
    # Save model
    torch.save(model.state_dict(), save_path)
    assert save_path.exists(), "Model should be saved"
    
    # Load model
    new_model = Net().to(device)
    new_model.load_state_dict(torch.load(save_path))
    
    # Compare model parameters
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2), "Loaded model parameters should match saved model"

def test_training_convergence(model_and_data):
    """Test if model shows signs of convergence"""
    model, train_loader, val_loader, _, device = model_and_data
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train_losses = []
    train_acc = []
    val_losses = []
    val_acc = []
    
    # Train for a few epochs
    for epoch in range(3):
        train(model, device, train_loader, optimizer, epoch, train_losses, train_acc)
        test(model, device, val_loader, val_losses, val_acc)
    
    # Check if loss is decreasing
    assert train_losses[-1] < train_losses[0], "Training loss should decrease"
    assert val_losses[-1] < val_losses[0], "Validation loss should decrease"

def test_training_loop():
    """Test complete training loop with all components"""
    device = torch.device("cpu")
    model = Net().to(device)
    train_loader, val_loader, _ = get_data_loaders(batch_size=32)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Run a complete training loop
    train_losses = []
    train_acc = []
    val_losses = []
    val_acc = []
    
    for epoch in range(2):
        train(model, device, train_loader, optimizer, epoch, train_losses, train_acc)
        test(model, device, val_loader, val_losses, val_acc)
    
    assert len(train_losses) > 0, "Training should record losses"
    assert len(val_losses) > 0, "Validation should record losses"
    assert all(0 <= acc <= 100 for acc in train_acc), "Training accuracy should be valid"
    assert all(0 <= acc <= 100 for acc in val_acc), "Validation accuracy should be valid" 