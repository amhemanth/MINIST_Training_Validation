"""
Training and Testing Utilities for MNIST Model

This module provides utility functions for training and evaluating the MNIST model.
It includes functions for training epochs and model evaluation on test/validation sets.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

def train(model, device, train_loader, optimizer, epoch, log_interval=10):
    """Train the model for one epoch.
    
    This function performs one complete training epoch, including:
        - Forward pass through the model
        - Loss calculation using negative log likelihood
        - Backward pass for gradient computation
        - Optimizer step for parameter updates
        - Progress logging at specified intervals
    
    Args:
        model (torch.nn.Module): The neural network model to train
        device (torch.device): Device to train on (cuda/cpu)
        train_loader (torch.utils.data.DataLoader): DataLoader for training data
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters
        epoch (int): Current epoch number (for logging)
        log_interval (int, optional): How often to log progress. Defaults to 10 batches.
    
    Note:
        The function automatically handles device placement of data and targets.
        Progress is printed showing the current batch, total samples processed,
        percentage completion, and current loss value.
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            progress = batch_idx * len(data)
            total = len(train_loader.dataset)
            print(
                f'Train Epoch: {epoch} [{progress}/{total} '
                f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}'
            )

def test(model, device, test_loader):
    """Evaluate the model on a test or validation set.
    
    This function performs model evaluation, including:
        - Forward pass through the model in evaluation mode
        - Loss calculation using negative log likelihood
        - Accuracy computation
        - Detailed metrics logging
    
    Args:
        model (torch.nn.Module): The neural network model to evaluate
        device (torch.device): Device to evaluate on (cuda/cpu)
        test_loader (torch.utils.data.DataLoader): DataLoader for test/validation data
    
    Returns:
        float: The accuracy percentage on the test set
    
    Note:
        The function automatically handles device placement of data and targets.
        The model is set to evaluation mode, disabling dropout and using
        batch statistics for batch normalization.
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # Sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(
        f'\nTest set: Average loss: {test_loss:.4f}, '
        f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n'
    )
    return accuracy 