"""
Training and Testing Utilities for MNIST Model
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

def train(model, device, train_loader, optimizer, epoch, train_losses, train_acc):
    """
    Train the model for one epoch
    
    Args:
        model: PyTorch model
        device: Device to train on
        train_loader: Training data loader
        optimizer: Optimizer for training
        epoch: Current epoch number
        train_losses: List to store training losses
        train_acc: List to store training accuracies
    """
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # Predict
        pred = model(data)
        
        # Calculate loss
        loss = F.nll_loss(pred, target)
        train_losses.append(loss.item())
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Update Progress Bar
        pred = pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        pbar.set_description(
            desc=f'Epoch={epoch} Loss={loss.item():.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}'
        )
    train_acc.append(100*correct/processed)

def test(model, device, test_loader, test_losses, test_acc):
    """
    Test the model
    
    Args:
        model: PyTorch model
        device: Device to test on
        test_loader: Test data loader
        test_losses: List to store test losses
        test_acc: List to store test accuracies
    
    Returns:
        float: Test accuracy
    """
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    
    accuracy = 100. * correct / len(test_loader.dataset)
    test_acc.append(accuracy)
    
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy 