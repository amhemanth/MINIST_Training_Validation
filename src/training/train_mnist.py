"""
Main training script for MNIST model
"""

import argparse
import os
import sys
import torch
import torch.optim as optim

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.mnist_model import Net
from src.data.mnist_data import get_data_loaders
from src.utils.training import train, test

def main(args):
    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        train_split=args.train_split,
        seed=args.seed
    )
    
    # Initialize model
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    # Training metrics
    train_losses = []
    train_acc = []
    val_losses = []
    val_acc = []
    
    # Training loop
    best_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}")
        train(model, device, train_loader, optimizer, epoch, train_losses, train_acc)
        
        # Validation
        print("\nValidation:")
        val_accuracy = test(model, device, val_loader, val_losses, val_acc)
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), "checkpoints/mnist_model.pt")
    
    # Load best model and test
    model.load_state_dict(torch.load("checkpoints/mnist_model.pt"))
    print("\nFinal Test Results:")
    test_losses = []
    test_acc = []
    test(model, device, test_loader, test_losses, test_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--train-split', type=float, default=0.85,
                        help='proportion of training data to use for training (default: 0.85)')
    parser.add_argument('--quick-test', action='store_true', default=False,
                        help='use reduced epochs and batch size for testing')
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.batch_size = 64
        args.epochs = 1
    
    main(args) 