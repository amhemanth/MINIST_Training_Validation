"""
MNIST CNN Model Architecture

This module implements a Convolutional Neural Network (CNN) for MNIST digit classification.
The architecture is designed to be lightweight while maintaining good performance.
"""

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """CNN architecture for MNIST digit classification.
    
    This model implements a lightweight CNN architecture with the following features:
        - Less than 20,000 parameters for efficiency
        - Batch Normalization for stable training
        - Dropout for regularization
        - Global Average Pooling to reduce parameters
        
    Architecture:
        - 4 convolutional blocks with increasing channels (8->16->24->32)
        - Each block includes Conv2D -> ReLU -> BatchNorm -> Dropout -> MaxPool
        - Global Average Pooling followed by a single fully connected layer
        - Log Softmax output for 10 digit classes
        
    Attributes:
        conv1 (nn.Conv2d): First convolutional layer, 1->8 channels
        bn1 (nn.BatchNorm2d): Batch normalization for first conv block
        dropout1 (nn.Dropout): Dropout layer for first conv block
        conv2 (nn.Conv2d): Second convolutional layer, 8->16 channels
        bn2 (nn.BatchNorm2d): Batch normalization for second conv block
        dropout2 (nn.Dropout): Dropout layer for second conv block
        conv3 (nn.Conv2d): Third convolutional layer, 16->24 channels
        bn3 (nn.BatchNorm2d): Batch normalization for third conv block
        dropout3 (nn.Dropout): Dropout layer for third conv block
        conv4 (nn.Conv2d): Fourth convolutional layer, 24->32 channels
        bn4 (nn.BatchNorm2d): Batch normalization for fourth conv block
        dropout4 (nn.Dropout): Dropout layer for fourth conv block
        gap (nn.AdaptiveAvgPool2d): Global average pooling layer
        fc (nn.Linear): Final fully connected layer, 32->10 outputs
    """
    
    def __init__(self):
        """Initialize the CNN architecture with all layers."""
        super(Net, self).__init__()
        # First Block
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # Input: 28x28, Output: 28x28, RF: 3x3
        self.bn1 = nn.BatchNorm2d(8)
        self.dropout1 = nn.Dropout(0.1)
        
        # Second Block
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)  # Input: 14x14, Output: 14x14, RF: 10x10
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout(0.1)
        
        # Third Block
        self.conv3 = nn.Conv2d(16, 24, 3, padding=1)  # Input: 7x7, Output: 7x7, RF: 24x24
        self.bn3 = nn.BatchNorm2d(24)
        self.dropout3 = nn.Dropout(0.2)
        
        # Fourth Block
        self.conv4 = nn.Conv2d(24, 32, 3, padding=1)  # Input: 3x3, Output: 3x3, RF: 44x44
        self.bn4 = nn.BatchNorm2d(32)
        self.dropout4 = nn.Dropout(0.2)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Final Layer
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        """Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
                            containing normalized MNIST images.
        
        Returns:
            torch.Tensor: Log probabilities for each digit class,
                         shape (batch_size, 10)
        
        Note:
            The forward pass includes 4 convolutional blocks, each followed by
            ReLU activation, batch normalization, dropout, and max pooling.
            The final output is produced by global average pooling and a
            fully connected layer with log softmax activation.
        """
        # First Block
        x = self.dropout1(self.bn1(F.relu(self.conv1(x))))  # Output: 28x28
        x = F.max_pool2d(x, 2)  # Output: 14x14
        
        # Second Block
        x = self.dropout2(self.bn2(F.relu(self.conv2(x))))  # Output: 14x14
        x = F.max_pool2d(x, 2)  # Output: 7x7
        
        # Third Block
        x = self.dropout3(self.bn3(F.relu(self.conv3(x))))  # Output: 7x7
        x = F.max_pool2d(x, 2)  # Output: 3x3
        
        # Fourth Block
        x = self.dropout4(self.bn4(F.relu(self.conv4(x))))  # Output: 3x3
        x = F.max_pool2d(x, 2)  # Output: 1x1
        
        # Global Average Pooling
        x = self.gap(x)  # Output: 1x1
        x = x.view(-1, 32)  # Flatten: (batch_size, 32)
        
        # Final Layer
        x = self.fc(x)  # Output: (batch_size, 10)
        return F.log_softmax(x, dim=1) 