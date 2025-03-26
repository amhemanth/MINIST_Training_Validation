"""
MNIST CNN Model Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    CNN architecture for MNIST digit classification.
    
    Requirements:
    - Parameters < 20,000
    - Uses Batch Normalization
    - Uses Dropout
    - Uses either GAP or FC layer
    """
    
    def __init__(self):
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