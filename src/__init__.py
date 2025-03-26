"""
MNIST Classification Project
Version: 1.0.0
"""

# Import statements should be relative to src
from .models.mnist_model import Net
from .data.mnist_data import get_data_loaders
from .utils.training import train, test

__version__ = "1.0.0" 