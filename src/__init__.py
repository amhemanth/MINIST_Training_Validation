"""
MNIST Classification Project
Version: 1.0.0
"""

# Import statements using absolute imports
from src.models.mnist_model import Net
from src.data.mnist_data import get_data_loaders
from src.utils.training import train, test

__version__ = "1.0.0"

__all__ = ['get_data_loaders', 'Net', 'train', 'test'] 