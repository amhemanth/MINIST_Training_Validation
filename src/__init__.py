"""
MNIST Classification Project
Version: 1.0.0
"""

import sys
print("Python path in src/__init__.py:", sys.path)

try:
    from src.models.mnist_model import Net
except ImportError as e:
    print("Error importing Net:", e)

try:
    from src.data.mnist_data import get_data_loaders
except ImportError as e:
    print("Error importing get_data_loaders:", e)

try:
    from src.utils.training import train, test
except ImportError as e:
    print("Error importing train/test:", e)

__version__ = "1.0.0"

__all__ = ['get_data_loaders', 'Net', 'train', 'test'] 