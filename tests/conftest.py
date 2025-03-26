"""
Pytest configuration file for handling common test fixtures and path setup.
"""

import os
import sys
import pytest
import torch

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

@pytest.fixture(scope="session")
def device():
    """Return the device to use for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="session")
def batch_size():
    """Return the batch size to use for testing."""
    return 32 