from setuptools import setup, find_packages

setup(
    name="mnist-classification",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.5",
        "matplotlib>=3.4.3",
        "pytest>=6.2.5",
        "pytest-cov>=2.12.1",
        "black>=21.7b0",
        "isort>=5.9.3",
        "flake8>=3.9.2",
        "mypy>=0.910",
        "torch-model-summary>=0.1.1",
    ],
    python_requires=">=3.7",
) 