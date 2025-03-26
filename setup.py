from setuptools import setup, find_namespace_packages

setup(
    name="mnist-classification",
    version="1.0.0",
    packages=find_namespace_packages(include=["src", "src.*"]),
    package_dir={"": "."},
    package_data={"": ["*.py"]},
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
        "torchinfo>=1.7.0",
    ],
    python_requires=">=3.7",
) 