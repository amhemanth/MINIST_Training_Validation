"""
Test runner script for MNIST Classification project
"""

import argparse
import subprocess
import sys

def run_command(command):
    """Run a command and return its exit code"""
    try:
        subprocess.run(command, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return e.returncode

def main(args):
    if args.all or args.quick:
        # Quick model validation
        print("\n=== Running Quick Model Validation ===")
        if run_command(["python", "src/training/train_mnist.py", "--quick-test"]) != 0:
            return 1

    if args.all or args.unit:
        # Run unit tests
        print("\n=== Running Unit Tests ===")
        if run_command(["pytest", "tests/unit/"]) != 0:
            return 1

    if args.all or args.integration:
        # Run integration tests
        print("\n=== Running Integration Tests ===")
        if run_command(["pytest", "tests/integration/"]) != 0:
            return 1

    if args.all or args.coverage:
        # Run coverage report
        print("\n=== Running Coverage Report ===")
        if run_command(["pytest", "--cov=src", "tests/", "--cov-report=term-missing"]) != 0:
            return 1

    if args.all or args.quality:
        # Run code quality checks
        print("\n=== Running Code Quality Checks ===")
        commands = [
            ["black", "--check", "src", "tests"],
            ["isort", "--check-only", "src", "tests"],
            ["flake8", "src", "tests"]
        ]
        for cmd in commands:
            if run_command(cmd) != 0:
                return 1

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for MNIST Classification project")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--quick", action="store_true", help="Run quick model validation")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--coverage", action="store_true", help="Run coverage report")
    parser.add_argument("--quality", action="store_true", help="Run code quality checks")

    args = parser.parse_args()
    
    # If no arguments provided, run all tests
    if not any(vars(args).values()):
        args.all = True

    sys.exit(main(args)) 