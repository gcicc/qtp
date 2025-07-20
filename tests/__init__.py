"""
QTP Test Suite

Comprehensive testing framework for the Quantitative Trading Platform.
Includes unit tests, integration tests, and performance benchmarks.

Test Structure:
- unit/: Unit tests for individual modules and components
- integration/: Integration tests for multi-module workflows
- performance/: Performance benchmarks and load testing
- fixtures/: Shared test data and fixtures
- utils/: Testing utilities and helper functions

Testing Standards:
- All modules must have >90% test coverage
- Critical path functions require multiple test scenarios
- Performance tests validate system scalability
- Integration tests verify end-to-end workflows
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path for testing
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Test configuration
TEST_DATA_DIR = project_root / "tests" / "fixtures" / "data"
TEST_CONFIG_DIR = project_root / "tests" / "fixtures" / "config"

# Ensure test directories exist
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
TEST_CONFIG_DIR.mkdir(parents=True, exist_ok=True)