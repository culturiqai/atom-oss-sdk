"""Pytest configuration and fixtures for ATOM."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
import torch

from atom.config import AtomConfig
from atom.mind.memory import AtomMemory
from atom.mind.scientist import AtomScientist


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for the test session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def config() -> AtomConfig:
    """Create a test configuration."""
    config = AtomConfig(
        seed=42,
        experiment_name="test_experiment",
        hardware={"device": "cpu", "enable_mixed_precision": False},
        physics={"grid_shape": [16, 8, 8]},  # Smaller for testing
        brain={"vision_dim": 128, "internal_neurons": 32},
        eyes={"fno_modes": 4, "width": 16},
        memory={"capacity": 100, "sequence_length": 8},
        training={"max_steps": 10, "save_interval": 5},
    )
    return config


@pytest.fixture
def device(config: AtomConfig) -> torch.device:
    """Get the appropriate device for testing."""
    device_str = config.get_device()
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    return torch.device(device_str)


@pytest.fixture(autouse=True)
def set_test_environment():
    """Set environment variables for testing."""
    os.environ["ATOM_LOG_LEVEL"] = "WARNING"  # Reduce log noise during tests
    os.environ["JAX_PLATFORM_NAME"] = "cpu"  # Force CPU for JAX in tests
    yield
    # Cleanup can go here if needed


@pytest.fixture
def mock_checkpoint_dir(temp_dir: Path) -> Path:
    """Create a mock checkpoint directory."""
    checkpoint_dir = temp_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir


@pytest.fixture
def memory() -> AtomMemory:
    """Shared default memory fixture for tests that need replay state."""
    return AtomMemory(
        capacity=100,
        seq_len=8,
        obs_shape=(4, 16, 16, 16),
        act_dim=1,
        hx_dim=64,
    )


@pytest.fixture
def scientist() -> AtomScientist:
    """Shared default scientist fixture with guaranteed shutdown."""
    s = AtomScientist(variable_names=["x", "y", "z"])
    try:
        yield s
    finally:
        s.shutdown()


# Custom markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# GPU availability check
def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if GPU is not available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
