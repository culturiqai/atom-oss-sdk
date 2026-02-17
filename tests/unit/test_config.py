"""Unit tests for configuration management."""

import pytest
from pydantic import ValidationError

from atom.config import (
    AtomConfig,
    BrainConfig,
    EyesConfig,
    HardwareConfig,
    LoggingConfig,
    PhysicsConfig,
)
from atom.exceptions import ConfigurationError


class TestLoggingConfig:
    """Test logging configuration."""

    def test_valid_config(self):
        """Test valid logging configuration."""
        config = LoggingConfig(level="INFO", file_path="/tmp/test.log")
        assert config.level == "INFO"
        assert config.file_path == Path("/tmp/test.log")

    def test_invalid_log_level(self):
        """Test invalid log level raises error."""
        with pytest.raises(ValidationError):
            LoggingConfig(level="INVALID")

    def test_log_level_normalization(self):
        """Test log level is normalized to uppercase."""
        config = LoggingConfig(level="info")
        assert config.level == "INFO"


class TestHardwareConfig:
    """Test hardware configuration."""

    def test_auto_device(self):
        """Test auto device detection."""
        config = HardwareConfig(device="auto")
        assert config.device == "auto"

    def test_valid_devices(self):
        """Test valid device options."""
        for device in ["cpu", "cuda", "mps"]:
            config = HardwareConfig(device=device)
            assert config.device == device

    def test_invalid_device(self):
        """Test invalid device raises error."""
        with pytest.raises(ValidationError):
            HardwareConfig(device="invalid")

    def test_memory_fraction_validation(self):
        """Test GPU memory fraction validation."""
        # Valid range
        config = HardwareConfig(gpu_memory_fraction=0.8)
        assert config.gpu_memory_fraction == 0.8

        # Invalid ranges
        with pytest.raises(ValidationError):
            HardwareConfig(gpu_memory_fraction=-0.1)

        with pytest.raises(ValidationError):
            HardwareConfig(gpu_memory_fraction=1.5)


class TestPhysicsConfig:
    """Test physics configuration."""

    def test_valid_grid_shape(self):
        """Test valid grid shape."""
        config = PhysicsConfig(grid_shape=[32, 16, 16])
        assert config.grid_shape == [32, 16, 16]

    def test_invalid_grid_shape_length(self):
        """Test invalid grid shape length."""
        with pytest.raises(ValidationError):
            PhysicsConfig(grid_shape=[32, 16])  # Too short

        with pytest.raises(ValidationError):
            PhysicsConfig(grid_shape=[32, 16, 16, 8])  # Too long

    def test_negative_grid_dimensions(self):
        """Test negative grid dimensions."""
        with pytest.raises(ValidationError):
            PhysicsConfig(grid_shape=[-32, 16, 16])

    def test_zero_grid_dimensions(self):
        """Test zero grid dimensions."""
        with pytest.raises(ValidationError):
            PhysicsConfig(grid_shape=[0, 16, 16])


class TestBrainConfig:
    """Test brain configuration."""

    def test_valid_config(self):
        """Test valid brain configuration."""
        config = BrainConfig(
            vision_dim=256,
            theory_dim=1,
            action_dim=1,
            internal_neurons=64,
            skeleton_bones=12
        )
        assert config.vision_dim == 256
        assert config.internal_neurons == 64

    def test_positive_validation(self):
        """Test positive value validation."""
        with pytest.raises(ValidationError):
            BrainConfig(vision_dim=-1)

        with pytest.raises(ValidationError):
            BrainConfig(learning_rate_actor=-0.1)

    def test_range_validation(self):
        """Test range validation for gamma and other parameters."""
        # Valid values
        config = BrainConfig(gamma=0.95, gae_lambda=0.9, clip_epsilon=0.1)
        assert config.gamma == 0.95

        # Invalid values
        with pytest.raises(ValidationError):
            BrainConfig(gamma=-0.1)  # Negative

        with pytest.raises(ValidationError):
            BrainConfig(gamma=1.5)  # > 1

        with pytest.raises(ValidationError):
            BrainConfig(clip_epsilon=-0.1)  # Negative


class TestEyesConfig:
    """Test eyes configuration."""

    def test_valid_config(self):
        """Test valid eyes configuration."""
        config = EyesConfig(fno_modes=8, fno_width=32, embedding_dim=256)
        assert config.fno_modes == 8
        assert config.embedding_dim == 256


class TestAtomConfig:
    """Test main ATOM configuration."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = AtomConfig()
        assert config.experiment_name is not None
        assert config.hardware.device == "auto"
        assert config.physics.grid_shape == [32, 16, 16]

    def test_config_with_seed(self):
        """Test configuration with random seed."""
        config = AtomConfig(seed=42)
        assert config.seed == 42

    def test_yaml_loading(self, tmp_path):
        """Test loading configuration from YAML."""
        yaml_content = """
experiment_name: test_experiment
hardware:
  device: cpu
physics:
  grid_shape: [16, 8, 8]
brain:
  vision_dim: 128
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        config = AtomConfig.from_yaml(yaml_file)
        assert config.experiment_name == "test_experiment"
        assert config.hardware.device == "cpu"
        assert config.physics.grid_shape == [16, 8, 8]
        assert config.brain.vision_dim == 128

    def test_invalid_yaml(self, tmp_path):
        """Test loading invalid YAML raises error."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("invalid: yaml: content: [")

        with pytest.raises(ConfigurationError):
            AtomConfig.from_yaml(yaml_file)

    def test_yaml_saving(self, tmp_path):
        """Test saving configuration to YAML."""
        config = AtomConfig(experiment_name="save_test")
        yaml_file = tmp_path / "saved_config.yaml"

        config.to_yaml(yaml_file)
        assert yaml_file.exists()

        # Reload and verify
        loaded_config = AtomConfig.from_yaml(yaml_file)
        assert loaded_config.experiment_name == "save_test"

    def test_device_detection(self):
        """Test device detection logic."""
        config = AtomConfig()
        device = config.get_device()

        # Should be either 'cpu', 'cuda', or 'mps'
        assert device in ["cpu", "cuda", "mps"]