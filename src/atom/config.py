"""Configuration management for ATOM system.

Provides centralized configuration with validation, type safety, and environment variable support.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, validator
from pydantic.types import PositiveFloat, PositiveInt
from pydantic_settings import BaseSettings
import torch


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    level: str = Field(default="INFO", env="ATOM_LOG_LEVEL")
    format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        env="ATOM_LOG_FORMAT"
    )
    file_path: Optional[Path] = Field(default=None, env="ATOM_LOG_FILE")
    max_file_size: str = Field(default="10 MB", env="ATOM_LOG_MAX_SIZE")
    retention: str = Field(default="30 days", env="ATOM_LOG_RETENTION")

    @validator("level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()


class HardwareConfig(BaseSettings):
    """Hardware and compute configuration."""

    device: str = Field(default="auto", env="ATOM_DEVICE")
    gpu_memory_fraction: float = Field(default=0.9, ge=0.1, le=1.0, env="ATOM_GPU_MEMORY_FRACTION")
    num_threads: Optional[int] = Field(default=None, env="ATOM_NUM_THREADS")
    enable_mixed_precision: bool = Field(default=True, env="ATOM_MIXED_PRECISION")
    enable_x64_precision: bool = Field(default=True, env="ATOM_X64_PRECISION")

    @validator("device")
    def validate_device(cls, v):
        if v == "auto":
            return v
        valid_devices = ["cpu", "cuda", "mps"]
        if v not in valid_devices:
            raise ValueError(f"Device must be one of {valid_devices} or 'auto'")
        return v


class PhysicsConfig(BaseSettings):
    """Physics simulation configuration."""

    grid_shape: List[int] = Field(default=[32, 16, 16], min_items=3, max_items=3)
    reynolds_number: PositiveFloat = Field(default=1000.0, env="ATOM_REYNOLDS")
    inlet_velocity: PositiveFloat = Field(default=0.05, env="ATOM_INLET_VELOCITY")
    viscosity: PositiveFloat = Field(default=1e-5, env="ATOM_VISCOSITY")
    density: PositiveFloat = Field(default=1.0, env="ATOM_DENSITY")
    time_step: PositiveFloat = Field(default=1e-3, env="ATOM_TIME_STEP")
    world_type: str = Field(default="fluid", env="ATOM_WORLD_TYPE")
    geometry_path: Optional[str] = Field(default=None, env="ATOM_GEOMETRY_PATH")

    @validator("grid_shape")
    def validate_grid_shape(cls, v):
        if len(v) != 3:
            raise ValueError("Grid shape must have exactly 3 dimensions")
        if any(x <= 0 for x in v):
            raise ValueError("All grid dimensions must be positive")
        return v


class BrainConfig(BaseSettings):
    """Neural controller configuration."""

    vision_dim: PositiveInt = Field(default=256, env="ATOM_VISION_DIM")
    theory_dim: PositiveInt = Field(default=1, env="ATOM_THEORY_DIM")
    action_dim: PositiveInt = Field(default=1, env="ATOM_ACTION_DIM")
    internal_neurons: PositiveInt = Field(default=64, env="ATOM_INTERNAL_NEURONS")
    skeleton_bones: PositiveInt = Field(default=12, env="ATOM_SKELETON_BONES")

    # Training hyperparameters
    learning_rate_actor: PositiveFloat = Field(default=1e-4, env="ATOM_LR_ACTOR")
    learning_rate_critic: PositiveFloat = Field(default=3e-4, env="ATOM_LR_CRITIC")
    learning_rate_eyes: PositiveFloat = Field(default=1e-4, env="ATOM_LR_EYES")

    gamma: float = Field(default=0.99, ge=0.0, le=1.0, env="ATOM_GAMMA")
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0, env="ATOM_GAE_LAMBDA")
    clip_epsilon: float = Field(default=0.2, ge=0.0, le=1.0, env="ATOM_CLIP_EPSILON")

    ppo_epochs: PositiveInt = Field(default=4, env="ATOM_PPO_EPOCHS")
    batch_size: PositiveInt = Field(default=16, env="ATOM_BATCH_SIZE")

    # Architecture choices
    use_symplectic: bool = Field(default=True, env="ATOM_USE_SYMPLECTIC")
    symplectic_dt: PositiveFloat = Field(default=0.1, env="ATOM_SYMPLECTIC_DT")
    symplectic_dissipation: float = Field(default=0.001, ge=0.0, env="ATOM_SYMPLECTIC_DISSIPATION")


class EyesConfig(BaseSettings):
    """Vision system configuration."""

    fno_modes: PositiveInt = Field(default=8, env="ATOM_FNO_MODES")
    fno_width: PositiveInt = Field(default=32, env="ATOM_FNO_WIDTH")
    fno_depth: PositiveInt = Field(default=4, env="ATOM_FNO_DEPTH")
    embedding_dim: PositiveInt = Field(default=256, env="ATOM_EMBEDDING_DIM")


class MemoryConfig(BaseSettings):
    """Experience replay configuration."""

    capacity: PositiveInt = Field(default=2000, env="ATOM_MEMORY_CAPACITY")
    sequence_length: PositiveInt = Field(default=16, env="ATOM_SEQUENCE_LENGTH")
    memory_dtype: str = Field(default="float16", env="ATOM_MEMORY_DTYPE")

    @validator("memory_dtype")
    def validate_memory_dtype(cls, v):
        valid_dtypes = ["float16", "float32", "float64"]
        if v not in valid_dtypes:
            raise ValueError(f"Memory dtype must be one of {valid_dtypes}")
        return v


class ScientistConfig(BaseSettings):
    """Symbolic reasoning configuration."""

    variable_names: List[str] = Field(
        default=[
            "action_z", "mean_speed", "turbulence", 
            "latent_0", "latent_1", "latent_2", "latent_3",
            "latent_4", "latent_5", "latent_6", "latent_7"
        ],
        env="ATOM_VARIABLE_NAMES"
    )
    memory_limit: PositiveInt = Field(default=10000, env="ATOM_SCIENTIST_MEMORY_LIMIT")
    wake_interval: PositiveFloat = Field(default=5.0, env="ATOM_WAKE_INTERVAL")  # seconds
    sleep_iterations: PositiveInt = Field(default=50, env="ATOM_SLEEP_ITERATIONS")


class TrainingConfig(BaseSettings):
    """Training loop configuration."""

    max_steps: PositiveInt = Field(default=10000, env="ATOM_MAX_STEPS")
    sleep_interval: PositiveInt = Field(default=500, env="ATOM_SLEEP_INTERVAL")
    render_interval: PositiveInt = Field(default=100, env="ATOM_RENDER_INTERVAL")
    save_interval: PositiveInt = Field(default=1000, env="ATOM_SAVE_INTERVAL")
    eval_interval: PositiveInt = Field(default=500, env="ATOM_EVAL_INTERVAL")

    # Checkpointing
    checkpoint_dir: Path = Field(default=Path("checkpoints"), env="ATOM_CHECKPOINT_DIR")
    save_best_only: bool = Field(default=True, env="ATOM_SAVE_BEST_ONLY")

    # Early stopping
    early_stopping_patience: Optional[PositiveInt] = Field(default=None, env="ATOM_EARLY_STOPPING_PATIENCE")
    early_stopping_metric: str = Field(default="reward", env="ATOM_EARLY_STOPPING_METRIC")


class MonitoringConfig(BaseSettings):
    """Monitoring and telemetry configuration."""

    enable_prometheus: bool = Field(default=False, env="ATOM_ENABLE_PROMETHEUS")
    prometheus_port: PositiveInt = Field(default=8000, env="ATOM_PROMETHEUS_PORT")
    enable_mlflow: bool = Field(default=False, env="ATOM_ENABLE_MLFLOW")
    mlflow_tracking_uri: Optional[str] = Field(default=None, env="ATOM_MLFLOW_URI")
    experiment_name: str = Field(default="atom-gpsi", env="ATOM_EXPERIMENT_NAME")


class AtomConfig(BaseSettings):
    """Main ATOM configuration combining all subsystems."""

    # Sub-configurations
    logging: LoggingConfig = LoggingConfig()
    hardware: HardwareConfig = HardwareConfig()
    physics: PhysicsConfig = PhysicsConfig()
    brain: BrainConfig = BrainConfig()
    eyes: EyesConfig = EyesConfig()
    memory: MemoryConfig = MemoryConfig()
    scientist: ScientistConfig = ScientistConfig()
    training: TrainingConfig = TrainingConfig()
    monitoring: MonitoringConfig = MonitoringConfig()

    # Global settings
    seed: Optional[int] = Field(default=None, env="ATOM_SEED")
    experiment_name: str = Field(default="default_experiment", env="ATOM_EXPERIMENT_NAME")
    output_dir: Path = Field(default=Path("outputs"), env="ATOM_OUTPUT_DIR")
    resume_from: Optional[Path] = Field(default=None, env="ATOM_RESUME_FROM")

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed if specified
        if self.seed is not None:
            import random
            import numpy as np
            import torch
            import jax

            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
            if hasattr(jax, 'random'):
                jax.random.PRNGKey(self.seed)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "AtomConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        import yaml

        with open(yaml_path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, sort_keys=False)

    def get_device(self) -> str:
        """Get the actual device to use based on configuration and availability."""
        if self.hardware.device == "auto":
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch, 'backends') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.hardware.device


# Global configuration instance
config = AtomConfig()


def get_config() -> AtomConfig:
    """Get the global configuration instance."""
    return config


def reload_config(**overrides) -> AtomConfig:
    """Reload configuration with overrides."""
    global config
    config = AtomConfig(**overrides)
    return config