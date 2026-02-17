"""Checkpointing system for ATOM.

Provides robust saving and loading of model states, training progress, and system configuration.
Supports incremental checkpoints, best model tracking, and recovery from failures.
"""

import json
import pickle
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Union

import torch
import numpy as np

from .config import get_config
from .exceptions import CheckpointError
from .logging import get_logger

logger = get_logger("checkpoint")


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint files."""

    experiment_name: str
    timestamp: float
    step: int
    epoch: Optional[int] = None
    best_metric: Optional[float] = None
    best_metric_name: Optional[str] = None
    config_hash: Optional[str] = None
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert Path objects to strings if any exist
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMetadata":
        """Create from dictionary."""
        return cls(**data)


class Checkpointable(Protocol):
    """Protocol for objects that can be checkpointed."""

    def get_checkpoint_state(self) -> Dict[str, Any]:
        """Get state to save in checkpoint."""
        ...

    def load_checkpoint_state(self, state: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        ...


class CheckpointManager:
    """Manages saving and loading of checkpoints."""

    def __init__(self, checkpoint_dir: Union[str, Path], experiment_name: str = "default"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.experiment_dir = self.checkpoint_dir / experiment_name

        # Create directories
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.experiment_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        self.best_model_path = self.models_dir / "best_model.pth"
        self.latest_checkpoint_path = self.experiment_dir / "latest_checkpoint.json"

        logger.info("Checkpoint manager initialized", extra={
            "checkpoint_dir": str(self.checkpoint_dir),
            "experiment_name": experiment_name
        })

    def save_checkpoint(
        self,
        components: Dict[str, Checkpointable],
        metadata: CheckpointMetadata,
        is_best: bool = False
    ) -> Path:
        """Save a checkpoint with all components."""
        try:
            # Create checkpoint filename
            timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(metadata.timestamp))
            checkpoint_name = f"checkpoint_{metadata.step}_{timestamp_str}"
            checkpoint_path = self.experiment_dir / f"{checkpoint_name}.pth"

            # Collect states from all components
            checkpoint_data = {
                "metadata": metadata.to_dict(),
                "components": {}
            }

            for name, component in components.items():
                try:
                    state = component.get_checkpoint_state()
                    checkpoint_data["components"][name] = state
                    logger.debug(f"Saved state for component: {name}")
                except Exception as e:
                    logger.error(f"Failed to save state for component {name}: {e}")
                    raise CheckpointError(
                        f"Failed to save component {name}",
                        details={"component": name, "error": str(e)}
                    )

            # Save to file
            torch.save(checkpoint_data, checkpoint_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)

            # Save metadata separately for easy access
            metadata_path = checkpoint_path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)

            # Update latest checkpoint pointer
            with open(self.latest_checkpoint_path, "w") as f:
                json.dump({
                    "checkpoint_path": str(checkpoint_path),
                    "metadata": metadata.to_dict()
                }, f, indent=2)

            # Save best model if requested
            if is_best:
                shutil.copy2(checkpoint_path, self.best_model_path)
                logger.info("Saved best model checkpoint")

            logger.info("Checkpoint saved successfully", extra={
                "path": str(checkpoint_path),
                "step": metadata.step,
                "size_mb": checkpoint_path.stat().st_size / (1024 * 1024)
            })

            return checkpoint_path

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise CheckpointError(
                "Checkpoint save failed",
                details={"error": str(e), "step": metadata.step}
            )

    def load_checkpoint(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        components: Optional[Dict[str, Checkpointable]] = None
    ) -> CheckpointMetadata:
        """Load a checkpoint and restore component states."""
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()

        if checkpoint_path is None:
            raise CheckpointError("No checkpoint found to load")

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise CheckpointError(
                "Checkpoint file does not exist",
                details={"path": str(checkpoint_path)}
            )

        try:
            # Load checkpoint data
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            # Extract metadata
            metadata_dict = checkpoint_data["metadata"]
            metadata = CheckpointMetadata.from_dict(metadata_dict)

            # Restore component states
            if components:
                component_states = checkpoint_data["components"]
                for name, component in components.items():
                    if name in component_states:
                        try:
                            component.load_checkpoint_state(component_states[name])
                            logger.debug(f"Loaded state for component: {name}")
                        except Exception as e:
                            logger.error(f"Failed to load state for component {name}: {e}")
                            raise CheckpointError(
                                f"Failed to load component {name}",
                                details={"component": name, "error": str(e)}
                            )
                    else:
                        logger.warning(f"Component {name} not found in checkpoint")

            logger.info("Checkpoint loaded successfully", extra={
                "path": str(checkpoint_path),
                "step": metadata.step,
                "timestamp": metadata.timestamp
            })

            return metadata

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise CheckpointError(
                "Checkpoint load failed",
                details={"path": str(checkpoint_path), "error": str(e)}
            )

    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Find the latest checkpoint file."""
        if self.latest_checkpoint_path.exists():
            try:
                with open(self.latest_checkpoint_path, "r") as f:
                    data = json.load(f)
                return Path(data["checkpoint_path"])
            except Exception as e:
                logger.warning(f"Failed to read latest checkpoint pointer: {e}")

        # Fallback: find most recent checkpoint file
        checkpoint_files = list(self.experiment_dir.glob("checkpoint_*.pth"))
        if not checkpoint_files:
            return None

        # Sort by modification time (most recent first)
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return checkpoint_files[0]

    def list_checkpoints(self) -> list[Dict[str, Any]]:
        """List all available checkpoints."""
        checkpoints = []

        # Check for metadata files
        for metadata_file in self.experiment_dir.glob("checkpoint_*.json"):
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                checkpoints.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to read checkpoint metadata {metadata_file}: {e}")

        # Sort by timestamp (most recent first)
        checkpoints.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return checkpoints

    def cleanup_old_checkpoints(self, keep_last: int = 5) -> None:
        """Clean up old checkpoints, keeping only the most recent ones."""
        checkpoints = self.list_checkpoints()

        if len(checkpoints) <= keep_last:
            return

        # Remove old checkpoints
        for checkpoint in checkpoints[keep_last:]:
            try:
                checkpoint_path = Path(checkpoint["checkpoint_path"])
                metadata_path = checkpoint_path.with_suffix(".json")

                checkpoint_path.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)

                logger.debug(f"Removed old checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint_path}: {e}")

        logger.info(f"Cleaned up old checkpoints, kept {min(keep_last, len(checkpoints))}")


class TrainingState:
    """Training state that can be checkpointed."""

    def __init__(self):
        self.step = 0
        self.epoch = 0
        self.best_metric = float('-inf')
        self.metrics_history = []
        self.lr_schedulers_state = {}
        self.optimizers_state = {}

    def get_checkpoint_state(self) -> Dict[str, Any]:
        """Get training state for checkpointing."""
        return {
            "step": self.step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "metrics_history": self.metrics_history,
            "lr_schedulers_state": self.lr_schedulers_state,
            "optimizers_state": self.optimizers_state,
        }

    def load_checkpoint_state(self, state: Dict[str, Any]) -> None:
        """Load training state from checkpoint."""
        self.step = state.get("step", 0)
        self.epoch = state.get("epoch", 0)
        self.best_metric = state.get("best_metric", float('-inf'))
        self.metrics_history = state.get("metrics_history", [])
        self.lr_schedulers_state = state.get("lr_schedulers_state", {})
        self.optimizers_state = state.get("optimizers_state", {})


def create_checkpoint_metadata(
    experiment_name: str,
    step: int,
    epoch: Optional[int] = None,
    best_metric: Optional[float] = None,
    best_metric_name: Optional[str] = None
) -> CheckpointMetadata:
    """Create checkpoint metadata."""
    config = get_config()
    config_hash = hash(str(config.dict())) if config else None

    return CheckpointMetadata(
        experiment_name=experiment_name,
        timestamp=time.time(),
        step=step,
        epoch=epoch,
        best_metric=best_metric,
        best_metric_name=best_metric_name,
        config_hash=config_hash
    )


def save_system_checkpoint(
    checkpoint_manager: CheckpointManager,
    brain,
    memory,
    scientist,
    training_state: TrainingState,
    step: int,
    metrics: Optional[Dict[str, Any]] = None
) -> None:
    """Save a complete system checkpoint."""
    components = {
        "brain": brain,
        "memory": memory,
        "scientist": scientist,
        "training": training_state,
    }

    # Determine if this is the best checkpoint
    is_best = False
    best_metric = None
    best_metric_name = None

    if metrics:
        config = get_config()
        best_metric_name = config.training.early_stopping_metric
        if best_metric_name in metrics:
            best_metric = metrics[best_metric_name]
            is_best = best_metric > training_state.best_metric
            if is_best:
                training_state.best_metric = best_metric

    metadata = create_checkpoint_metadata(
        checkpoint_manager.experiment_name,
        step,
        training_state.epoch,
        best_metric,
        best_metric_name
    )

    checkpoint_manager.save_checkpoint(components, metadata, is_best)


def load_system_checkpoint(
    checkpoint_manager: CheckpointManager,
    brain,
    memory,
    scientist,
    training_state: TrainingState,
    checkpoint_path: Optional[Union[str, Path]] = None
) -> CheckpointMetadata:
    """Load a complete system checkpoint."""
    components = {
        "brain": brain,
        "memory": memory,
        "scientist": scientist,
        "training": training_state,
    }

    return checkpoint_manager.load_checkpoint(checkpoint_path, components)