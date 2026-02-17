"""Logging system for ATOM using Loguru.

Provides structured logging with configurable levels, file rotation, and telemetry support.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from .config import get_config
from .exceptions import LoggingError


class AtomLogger:
    """ATOM logging system with structured logging and telemetry."""

    def __init__(self):
        self._configured = False
        self._telemetry_handlers = []

    def configure(self, config=None) -> None:
        """Configure logging based on provided config."""
        if config is None:
            config = get_config()

        if self._configured:
            logger.remove()  # Reset existing handlers

        # Remove default handler
        logger.remove()

        # Console handler
        console_format = config.logging.format
        logger.add(
            sys.stdout,
            level=config.logging.level,
            format=console_format,
            colorize=True,
            enqueue=True,  # Thread-safe
        )

        # File handler (if specified)
        if config.logging.file_path:
            file_path = Path(config.logging.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            logger.add(
                str(file_path),
                level=config.logging.level,
                format=config.logging.format,
                rotation=config.logging.max_file_size,
                retention=config.logging.retention,
                encoding="utf-8",
                enqueue=True,
            )

        # Add custom telemetry handlers
        for handler in self._telemetry_handlers:
            logger.add(**handler)

        self._configured = True

        # Log configuration
        logger.info("ATOM logging configured", extra={
            "level": config.logging.level,
            "file": str(config.logging.file_path) if config.logging.file_path else None,
            "telemetry_handlers": len(self._telemetry_handlers)
        })

    def add_telemetry_handler(self, handler_config: Dict[str, Any]) -> None:
        """Add a telemetry handler for monitoring systems."""
        self._telemetry_handlers.append(handler_config)

        if self._configured:
            logger.add(**handler_config)
            logger.debug("Added telemetry handler", extra={"handler": handler_config})

    def log_experiment_start(self, experiment_name: str, config_dict: Dict[str, Any]) -> None:
        """Log experiment initialization."""
        logger.info("ATOM experiment started", extra={
            "experiment": experiment_name,
            "config": config_dict,
            "timestamp": None  # Will be added by loguru
        })

    def log_training_step(self, step: int, metrics: Dict[str, Any]) -> None:
        """Log training step metrics."""
        logger.info("Training step completed", extra={
            "step": step,
            "metrics": metrics
        })

    def log_physics_simulation(self, step: int, physics_metrics: Dict[str, Any]) -> None:
        """Log physics simulation metrics."""
        logger.debug("Physics simulation step", extra={
            "step": step,
            "physics": physics_metrics
        })

    def log_symbolic_discovery(self, law: str, score: float, complexity: int) -> None:
        """Log symbolic law discovery."""
        logger.info("Symbolic law discovered", extra={
            "law": law,
            "score": score,
            "complexity": complexity
        })

    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Log error with context."""
        context = context or {}
        logger.error("ATOM error occurred", extra={
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        })

    def log_performance_metric(self, metric_name: str, value: float, unit: str = "") -> None:
        """Log performance metric for monitoring."""
        logger.info("Performance metric", extra={
            "metric": metric_name,
            "value": value,
            "unit": unit
        })

    def create_child_logger(self, name: str) -> "AtomChildLogger":
        """Create a child logger with specific context."""
        return AtomChildLogger(name, self)


class AtomChildLogger:
    """Child logger with specific context."""

    def __init__(self, name: str, parent: AtomLogger):
        self.name = name
        self.parent = parent
        self.logger = logger.bind(component=name)

    def info(self, message: str, **kwargs) -> None:
        """Log info message with context."""
        self.logger.info(message, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with context."""
        self.logger.debug(message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with context."""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message with context."""
        self.logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message with context."""
        self.logger.critical(message, **kwargs)


# Global logger instance
atom_logger = AtomLogger()


def get_logger(name: Optional[str] = None) -> AtomChildLogger:
    """Get a logger instance for a specific component."""
    if name:
        return atom_logger.create_child_logger(name)
    return AtomChildLogger("atom", atom_logger)


def setup_logging(config=None) -> None:
    """Setup logging for the entire ATOM system."""
    atom_logger.configure(config)


# Convenience functions
def log_experiment_start(experiment_name: str, config_dict: Dict[str, Any]) -> None:
    """Log experiment start."""
    atom_logger.log_experiment_start(experiment_name, config_dict)


def log_training_metrics(step: int, metrics: Dict[str, Any]) -> None:
    """Log training metrics."""
    atom_logger.log_training_step(step, metrics)


def log_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """Log error with context."""
    atom_logger.log_error(error, context)


# Configure on import
try:
    setup_logging()
except Exception as e:
    # Fallback to basic logging if configuration fails
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.error(f"Failed to configure ATOM logging: {e}")