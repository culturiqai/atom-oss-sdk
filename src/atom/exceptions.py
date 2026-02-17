"""Custom exceptions for the ATOM system.

Provides structured error handling with specific exception types for different
failure modes in the neuro-symbolic intelligence system.
"""

from typing import Any, Dict, Optional


class AtomError(Exception):
    """Base exception for all ATOM-related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ConfigurationError(AtomError):
    """Raised when there's a configuration-related error."""
    pass


class HardwareError(AtomError):
    """Raised when there's a hardware or device-related error."""
    pass


class PhysicsError(AtomError):
    """Raised when there's a physics simulation error."""
    pass


class BrainError(AtomError):
    """Raised when there's a neural controller error."""
    pass


class EyesError(AtomError):
    """Raised when there's a vision system error."""
    pass


class MemoryError(AtomError):
    """Raised when there's a memory system error."""
    pass


class ScientistError(AtomError):
    """Raised when there's a symbolic reasoning error."""
    pass


class TrainingError(AtomError):
    """Raised when there's a training loop error."""
    pass


class CheckpointError(AtomError):
    """Raised when there's a checkpointing error."""
    pass


class ValidationError(AtomError):
    """Raised when input validation fails."""
    pass


class ResourceError(AtomError):
    """Raised when there's a resource allocation error (memory, GPU, etc.)."""
    pass


class ConvergenceError(AtomError):
    """Raised when training fails to converge."""
    pass


class SymbolicRegressionError(AtomError):
    """Raised when symbolic regression fails."""
    pass


class DependencyError(AtomError):
    """Raised when a required dependency is missing or incompatible."""
    pass


class InitializationError(AtomError):
    """Raised when component initialization fails."""
    pass


class LoggingError(AtomError):
    """Raised when there's a logging system error."""
    pass


class RuntimeError(AtomError):
    """Raised when there's a runtime execution error."""
    pass


class TimeoutError(AtomError):
    """Raised when an operation times out."""
    pass


def handle_exceptions(func):
    """Decorator to handle and log exceptions in ATOM components."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AtomError:
            # Re-raise ATOM-specific errors
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise RuntimeError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                details={"original_error": type(e).__name__, "function": func.__name__}
            ) from e
    return wrapper


def validate_positive(value: float, name: str) -> float:
    """Validate that a value is positive."""
    if value <= 0:
        raise ValidationError(f"{name} must be positive", details={"value": value, "parameter": name})
    return value


def validate_range(value: float, min_val: float, max_val: float, name: str) -> float:
    """Validate that a value is within a range."""
    if not (min_val <= value <= max_val):
        raise ValidationError(
            f"{name} must be between {min_val} and {max_val}",
            details={"value": value, "min": min_val, "max": max_val, "parameter": name}
        )
    return value


def validate_shape(tensor, expected_shape: tuple, name: str) -> None:
    """Validate tensor shape."""
    if tensor.shape != expected_shape:
        raise ValidationError(
            f"{name} has incorrect shape",
            details={
                "actual_shape": tensor.shape,
                "expected_shape": expected_shape,
                "parameter": name
            }
        )


def validate_device_compatibility(device1: str, device2: str, operation: str) -> None:
    """Validate that two devices are compatible for an operation."""
    if device1 != device2:
        raise HardwareError(
            f"Device mismatch for {operation}",
            details={"device1": device1, "device2": device2, "operation": operation}
        )