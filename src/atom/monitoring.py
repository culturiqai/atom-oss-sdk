"""Monitoring and metrics system for ATOM.

Provides Prometheus metrics, health checks, and performance monitoring.
"""

import time
from typing import Dict, Optional

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Summary,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from .config import get_config
from .logging import get_logger

logger = get_logger("monitoring")


class AtomMetrics:
    """ATOM metrics collection and exposure."""

    def __init__(self):
        if not HAS_PROMETHEUS:
            logger.warning("Prometheus client not available, metrics disabled")
            return

        # Training metrics
        self.training_step = Counter(
            'atom_training_step_total',
            'Total number of training steps completed'
        )

        self.training_reward = Gauge(
            'atom_training_reward',
            'Current training reward'
        )

        self.training_loss = Gauge(
            'atom_training_loss',
            'Current training loss',
            ['loss_type']  # actor, critic, total
        )

        self.training_stress = Gauge(
            'atom_training_stress',
            'Neural network structural stress'
        )

        # Physics simulation metrics
        self.physics_simulations = Counter(
            'atom_physics_simulations_total',
            'Total number of physics simulations run'
        )

        self.physics_mlups = Gauge(
            'atom_physics_mlups',
            'Physics simulation performance in MLUPS'
        )

        # Symbolic reasoning metrics
        self.symbolic_discoveries = Counter(
            'atom_symbolic_discoveries_total',
            'Total number of symbolic laws discovered'
        )

        self.symbolic_complexity = Gauge(
            'atom_symbolic_complexity',
            'Complexity of current best symbolic law'
        )

        self.symbolic_score = Gauge(
            'atom_symbolic_score',
            'Score of current best symbolic law'
        )

        # System resource metrics
        self.memory_usage = Gauge(
            'atom_memory_usage_bytes',
            'Memory usage in bytes',
            ['type']  # rss, vms, gpu
        )

        self.cpu_usage = Gauge(
            'atom_cpu_usage_percent',
            'CPU usage percentage'
        )

        self.gpu_usage = Gauge(
            'atom_gpu_usage_percent',
            'GPU usage percentage'
        )

        # Checkpointing metrics
        self.checkpoints_saved = Counter(
            'atom_checkpoints_saved_total',
            'Total number of checkpoints saved'
        )

        self.checkpoints_loaded = Counter(
            'atom_checkpoints_loaded_total',
            'Total number of checkpoints loaded'
        )

        # Error metrics
        self.errors_total = Counter(
            'atom_errors_total',
            'Total number of errors',
            ['error_type']
        )

        # Performance histograms
        self.step_duration = Histogram(
            'atom_step_duration_seconds',
            'Time taken for training steps',
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
        )

        self.simulation_duration = Histogram(
            'atom_simulation_duration_seconds',
            'Time taken for physics simulations',
            buckets=(0.01, 0.1, 0.5, 1.0, 2.0, 5.0)
        )

        logger.info("ATOM metrics initialized")

    def record_training_metrics(
        self,
        step: int,
        reward: float,
        losses: Dict[str, float],
        stress: float
    ):
        """Record training-related metrics."""
        if not HAS_PROMETHEUS:
            return

        self.training_step.inc()
        self.training_reward.set(reward)
        self.training_stress.set(stress)

        for loss_type, loss_value in losses.items():
            self.training_loss.labels(loss_type=loss_type).set(loss_value)

    def record_physics_metrics(self, mlups: float):
        """Record physics simulation metrics."""
        if not HAS_PROMETHEUS:
            return

        self.physics_simulations.inc()
        self.physics_mlups.set(mlups)

    def record_symbolic_metrics(
        self,
        discoveries: int = 1,
        complexity: Optional[float] = None,
        score: Optional[float] = None
    ):
        """Record symbolic reasoning metrics."""
        if not HAS_PROMETHEUS:
            return

        self.symbolic_discoveries.inc(discoveries)
        if complexity is not None:
            self.symbolic_complexity.set(complexity)
        if score is not None:
            self.symbolic_score.set(score)

    def record_system_metrics(self):
        """Record system resource metrics."""
        if not HAS_PROMETHEUS or not HAS_PSUTIL:
            return

        try:
            process = psutil.Process()

            # Memory metrics
            memory_info = process.memory_info()
            self.memory_usage.labels(type='rss').set(memory_info.rss)
            self.memory_usage.labels(type='vms').set(memory_info.vms)

            # CPU metrics
            cpu_percent = process.cpu_percent()
            self.cpu_usage.set(cpu_percent)

        except Exception as e:
            logger.warning(f"Failed to record system metrics: {e}")

    def record_error(self, error_type: str):
        """Record error occurrence."""
        if not HAS_PROMETHEUS:
            return

        self.errors_total.labels(error_type=error_type).inc()

    def time_step(self):
        """Context manager for timing training steps."""
        if not HAS_PROMETHEUS:
            return _NullTimer()

        return StepTimer(self.step_duration)

    def time_simulation(self):
        """Context manager for timing physics simulations."""
        if not HAS_PROMETHEUS:
            return _NullTimer()

        return StepTimer(self.simulation_duration)


class StepTimer:
    """Context manager for timing operations."""

    def __init__(self, histogram):
        self.histogram = histogram
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.histogram.observe(duration)


class _NullTimer:
    """Null timer for when Prometheus is not available."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class HealthChecker:
    """Health check system for ATOM components."""

    def __init__(self):
        self.components = {}
        self.last_checks = {}

    def register_component(self, name: str, check_func):
        """Register a component health check function."""
        self.components[name] = check_func
        logger.debug(f"Registered health check for component: {name}")

    def check_component(self, name: str) -> Dict[str, any]:
        """Check health of a specific component."""
        if name not in self.components:
            return {
                "status": "unknown",
                "message": f"No health check registered for {name}",
                "timestamp": time.time()
            }

        try:
            result = self.components[name]()
            self.last_checks[name] = {
                "status": result.get("status", "unknown"),
                "message": result.get("message", ""),
                "timestamp": time.time(),
                "details": result.get("details", {})
            }
            return self.last_checks[name]
        except Exception as e:
            error_result = {
                "status": "error",
                "message": f"Health check failed: {e}",
                "timestamp": time.time()
            }
            self.last_checks[name] = error_result
            return error_result

    def check_all(self) -> Dict[str, Dict]:
        """Check health of all registered components."""
        results = {}
        for name in self.components:
            results[name] = self.check_component(name)
        return results

    def is_healthy(self) -> bool:
        """Check if all components are healthy."""
        all_checks = self.check_all()
        return all(
            check["status"] in ["healthy", "ok"]
            for check in all_checks.values()
        )


# Global instances
metrics = AtomMetrics()
health_checker = HealthChecker()


def get_metrics() -> AtomMetrics:
    """Get the global metrics instance."""
    return metrics


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    return health_checker


def get_prometheus_metrics() -> str:
    """Get Prometheus metrics as string."""
    if not HAS_PROMETHEUS:
        return "# Prometheus client not available"

    return generate_latest().decode('utf-8')


# Default health check functions
def brain_health_check():
    """Check brain component health."""
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Try a simple tensor operation
        x = torch.randn(10, 10).to(device)
        y = torch.matmul(x, x.t())
        del x, y

        return {
            "status": "healthy",
            "message": f"Brain healthy on {device}",
            "details": {"device": str(device)}
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Brain health check failed: {e}",
            "details": {"error": str(e)}
        }


def config_health_check():
    """Check configuration health."""
    try:
        from .config import get_config
        config = get_config()

        # Check required fields
        required_fields = ["experiment_name", "hardware", "physics", "brain"]
        missing_fields = []

        for field in required_fields:
            if not hasattr(config, field):
                missing_fields.append(field)

        if missing_fields:
            return {
                "status": "error",
                "message": f"Missing required config fields: {missing_fields}",
                "details": {"missing_fields": missing_fields}
            }

        return {
            "status": "healthy",
            "message": "Configuration is valid",
            "details": {"experiment": config.experiment_name}
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Configuration health check failed: {e}",
            "details": {"error": str(e)}
        }


# Register default health checks
health_checker.register_component("brain", brain_health_check)
health_checker.register_component("config", config_health_check)


if __name__ == "__main__":
    # Simple health check script
    import json

    print("ATOM Health Check")
    print("=" * 50)

    results = health_checker.check_all()

    for component, result in results.items():
        status = result["status"]
        message = result["message"]
        emoji = "✅" if status in ["healthy", "ok"] else "❌" if status == "error" else "⚠️"

        print(f"{emoji} {component}: {message}")

    print("\nMetrics Sample:")
    if HAS_PROMETHEUS:
        sample_metrics = get_prometheus_metrics()
        # Show first few lines
        lines = sample_metrics.split('\n')[:10]
        print('\n'.join(lines))
        print("...")
    else:
        print("Prometheus not available")