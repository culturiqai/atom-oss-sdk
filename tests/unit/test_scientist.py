"""Unit tests for AtomScientist (symbolic reasoning)."""

import time
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

from atom.mind.scientist import (
    AtomScientist,
    _heavy_think_global,
    _select_best_candidate,
    create_scientist_from_config,
)


class TestAtomScientist:
    """Test AtomScientist symbolic reasoning system."""

    @pytest.fixture
    def scientist(self):
        """Create a test scientist instance."""
        s = AtomScientist(variable_names=["x", "y", "z"])
        try:
            yield s
        finally:
            s.shutdown()

    def test_initialization(self, scientist):
        """Test scientist initialization."""
        assert scientist.variable_names == ["x", "y", "z"]
        assert scientist.memory_limit == 10000
        assert scientist.running is True
        assert scientist.best_law is None
        assert scientist.best_law_score == float('inf')
        assert len(scientist.theory_archive) == 0

    def test_observe_data(self, scientist):
        """Test observing data points."""
        features = [1.0, 2.0, 3.0]
        target = 5.0

        scientist.observe(features, target)

        assert len(scientist.short_term_X) == 1
        assert len(scientist.short_term_y) == 1
        assert scientist.short_term_y[0] == 5.0

    def test_observe_with_numpy_array(self, scientist):
        """Test observing with numpy arrays."""
        features = np.array([1.0, 2.0, 3.0])
        target = 5.0

        scientist.observe(features, target)

        assert len(scientist.short_term_X) == 1
        np.testing.assert_array_equal(scientist.short_term_X[0], features)

    def test_memory_limit(self, scientist):
        """Test memory limit enforcement."""
        # Add more data than the limit
        for i in range(scientist.memory_limit + 10):
            features = [float(i), 2.0, 3.0]
            target = float(i)
            scientist.observe(features, target)

        # Should maintain memory limit
        assert len(scientist.short_term_X) == scientist.memory_limit
        assert len(scientist.short_term_y) == scientist.memory_limit

    def test_predict_without_law(self, scientist):
        """Test prediction when no law is available."""
        features = [1.0, 2.0, 3.0]
        prediction, trust = scientist.predict_theory(features)

        assert prediction == 0.0
        assert trust == 0.0

    @patch('atom.mind.scientist.AtomScientist._compile_new_law')
    def test_process_discovery(self, mock_compile, scientist):
        """Test processing a new discovery."""
        scientist._process_discovery("x + y", 0.5)

        # Should call compile with correct parameters
        mock_compile.assert_called_once_with("x + y", 0.5, 3)

    def test_predict_with_compiled_law(self, scientist):
        """Test prediction with a compiled law."""
        # Manually set a compiled law
        scientist.current_law_func = lambda x, y, z: x + y + z
        scientist.best_law = "x + y + z"

        features = [1.0, 2.0, 3.0]
        prediction, trust = scientist.predict_theory(features)

        assert prediction == 6.0
        assert 0.0 <= trust <= 1.0

    def test_predict_clipping(self, scientist):
        """Test prediction value clipping."""
        scientist.current_law_func = lambda x, y, z: x * 100  # Large value

        features = [100.0, 2.0, 3.0]
        prediction, trust = scientist.predict_theory(features)

        # Should be clipped to [-10, 10]
        assert prediction == 10.0
        assert 0.0 <= trust <= 1.0

    def test_predict_error_handling(self, scientist):
        """Test prediction error handling."""
        scientist.current_law_func = lambda x, y, z: 1 / 0  # Will raise error

        features = [1.0, 2.0, 3.0]
        prediction, trust = scientist.predict_theory(features)

        # Should return 0.0 on error
        assert prediction == 0.0
        assert trust == 0.0

    def test_compile_new_law(self, scientist):
        """Test compiling a new mathematical law."""
        law_str = "x**2 + y"
        score = 0.8

        scientist._compile_new_law(law_str, score, 3)

        assert scientist.best_law == law_str
        assert scientist.best_law_score == score
        assert len(scientist.theory_archive) == 1
        assert scientist.theory_archive[0] == (law_str, score)
        assert scientist.new_discovery_alert == law_str
        assert scientist.current_law_func is not None

    def test_compile_invalid_law(self, scientist):
        """Test compiling an invalid mathematical expression."""
        invalid_law = "invalid syntax +++"
        compile_fail_before = scientist.get_stats()["compile_fail"]
        scientist._compile_new_law(invalid_law, 0.5, 2)
        compile_fail_after = scientist.get_stats()["compile_fail"]
        assert compile_fail_after == compile_fail_before + 1

    def test_get_stats(self, scientist):
        """Test getting scientist statistics."""
        stats = scientist.get_stats()

        expected_keys = [
            "best_law", "best_score", "archive_size",
            "short_term_samples", "pysr_initialized", "running"
        ]

        for key in expected_keys:
            assert key in stats

        assert stats["archive_size"] == 0
        assert stats["short_term_samples"] == 0
        assert stats["running"] is True

    def test_shutdown(self, scientist):
        """Test scientist shutdown."""
        assert scientist.running is True

        scientist.shutdown()

        assert scientist.running is False

    def test_fit_offline_with_mock(self, scientist):
        """Test offline fitting with mocked PySR."""
        # Mock the model directly (fit_offline should integrate discovery path).
        mock_model = MagicMock()
        mock_model.get_best.return_value = {"sympy_format": "x + y", "score": 0.7}
        mock_model.equations_ = object()
        scientist.model = mock_model
        scientist.conformal = None
        scientist.pysr_initialized = True

        X = np.random.rand(50, 3)
        y = np.random.rand(50)

        scientist.fit_offline(X, y)

        # Should have updated the best law
        assert scientist.best_law == "x + y"
        assert scientist.best_law_score == 0.7

    def test_fit_offline_without_pysr(self, scientist):
        """Test offline fitting when PySR is not available."""
        scientist.pysr_initialized = False

        # Mock import error
        with patch.dict('sys.modules', {'pysr': None}):
            X = np.random.rand(10, 3)
            y = np.random.rand(10)

            # Should not raise error
            scientist.fit_offline(X, y)

    def test_ponder_no_discovery(self, scientist):
        """Test pondering when no new discovery is available."""
        result = scientist.ponder()

        assert result is None

    def test_ponder_with_discovery(self, scientist):
        """Test pondering when a discovery is available."""
        scientist.new_discovery_alert = "x**2"

        result = scientist.ponder()

        assert result == "x**2"
        assert scientist.new_discovery_alert is None

    def test_save_laws(self, scientist, tmp_path):
        """Test saving discovered laws."""
        # Add some laws
        scientist.theory_archive = [("x + y", 0.8), ("x**2", 0.6)]
        scientist.best_law = "x + y"

        laws_file = tmp_path / "test_laws.txt"
        scientist.save_laws(laws_file)

        assert laws_file.exists()

        content = laws_file.read_text()
        assert "ATOM Discovered Laws" in content
        assert "Best Law" in content
        assert "x + y" in content
        assert "x**2" in content


class TestScientistConfiguration:
    """Test scientist configuration integration."""

    def test_create_from_config(self):
        """Test creating scientist from configuration."""
        from atom.config import AtomConfig

        config = AtomConfig(
            scientist={
                "variable_names": ["a", "b", "c"],
                "memory_limit": 5000,
                "wake_interval": 10.0
            }
        )

        # Temporarily set as global config
        from atom.config import config as global_config
        original_config = global_config
        try:
            import atom.config
            atom.config.config = config

            scientist = create_scientist_from_config()

            assert scientist.variable_names == ["a", "b", "c"]
            assert scientist.memory_limit == 5000

        finally:
            atom.config.config = original_config


class TestScientistThreading:
    """Test scientist threading behavior."""

    def test_monitor_thread_lifecycle(self, scientist):
        """Test monitor thread starts and stops properly."""
        assert scientist.running is True

        # Give thread time to start
        time.sleep(0.1)

        scientist.shutdown()

        assert scientist.running is False

    def test_concurrent_observe_and_ponder(self, scientist):
        """Test concurrent observation and pondering."""
        import threading

        def observe_worker():
            for i in range(100):
                scientist.observe([float(i), 2.0, 3.0], float(i))

        def ponder_worker():
            for _ in range(10):
                scientist.ponder()
                time.sleep(0.01)

        # Start threads
        observe_thread = threading.Thread(target=observe_worker)
        ponder_thread = threading.Thread(target=ponder_worker)

        observe_thread.start()
        ponder_thread.start()

        observe_thread.join()
        ponder_thread.join()

        # Should have observed data
        assert len(scientist.short_term_X) > 0


class TestScientistIntegration:
    """Test scientist integration with other components."""

    def test_end_to_end_workflow(self, scientist):
        """Test complete scientist workflow."""
        # 1. Observe data
        for i in range(100):
            x = float(i)
            y = 2.0
            z = 3.0
            target = x + y + z  # Simple linear relationship
            scientist.observe([x, y, z], target)

        # 2. Allow some processing time
        time.sleep(0.1)

        # 3. Check if any discoveries were made
        discovery = scientist.ponder()

        # Note: In a real scenario with PySR, this might discover laws
        # For this test, we just check that the workflow completes

        # 4. Test prediction (will be 0.0 without compiled law)
        prediction, trust = scientist.predict_theory([1.0, 2.0, 3.0])
        assert isinstance(prediction, float)
        assert isinstance(trust, float)

        # 5. Test stats
        stats = scientist.get_stats()
        assert isinstance(stats, dict)

        # 6. Test shutdown
        scientist.shutdown()
        assert scientist.running is False

    def test_heavy_think_discovers_physics_law_on_physical_target(self):
        """Wake worker should recover physics-feature laws, not latent-only equations."""
        rng = np.random.default_rng(123)
        n = 256
        action = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
        mean_speed = rng.normal(0.08, 0.01, size=n).astype(np.float32)
        turbulence = rng.normal(0.03, 0.005, size=n).astype(np.float32)
        latents = rng.normal(0.0, 0.4, size=(n, 8)).astype(np.float32)

        X = np.column_stack([action, mean_speed, turbulence, latents]).astype(np.float32)
        y = (
            3.2 * mean_speed
            - 1.7 * turbulence
            + 0.4 * action
            + 0.01 * rng.normal(size=n).astype(np.float32)
        ).astype(np.float32)

        var_names = ["action", "mean_speed", "turbulence"] + [f"latent_{i}" for i in range(8)]
        law, score, _ = _heavy_think_global(X, y, var_names)

        assert law is not None
        assert score is not None
        assert any(token in law for token in ("action", "mean_speed", "turbulence"))

    def test_candidate_selection_prefers_physics_when_fit_is_close(self, monkeypatch):
        """Latent-only laws should not beat physics-bearing laws on tiny score deltas."""
        monkeypatch.setenv("ATOM_SCIENTIST_ENFORCE_PHYSICS_PRIORITY", "1")
        monkeypatch.setenv("ATOM_SCIENTIST_PHYSICS_PRIORITY_MARGIN", "0.10")
        monkeypatch.setenv("ATOM_SCIENTIST_PHYSICS_PRIORITY_NRMSE_MARGIN", "0.10")

        rng = np.random.default_rng(7)
        n = 256
        action = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
        mean_speed = rng.normal(0.08, 0.01, size=n).astype(np.float32)
        turbulence = rng.normal(0.03, 0.005, size=n).astype(np.float32)
        latent_0 = mean_speed + 0.002 * rng.normal(size=n).astype(np.float32)
        other_latents = rng.normal(0.0, 0.2, size=(n, 7)).astype(np.float32)

        X = np.column_stack(
            [action, mean_speed, turbulence, latent_0, other_latents]
        ).astype(np.float32)
        y = (2.0 * mean_speed + 0.005 * rng.normal(size=n).astype(np.float32)).astype(np.float32)
        var_names = ["action", "mean_speed", "turbulence"] + [f"latent_{i}" for i in range(8)]

        best = _select_best_candidate(
            X,
            y,
            var_names,
            [
                ("sparse", "(2.0)*(mean_speed)"),
                ("sparse", "(2.0)*(latent_0)"),
            ],
        )

        assert best is not None
        assert "mean_speed" in best.law
