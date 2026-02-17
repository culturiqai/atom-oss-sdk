"""Integration tests for training loop orchestrator."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock

from atom.sim.training_loop import AtomOrchestrator, TrainingState


class _DummyEyes(nn.Module):
    def __init__(self, vision_dim: int):
        super().__init__()
        self.proj = nn.Linear(4, vision_dim)

    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        pooled = obs.mean(dim=(2, 3, 4))
        return self.proj(pooled)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[:, :3]


class _DummyBrain(nn.Module):
    """Minimal differentiable brain used to test orchestrator logic."""

    def __init__(
        self,
        vision_dim: int = 64,
        internal_neurons: int = 16,
        action_dim: int = 1,
        theory_dim: int = 10,
    ):
        super().__init__()
        self.vision_mode = "linear"
        self.internal_neurons = internal_neurons
        self.action_dim = action_dim
        self.theory_dim = theory_dim

        self.eyes = _DummyEyes(vision_dim)
        self.liquid = nn.Linear(vision_dim + theory_dim + action_dim, internal_neurons)
        self.actor_mu = nn.Linear(internal_neurons, action_dim)
        self.actor_std_param = nn.ParameterDict({
            "log_std": nn.Parameter(torch.zeros(1, action_dim) - 0.5)
        })
        self.critic = nn.Linear(internal_neurons, 1)

        self.growth_calls = 0

    @property
    def actor_log_std(self) -> nn.Parameter:
        return self.actor_std_param["log_std"]

    def forward(
        self,
        obs_frame: torch.Tensor,
        theory_signal: torch.Tensor,
        last_action: torch.Tensor,
        hx=None,
        theory_confidence=None,
    ):
        if obs_frame.dim() == 4:
            obs_frame = obs_frame.unsqueeze(0)

        batch_size = int(obs_frame.shape[0])
        if theory_signal.dim() == 1:
            theory_signal = theory_signal.view(1, -1)
        if theory_signal.shape[0] == 1 and batch_size > 1:
            theory_signal = theory_signal.expand(batch_size, -1)

        if last_action.dim() == 1:
            last_action = last_action.view(batch_size, -1)
        if last_action.shape[0] == 1 and batch_size > 1:
            last_action = last_action.expand(batch_size, -1)

        visual = self.eyes.embed(obs_frame)
        theory = theory_signal[:, : self.theory_dim]

        hidden = torch.tanh(self.liquid(torch.cat([visual, theory, last_action], dim=1)))
        mu = self.actor_mu(hidden)
        std = torch.exp(self.actor_log_std).expand_as(mu)
        value = self.critic(hidden)

        new_hx = hidden.detach()
        stress = torch.zeros(batch_size, 1, device=obs_frame.device)
        return (mu, std), value, new_hx, stress

    def apply_hebbian_growth(self):
        self.growth_calls += 1


def _make_batch(batch_size: int = 4, seq_len: int = 4, hx_dim: int = 16):
    return {
        "obs": np.random.randn(batch_size, seq_len, 4, 16, 8, 8).astype(np.float32),
        "action": np.random.randn(batch_size, seq_len, 1).astype(np.float32),
        "reward": np.random.randn(batch_size, seq_len, 1).astype(np.float32),
        "done": np.zeros((batch_size, seq_len, 1), dtype=np.float32),
        "hx": np.random.randn(batch_size, hx_dim).astype(np.float32),
        "old_log_prob": np.random.randn(batch_size, seq_len, 1).astype(np.float32),
        "old_value": np.random.randn(batch_size, seq_len, 1).astype(np.float32),
        "theory": np.random.randn(batch_size, seq_len, 10).astype(np.float32),
        "trust": np.clip(np.random.rand(batch_size, seq_len, 1), 0.0, 1.0).astype(np.float32),
        "next_obs": np.random.randn(batch_size, 4, 16, 8, 8).astype(np.float32),
        "next_hx": np.random.randn(batch_size, hx_dim).astype(np.float32),
    }


class TestAtomOrchestrator:
    """Integration tests for AtomOrchestrator."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        from atom.config import AtomConfig

        config = AtomConfig(
            hardware={"device": "cpu"},
            physics={"grid_shape": [16, 8, 8]},
            brain={
                "vision_dim": 64,
                "internal_neurons": 16,
                "batch_size": 4,
                "ppo_epochs": 1,
                "learning_rate_actor": 1e-4,
                "learning_rate_critic": 1e-3,
                "learning_rate_eyes": 1e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_epsilon": 0.2,
                "skeleton_bones": 4,
            },
            memory={
                "capacity": 100,
                "sequence_length": 4,
            },
            training={
                "max_steps": 5,
                "sleep_interval": 10,
                "render_interval": 10,
            },
        )
        return config

    @pytest.fixture
    def orchestrator(self, mock_config):
        """Create a test orchestrator with lightweight dependencies."""
        with patch("atom.sim.training_loop.FluidWorld"), \
             patch("atom.sim.training_loop.create_brain_from_config") as mock_brain_factory, \
             patch("atom.sim.training_loop.create_memory_from_config"), \
             patch("atom.sim.training_loop.create_scientist_from_config"):

            mock_brain_factory.return_value = _DummyBrain(
                vision_dim=mock_config.brain.vision_dim,
                internal_neurons=mock_config.brain.internal_neurons,
                action_dim=mock_config.brain.action_dim,
                theory_dim=10,
            )

            orchestrator = AtomOrchestrator(mock_config)

            # Mock world
            mock_obs = np.random.randn(1, 4, 16, 8, 8).astype(np.float32)
            orchestrator.world = MagicMock()
            orchestrator.world.reset.return_value = (mock_obs, np.zeros((1, 16, 8, 8), dtype=np.float32))
            orchestrator.world.step.return_value = (mock_obs, 1.0, False, {})
            orchestrator.world.export_to_web = MagicMock()
            orchestrator.world.render = MagicMock()
            orchestrator.world.nx = 16
            orchestrator.world.ny = 8
            orchestrator.world.nz = 8

            # Mock memory
            orchestrator.memory = MagicMock()
            orchestrator.memory.size = 20
            orchestrator.memory.seq_len = 4
            orchestrator.memory.obs_buf = np.zeros((120, 4, 16, 8, 8), dtype=np.float32)
            orchestrator.memory.sample.return_value = _make_batch(hx_dim=mock_config.brain.internal_neurons)
            orchestrator.memory.push = MagicMock()

            # Mock scientist v2 surface
            orchestrator.scientist = MagicMock()
            def _signal(**kwargs):
                dev = kwargs.get("device", orchestrator.config.get_device())
                return {
                    "tensor": torch.zeros(1, 10, device=dev),
                    "trust": 0.5,
                    "prediction": 0.25,
                }
            orchestrator.scientist.get_signal.side_effect = _signal
            orchestrator.scientist.observe_and_verify = MagicMock()
            orchestrator.scientist.ponder.return_value = None
            orchestrator.scientist.shutdown = MagicMock()
            orchestrator.scientist.fit_offline = MagicMock()

            return orchestrator

    def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.world is not None
        assert orchestrator.brain is not None
        assert orchestrator.memory is not None
        assert orchestrator.scientist is not None
        assert orchestrator.optimizer is not None
        assert "reward" in orchestrator.history
        assert "stress" in orchestrator.history
        assert "theory_score" in orchestrator.history
        assert "safety_intervention" in orchestrator.history
        assert "safety_fallback" in orchestrator.history

    def test_scientist_feature_schema_includes_density_observables(self, orchestrator):
        """Scientist feature schema should include explicit density channels."""
        assert orchestrator.scientist_vars[:5] == [
            "action",
            "mean_speed",
            "turbulence",
            "rho_mean",
            "rho_std",
        ]
        assert len(orchestrator.scientist_vars) == 13

    def test_tensorify_conversion(self, orchestrator):
        """Test tensor conversion from JAX arrays."""
        jax_array = np.random.randn(2, 3, 4)
        tensor = orchestrator._tensorify(jax_array)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, 3, 4)
        assert tensor.dtype == torch.float32

    def test_safe_gradient_clipping(self, orchestrator):
        """Test safe gradient clipping."""
        param1 = torch.nn.Parameter(torch.randn(10, 10))
        param2 = torch.nn.Parameter(torch.randn(5, 5))
        param1.grad = torch.randn_like(param1)
        param2.grad = torch.randn_like(param2)

        norm = orchestrator._safe_clip_grads([param1, param2], max_norm=1.0)
        assert isinstance(norm, float)
        assert norm >= 0

    def test_brain_update_with_insufficient_data(self, orchestrator):
        """Test brain update when memory has insufficient data."""
        orchestrator.memory.sample.return_value = None
        metrics = orchestrator._update_brain()
        assert metrics == {"loss": 0.0}

    def test_brain_update_full_cycle(self, orchestrator):
        """Test full brain update cycle."""
        metrics = orchestrator._update_brain()
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)
        assert orchestrator.brain.growth_calls > 0

    def test_dream_mode_insufficient_memories(self, orchestrator):
        """Test dream mode with insufficient memories."""
        orchestrator.memory.size = 50
        orchestrator.run_dream_mode()
        assert not orchestrator.scientist.fit_offline.called

    def test_dream_mode_full_cycle(self, orchestrator):
        """Test dream mode with sufficient memories."""
        orchestrator.memory.size = 200
        orchestrator.memory.sample.return_value = _make_batch(hx_dim=orchestrator.config.brain.internal_neurons)
        orchestrator.run_dream_mode()
        assert orchestrator.scientist.fit_offline.called

    def test_training_loop_short_run(self, orchestrator):
        """Test a short training loop run."""
        orchestrator.memory.size = 0
        orchestrator.run()

        assert orchestrator.world.reset.called
        assert orchestrator.world.step.called
        assert orchestrator.memory.push.called
        assert orchestrator.scientist.observe_and_verify.called

    def test_training_loop_passes_extended_scientist_feature_vector(self, orchestrator):
        orchestrator.memory.size = 0
        orchestrator.config.training.max_steps = 1
        orchestrator.run()

        assert orchestrator.scientist.observe_and_verify.called
        f_vec = orchestrator.scientist.observe_and_verify.call_args.args[0]
        assert isinstance(f_vec, torch.Tensor)
        assert int(f_vec.numel()) == len(orchestrator.scientist_vars)

    def test_runtime_diagnostics_include_obstacle_overlay(self, orchestrator):
        obs_t = torch.randn(1, 4, 16, 8, 8, dtype=torch.float32)
        theory_t = torch.zeros(1, 10, dtype=torch.float32)
        last_action = torch.zeros(1, 1, dtype=torch.float32)
        hx = torch.zeros(1, orchestrator.config.brain.internal_neurons, dtype=torch.float32)
        theory_conf = torch.tensor([[0.5]], dtype=torch.float32)
        info = {"mask": np.ones((1, 16, 8, 8), dtype=np.float32)}

        diagnostics = orchestrator._collect_runtime_diagnostics(
            step=1,
            obs_t=obs_t,
            theory_t=theory_t,
            last_action=last_action,
            hx=hx,
            theory_confidence=theory_conf,
            info=info,
        )
        live_view = diagnostics.get("live_view", {})
        assert "obstacle_xy" in live_view
        assert live_view["obstacle_xy"]["shape"][0] > 0
        assert live_view["obstacle_xy"]["shape"][1] > 0

    def test_training_state_tracking(self):
        """Test training state tracking."""
        state = TrainingState()

        assert state.step == 0
        assert state.epoch == 0
        assert state.best_metric == float("-inf")
        assert len(state.metrics_history) == 0
        assert len(state.lr_schedulers_state) == 0
        assert len(state.optimizers_state) == 0


class TestOrchestratorConfiguration:
    """Test orchestrator configuration integration."""

    def test_create_from_config(self):
        """Test creating orchestrator from configuration."""
        from atom.config import AtomConfig

        config = AtomConfig(
            physics={"grid_shape": [16, 8, 8]},
            brain={"vision_dim": 64, "internal_neurons": 16},
            memory={"capacity": 100, "sequence_length": 4},
        )

        from atom.config import config as global_config
        original_config = global_config
        try:
            import atom.config
            atom.config.config = config

            with patch("atom.sim.training_loop.FluidWorld"), \
                 patch("atom.sim.training_loop.create_brain_from_config") as mock_brain_factory, \
                 patch("atom.sim.training_loop.create_memory_from_config"), \
                 patch("atom.sim.training_loop.create_scientist_from_config"):

                mock_brain_factory.return_value = _DummyBrain(
                    vision_dim=config.brain.vision_dim,
                    internal_neurons=config.brain.internal_neurons,
                    action_dim=config.brain.action_dim,
                    theory_dim=10,
                )
                orchestrator = AtomOrchestrator()
                assert orchestrator.config == config

        finally:
            atom.config.config = original_config


class TestOrchestratorErrorHandling:
    """Test error handling in orchestrator."""

    @pytest.fixture
    def orchestrator(self):
        from atom.config import AtomConfig

        config = AtomConfig(
            hardware={"device": "cpu"},
            physics={"grid_shape": [16, 8, 8]},
            brain={
                "vision_dim": 64,
                "internal_neurons": 16,
                "batch_size": 4,
                "ppo_epochs": 1,
                "learning_rate_actor": 1e-4,
                "learning_rate_critic": 1e-3,
                "learning_rate_eyes": 1e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_epsilon": 0.2,
                "skeleton_bones": 4,
            },
            memory={"capacity": 100, "sequence_length": 4},
            training={"max_steps": 2, "sleep_interval": 10, "render_interval": 10},
        )

        with patch("atom.sim.training_loop.FluidWorld"), \
             patch("atom.sim.training_loop.create_brain_from_config") as mock_brain_factory, \
             patch("atom.sim.training_loop.create_memory_from_config"), \
             patch("atom.sim.training_loop.create_scientist_from_config"):
            mock_brain_factory.return_value = _DummyBrain(
                vision_dim=config.brain.vision_dim,
                internal_neurons=config.brain.internal_neurons,
                action_dim=config.brain.action_dim,
                theory_dim=10,
            )
            orch = AtomOrchestrator(config)
            orch.memory = MagicMock()
            orch.memory.size = 20
            orch.memory.seq_len = 4
            orch.memory.sample.return_value = _make_batch(hx_dim=config.brain.internal_neurons)
            return orch

    def test_tensorify_error_handling(self):
        """Test tensorify error handling."""
        from atom.config import AtomConfig

        config = AtomConfig(hardware={"device": "cpu"})
        with patch("atom.sim.training_loop.FluidWorld"), \
             patch("atom.sim.training_loop.create_brain_from_config") as mock_brain_factory, \
             patch("atom.sim.training_loop.create_memory_from_config"), \
             patch("atom.sim.training_loop.create_scientist_from_config"):
            mock_brain_factory.return_value = _DummyBrain()
            orchestrator = AtomOrchestrator(config)

            with pytest.raises(Exception):
                orchestrator._tensorify("invalid_input")

    def test_brain_update_error_handling(self, orchestrator):
        """Test brain update error handling."""
        with patch.object(orchestrator.brain, "forward", side_effect=RuntimeError("Brain error")):
            with pytest.raises(Exception):
                orchestrator._update_brain()
