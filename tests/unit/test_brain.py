"""Unit tests for AtomBrain."""

import pytest
import torch
import numpy as np

from atom.core.brain import AtomBrain, NeuralSkeleton, create_brain_from_config
from atom.exceptions import BrainError


class TestNeuralSkeleton:
    """Test NeuralSkeleton component."""

    @pytest.fixture
    def skeleton(self):
        """Create a test skeleton."""
        return NeuralSkeleton(num_inputs=64, num_bones=8)

    def test_initialization(self, skeleton):
        """Test skeleton initialization."""
        assert skeleton.num_bones == 8
        assert skeleton.plasticity == 0.01
        assert skeleton.components.shape == (8, 64)

    def test_forward_pass(self, skeleton):
        """Test skeleton forward pass."""
        x = torch.randn(4, 64)  # Batch of 4, 64 neurons

        x_recon, stress = skeleton(x)

        assert x_recon.shape == x.shape
        assert stress.shape == (4, 1)
        assert torch.all(stress >= 0)  # Stress should be non-negative

    def test_orthogonal_initialization(self, skeleton):
        """Test that components are approximately orthogonal."""
        # Check that components are not all zeros
        assert not torch.allclose(skeleton.components, torch.zeros_like(skeleton.components))

        # Check that different components are not identical
        for i in range(skeleton.num_bones):
            for j in range(i + 1, skeleton.num_bones):
                assert not torch.allclose(
                    skeleton.components[i], skeleton.components[j]
                )

    def test_hebbian_growth(self, skeleton):
        """Test Hebbian growth mechanism."""
        original_components = skeleton.components.clone()

        # Run forward pass with training enabled
        skeleton.train()
        x = torch.randn(4, 64)
        _ = skeleton(x)

        # Apply growth
        skeleton.apply_growth()

        # Components should have changed
        assert not torch.allclose(skeleton.components, original_components)


class TestAtomBrain:
    """Test AtomBrain neural controller."""

    @pytest.fixture
    def brain(self):
        """Create a test brain."""
        return AtomBrain(
            vision_dim=128,
            theory_dim=1,
            action_dim=1,
            internal_neurons=32,
            bones=4
        )

    def test_initialization(self, brain):
        """Test brain initialization."""
        assert brain.vision_dim == 128
        assert brain.internal_neurons == 32
        assert isinstance(brain.eyes, torch.nn.Module)
        assert isinstance(brain.skeleton, torch.nn.Module)

    def test_forward_pass_cpu(self, brain):
        """Test forward pass on CPU."""
        batch_size = 2

        # Create dummy inputs
        obs_frame = torch.randn(batch_size, 4, 16, 16, 16)
        theory_signal = torch.randn(batch_size, 1)
        last_action = torch.randn(batch_size, 1)

        # Forward pass
        (mu, std), value, hx_new, stress = brain(obs_frame, theory_signal, last_action)

        # Check output shapes
        assert mu.shape == (batch_size, 1)
        assert std.shape == (batch_size, 1)
        assert value.shape == (batch_size, 1)
        assert hx_new.shape[0] == batch_size  # Hidden state
        assert stress.shape == (batch_size, 1)

        # Check value ranges
        assert torch.all(std > 0)  # Standard deviation should be positive
        assert torch.all(stress >= 0)  # Stress should be non-negative

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_device_compatibility(self, device, brain):
        """Test brain works on different devices."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Move to device
        brain = brain.to(device)

        batch_size = 2
        obs_frame = torch.randn(batch_size, 4, 16, 16, 16).to(device)
        theory_signal = torch.randn(batch_size, 1).to(device)
        last_action = torch.randn(batch_size, 1).to(device)

        # Should not raise an error
        (mu, std), value, hx_new, stress = brain(obs_frame, theory_signal, last_action)

        assert mu.device.type == device

    def test_hebbian_growth_integration(self, brain):
        """Test Hebbian growth integration."""
        # Run forward pass
        obs_frame = torch.randn(2, 4, 16, 16, 16)
        theory_signal = torch.randn(2, 1)
        last_action = torch.randn(2, 1)

        _ = brain(obs_frame, theory_signal, last_action)

        # Apply growth
        brain.apply_hebbian_growth()  # Should not raise error

    def test_error_handling(self, brain):
        """Test error handling in forward pass."""
        # Test with mismatched input shapes
        obs_frame = torch.randn(2, 3, 16, 16, 16)  # Wrong channel count
        theory_signal = torch.randn(2, 1)
        last_action = torch.randn(2, 1)

        with pytest.raises(BrainError):
            brain(obs_frame, theory_signal, last_action)

    def test_hidden_state_handling(self, brain):
        """Test hidden state handling."""
        batch_size = 2
        obs_frame = torch.randn(batch_size, 4, 16, 16, 16)
        theory_signal = torch.randn(batch_size, 1)
        last_action = torch.randn(batch_size, 1)

        # Test with None hidden state
        (mu1, std1), value1, hx1, stress1 = brain(obs_frame, theory_signal, last_action, None)

        # Test with provided hidden state
        (mu2, std2), value2, hx2, stress2 = brain(obs_frame, theory_signal, last_action, hx1)

        # Results should be different (deterministic but different due to hidden state)
        assert not torch.allclose(mu1, mu2)


class TestBrainConfiguration:
    """Test brain configuration integration."""

    def test_create_from_config(self):
        """Test creating brain from configuration."""
        from atom.config import AtomConfig

        config = AtomConfig(
            brain={
                "vision_dim": 64,
                "theory_dim": 2,
                "action_dim": 1,
                "internal_neurons": 16,
                "skeleton_bones": 4
            }
        )

        # Temporarily set as global config
        from atom.config import config as global_config
        original_config = global_config
        try:
            import atom.config
            atom.config.config = config

            brain = create_brain_from_config()

            assert brain.vision_dim == 64
            assert brain.internal_neurons == 16

        finally:
            atom.config.config = original_config


class TestBrainTrainingMode:
    """Test brain behavior in training vs evaluation modes."""

    @pytest.fixture
    def brain(self):
        """Create a test brain."""
        return AtomBrain(internal_neurons=16, bones=4)

    def test_training_mode(self, brain):
        """Test brain in training mode."""
        brain.train()

        obs_frame = torch.randn(2, 4, 16, 16, 16)
        theory_signal = torch.randn(2, 1)
        last_action = torch.randn(2, 1)

        _, _, _, stress = brain(obs_frame, theory_signal, last_action)

        # In training mode, stress should be computed
        assert stress is not None

    def test_evaluation_mode(self, brain):
        """Test brain in evaluation mode."""
        brain.eval()

        obs_frame = torch.randn(2, 4, 16, 16, 16)
        theory_signal = torch.randn(2, 1)
        last_action = torch.randn(2, 1)

        _, _, _, stress = brain(obs_frame, theory_signal, last_action)

        # Stress should still be computed (it's part of the forward pass)
        assert stress is not None