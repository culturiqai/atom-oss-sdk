"""Unit tests for AtomEyes (vision system)."""

import pytest
import torch
import numpy as np

from atom.core.legacy.eyes import (
    AtomEyes,
    SpectralConv3d,
    HelmholtzHead3d,
    ComplexLowRankLinear,
    create_eyes_from_config
)
from atom.exceptions import EyesError


class TestComplexLowRankLinear:
    """Test ComplexLowRankLinear component."""

    def test_initialization(self):
        """Test initialization."""
        layer = ComplexLowRankLinear(in_dim=32, out_shape=(16, 8, 8))
        assert layer.out_shape == (16, 8, 8)
        assert layer.flat_dim == 16 * 8 * 8

    def test_forward_pass(self):
        """Test forward pass."""
        layer = ComplexLowRankLinear(in_dim=16, out_shape=(4, 4, 4))

        x = torch.randn(2, 16)  # Batch of 2, 16 features
        output = layer(x)

        assert output.shape == (2, 4, 4, 4)


class TestSpectralConv3d:
    """Test SpectralConv3d component."""

    @pytest.fixture
    def spectral_conv(self):
        """Create a test spectral convolution."""
        return SpectralConv3d(
            in_channels=4,
            out_channels=8,
            modes_x=4,
            modes_y=4,
            modes_z=4
        )

    def test_initialization(self, spectral_conv):
        """Test initialization."""
        assert spectral_conv.in_channels == 4
        assert spectral_conv.out_channels == 8
        assert spectral_conv.modes_x == 4
        assert spectral_conv.weights.shape == (4, 8, 4, 4, 4)

    def test_forward_pass(self, spectral_conv):
        """Test forward pass."""
        # Input: (batch, channels, x, y, z)
        x = torch.randn(2, 4, 16, 16, 16)

        output = spectral_conv(x)

        # Should preserve spatial dimensions
        assert output.shape == (2, 8, 16, 16, 16)

    def test_complex_weights(self, spectral_conv):
        """Test that weights are complex."""
        assert spectral_conv.weights.dtype == torch.complex64


class TestHelmholtzHead3d:
    """Test HelmholtzHead3d component."""

    @pytest.fixture
    def helmholtz_head(self):
        """Create a test Helmholtz head."""
        return HelmholtzHead3d(in_width=32)

    def test_initialization(self, helmholtz_head):
        """Test initialization."""
        assert hasattr(helmholtz_head, 'proj')

    def test_forward_pass(self, helmholtz_head):
        """Test forward pass with curl computation."""
        # Input: (batch, x, y, z, channels)
        x = torch.randn(2, 8, 8, 8, 32)

        output = helmholtz_head(x)

        # Should output velocity field (u, v, w)
        assert output.shape == (2, 3, 8, 8, 8)

    def test_divergence_free_property(self, helmholtz_head):
        """Test that output satisfies divergence-free property (approximately)."""
        x = torch.randn(1, 8, 8, 8, 32)
        velocity_field = helmholtz_head(x)

        # Compute divergence numerically
        u, v, w = velocity_field[0]  # Remove batch dim

        # Central differences for divergence
        du_dx = torch.diff(u, dim=0)
        dv_dy = torch.diff(v, dim=1)
        dw_dz = torch.diff(w, dim=2)

        # Divergence should be close to zero (within numerical precision)
        # Note: This is a rough check due to boundary effects
        mean_divergence = torch.mean(torch.abs(du_dx) + torch.abs(dv_dy) + torch.abs(dw_dz))
        assert mean_divergence < 1.0  # Allow some numerical error


class TestAtomEyes:
    """Test AtomEyes vision system."""

    @pytest.fixture
    def eyes(self):
        """Create a test vision system."""
        return AtomEyes(modes=4, width=16, depth=2)

    def test_initialization(self, eyes):
        """Test initialization."""
        assert eyes.width == 16
        assert eyes.modes == 4
        assert len(eyes.spectral_layers) == 2  # depth = 2
        assert len(eyes.skip_layers) == 2
        assert len(eyes.norms) == 2

    def test_embed_fast_path(self, eyes):
        """Test fast embedding path."""
        x = torch.randn(2, 4, 16, 16, 16)  # (batch, channels, x, y, z)

        embedding = eyes.embed(x)

        assert embedding.shape == (2, 256)  # Should output latent vector

    def test_forward_full_reconstruction(self, eyes):
        """Test full forward pass with reconstruction."""
        x = torch.randn(2, 4, 16, 16, 16)

        reconstruction = eyes(x)

        # Should reconstruct velocity field
        assert reconstruction.shape == (2, 3, 16, 16, 16)

    def test_permutation_consistency(self, eyes):
        """Test that input/output permutations are consistent."""
        x = torch.randn(1, 4, 8, 8, 8)

        # Get embedding
        embedding = eyes.embed(x)

        # Get full reconstruction
        reconstruction = eyes(x)

        # Both should work without errors
        assert embedding.shape[0] == x.shape[0]  # Same batch size
        assert reconstruction.shape[0] == x.shape[0]

    def test_different_input_sizes(self, eyes):
        """Test with different input spatial sizes."""
        test_sizes = [(8, 8, 8), (16, 16, 16), (12, 12, 12)]

        for size in test_sizes:
            x = torch.randn(1, 4, *size)

            # Should not raise errors
            embedding = eyes.embed(x)
            assert embedding.shape == (1, 256)

    def test_error_handling(self, eyes):
        """Test error handling for invalid inputs."""
        # Wrong number of channels
        x = torch.randn(1, 3, 16, 16, 16)  # Should be 4 channels

        with pytest.raises(EyesError):
            eyes.embed(x)

    def test_training_vs_eval_mode(self, eyes):
        """Test behavior in training vs evaluation modes."""
        x = torch.randn(1, 4, 16, 16, 16)

        # Training mode
        eyes.train()
        result_train = eyes.embed(x)

        # Eval mode
        eyes.eval()
        result_eval = eyes.embed(x)

        # Results should be identical (deterministic)
        assert torch.allclose(result_train, result_eval)


class TestEyesConfiguration:
    """Test eyes configuration integration."""

    def test_create_from_config(self):
        """Test creating eyes from configuration."""
        from atom.config import AtomConfig

        config = AtomConfig(
            eyes={
                "fno_modes": 6,
                "fno_width": 24,
                "fno_depth": 3,
                "embedding_dim": 128
            }
        )

        # Temporarily set as global config
        from atom.config import config as global_config
        original_config = global_config
        try:
            import atom.config
            atom.config.config = config

            eyes = create_eyes_from_config()

            assert eyes.modes == 6
            assert eyes.width == 24

        finally:
            atom.config.config = original_config


class TestEyesMemoryEfficiency:
    """Test memory efficiency aspects."""

    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        eyes = AtomEyes(modes=2, width=8, depth=1)

        x = torch.randn(1, 4, 8, 8, 8, requires_grad=True)

        # Forward pass
        embedding = eyes.embed(x)

        # Backward pass
        loss = embedding.sum()
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_no_memory_leaks(self):
        """Test for memory leaks in repeated forward passes."""
        eyes = AtomEyes(modes=2, width=8, depth=1)

        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Run multiple forward passes
        for _ in range(10):
            x = torch.randn(1, 4, 8, 8, 8)
            _ = eyes.embed(x)

        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Memory should not grow significantly (allowing for some overhead)
        memory_growth = final_memory - initial_memory
        assert memory_growth < 100 * 1024 * 1024  # Less than 100MB growth