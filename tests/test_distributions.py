
import unittest
import torch
import math
from atom.core.distributions import TanhNormal

class TestTanhNormal(unittest.TestCase):
    def setUp(self):
        self.B = 10
        self.act_dim = 2
        self.loc = torch.zeros(self.B, self.act_dim)
        self.scale = torch.ones(self.B, self.act_dim)
        self.dist = TanhNormal(self.loc, self.scale)

    def test_shapes_correctness(self):
        """Verify batch_shape and event_shape are consistent with Torch standards."""
        # Current implementation sets batch_shape to loc.shape (B, D)
        # Correct implementation should be batch_shape=(B,), event_shape=(D,)
        
        # This assertion expects the FIX. It will FAIL on current code.
        expected_batch = (self.B,)
        expected_event = (self.act_dim,)
        
        print(f"Dist batch_shape: {self.dist.batch_shape}")
        print(f"Dist event_shape: {self.dist.event_shape}")
        
        self.assertEqual(self.dist.batch_shape, expected_batch, "Batch shape mismatch")
        self.assertEqual(self.dist.event_shape, expected_event, "Event shape mismatch")

    def test_sample_shape_argument(self):
        """Verify sample() accepts a sample_shape argument."""
        # Current implementation does not accept arguments.
        try:
            sample, raw = self.dist.sample((5,))
            self.assertEqual(sample.shape, (5, self.B, self.act_dim))
        except TypeError:
            self.fail("sample() does not accept sample_shape argument")

    def test_log_prob_consistency(self):
        """
        Verify log_prob is consistent between:
        1. Using pre_tanh_value (stable)
        2. Using value only (inverted)
        """
        # Create a value that is safe from clamping effects
        z = torch.tensor([0.5, -0.5])
        action = torch.tanh(z)
        
        # Case 1: With pre_tanh
        lp_ref = self.dist.log_prob(action, pre_tanh_value=z)
        
        # Case 2: Without pre_tanh
        lp_inv = self.dist.log_prob(action)
        
        # They should be very close
        diff = (lp_ref - lp_inv).abs().max()
        self.assertTrue(diff < 1e-6, f"Log prob divergence: {diff}")

    def test_log_prob_clamping_drift(self):
        """
        Verify that clamping saturated values introduces error if pre_tanh isn't used.
        This demonstrates the PPO update mismatch risk.
        """
        # Saturated value
        z = torch.tensor([10.0]) # tanh(10) ~= 1.0
        action = torch.tanh(z)
        
        # Log prob with ground truth z
        lp_ref = self.dist.log_prob(action, pre_tanh_value=z)
        
        # Log prob via inversion (will hit clamp)
        lp_inv = self.dist.log_prob(action)
        
        # Discrepancy expected
        diff = (lp_ref - lp_inv).abs().max().item()
        print(f"Saturation drift: {diff}")
        # We don't assert fail here because it's a property of the approximation, 
        # but the fix is to ENSURE we use pre_tanh in PPO.
        
if __name__ == "__main__":
    unittest.main()
