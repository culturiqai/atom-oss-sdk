import unittest
import numpy as np
import jax.numpy as jnp
from atom.mind.memory import AtomMemory
from atom.sim.world import FluidWorld, CylinderWorld

class TestPhase3System(unittest.TestCase):
    def test_memory_wrap_around_protection(self):
        """Verify memory rejects sequences that wrap around the pointer."""
        capacity = 10
        seq_len = 4
        mem = AtomMemory(capacity=capacity, seq_len=seq_len)
        
        # Fill buffer completely
        # ptr will loop back to 0
        for i in range(capacity):
            mem.push(np.zeros((1,)), 0, 0, False, np.zeros((1,)))
            
        self.assertEqual(mem.ptr, 0)
        self.assertEqual(mem.size, capacity)
        
        # Case 1: Sequence at end of buffer
        # idx = 6. seq = [6, 7, 8, 9]. Next = 0 (which is ptr!).
        # This means next_obs would read from the overwrite pointer.
        # This should be INVALID per our fix.
        # (idx + seq_len) % capacity == (6 + 4) % 10 == 0 == ptr
        is_valid = mem._is_valid_sequence(6)
        self.assertFalse(is_valid, "Should reject sequence ending exactly at pointer")
        
        # Case 2: Safe sequence
        # idx = 5. seq = [5, 6, 7, 8]. Next = 9. != ptr.
        is_valid = mem._is_valid_sequence(5)
        self.assertTrue(is_valid, "Should accept safe sequence")

    def test_fluid_world_batching(self):
        """Verify FluidWorld handles batched actions (B=2)."""
        batch_size = 2
        # Small grid for speed
        env = FluidWorld(nx=32, ny=16, nz=16, batch_size=batch_size)
        env.reset()
        
        # Action: Batch of 2 scalars
        # Use larger values to ensure grid quantization sees a shift (nz=16, nz/4=4, 0.1*4=0)
        actions = np.array([1.0, -1.0])
        
        obs, rew, done, info = env.step(actions, sub_steps=1)
        
        # Check Shapes
        self.assertEqual(obs.shape, (batch_size, 4, 32, 16, 16))
        self.assertEqual(rew.shape, (batch_size, 1))
        
        # Check Masks Differ (since actions differ)
        masks = info['mask'] # (B, X, Y, Z)
        self.assertEqual(masks.shape, (batch_size, 32, 16, 16))
        
        # Action 0.1 should shift mask differently than -0.1
        # We can check simple equality
        self.assertFalse(np.allclose(masks[0], masks[1]), "Masks should differ for different actions")

    def test_cylinder_world_batching(self):
        """Verify CylinderWorld handles batched actions (B=2)."""
        batch_size = 2
        env = CylinderWorld(nx=32, ny=16, nz=16, batch_size=batch_size)
        env.reset()
        
        # Action: Batch of 2 scalars
        actions = np.array([0.5, -0.5])
        
        obs, rew, done, info = env.step(actions, sub_steps=1)
        
        self.assertEqual(obs.shape, (batch_size, 4, 32, 16, 16))
        self.assertEqual(rew.shape, (batch_size, 1))
        
        # Check Rewards (might be same if physics hasn't evolved enough, but shape matters)
        # With different jets, flow should diverge eventually.
        # Here we just check it runs.

if __name__ == "__main__":
    unittest.main()
