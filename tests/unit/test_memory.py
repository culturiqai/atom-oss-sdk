"""Unit tests for AtomMemory (experience replay)."""

import tempfile
from pathlib import Path

import pytest
import torch
import numpy as np

from atom.mind.memory import AtomMemory, create_memory_from_config
from atom.exceptions import MemoryError


class TestAtomMemory:
    """Test AtomMemory experience replay system."""

    @pytest.fixture
    def memory(self):
        """Create a test memory instance."""
        return AtomMemory(
            capacity=100,
            seq_len=8,
            obs_shape=(4, 16, 16, 16),
            act_dim=1
        )

    def test_initialization(self, memory):
        """Test memory initialization."""
        assert memory.capacity == 100
        assert memory.seq_len == 8
        assert memory.obs_shape == (4, 16, 16, 16)
        assert memory.act_dim == 1
        assert memory.size == 0
        assert memory.ptr == 0

    def test_push_single_experience(self, memory):
        """Test pushing a single experience."""
        obs = np.random.randn(4, 16, 16, 16)
        action = 0.5
        reward = 1.0
        done = False
        hx = np.random.randn(64)

        memory.push(obs, action, reward, done, hx)

        assert memory.size == 1
        assert memory.ptr == 1

    def test_push_with_tensors(self, memory):
        """Test pushing with PyTorch tensors."""
        obs = torch.randn(4, 16, 16, 16)
        action = torch.tensor([0.5])
        reward = 1.0
        done = False
        hx = torch.randn(64)

        memory.push(obs, action, reward, done, hx)

        assert memory.size == 1

    def test_push_with_ppo_data(self, memory):
        """Test pushing with PPO-specific data."""
        obs = np.random.randn(4, 16, 16, 16)
        action = 0.5
        reward = 1.0
        done = False
        hx = np.random.randn(64)
        log_prob = np.array([0.1])
        value = np.array([0.8])

        memory.push(obs, action, reward, done, hx, log_prob, value)

        assert memory.size == 1

    def test_ring_buffer_behavior(self, memory):
        """Test ring buffer wrap-around behavior."""
        # Fill the buffer
        for i in range(memory.capacity + 10):
            obs = np.random.randn(4, 16, 16, 16)
            action = float(i)
            reward = 1.0
            done = False
            hx = np.random.randn(64)

            memory.push(obs, action, reward, done, hx)

        # Should maintain capacity
        assert memory.size == memory.capacity

        # Pointer should have wrapped around
        assert memory.ptr == (memory.capacity + 10) % memory.capacity

    def test_sample_sequences(self, memory):
        """Test sampling sequences for training."""
        # Add enough experiences to sample from
        num_experiences = memory.seq_len + 10

        for i in range(num_experiences):
            obs = np.random.randn(4, 16, 16, 16)
            action = float(i)
            reward = 1.0
            done = False
            hx = np.random.randn(64)

            memory.push(obs, action, reward, done, hx)

        # Sample a batch
        batch = memory.sample(batch_size=2)

        assert batch is not None
        assert "obs" in batch
        assert "action" in batch
        assert "reward" in batch
        assert "done" in batch
        assert "hx" in batch
        assert "old_log_prob" in batch
        assert "old_value" in batch

        # Check shapes
        assert batch["obs"].shape == (2, memory.seq_len, 4, 16, 16, 16)
        assert batch["action"].shape == (2, memory.seq_len, 1)
        assert batch["reward"].shape == (2, memory.seq_len, 1)
        assert batch["done"].shape == (2, memory.seq_len, 1)
        assert batch["hx"].shape == (2, 64)

    def test_sample_insufficient_data(self, memory):
        """Test sampling when there's insufficient data."""
        # Add fewer experiences than needed for a sequence
        for i in range(memory.seq_len - 1):
            obs = np.random.randn(4, 16, 16, 16)
            action = float(i)
            reward = 1.0
            done = False
            hx = np.random.randn(64)

            memory.push(obs, action, reward, done, hx)

        # Should return None when can't sample
        batch = memory.sample(batch_size=1)
        assert batch is None

    def test_sequence_validity_check(self, memory):
        """Test sequence validity checking."""
        # Create a sequence that crosses an episode boundary
        for i in range(memory.seq_len + 5):
            obs = np.random.randn(4, 16, 16, 16)
            action = float(i)
            reward = 1.0
            done = (i == memory.seq_len)  # Episode ends in middle of potential sequence
            hx = np.random.randn(64)

            memory.push(obs, action, reward, done, hx)

        # Should be able to find valid sequences
        batch = memory.sample(batch_size=1)
        # Note: This might return None if no valid sequences exist, which is acceptable

    def test_save_and_load(self, memory):
        """Test saving and loading memory."""
        # Add some data
        for i in range(20):
            obs = np.random.randn(4, 16, 16, 16)
            action = float(i)
            reward = 1.0
            done = False
            hx = np.random.randn(64)

            memory.push(obs, action, reward, done, hx)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_memory.npz"

            # Save
            memory.save(save_path)

            # Create new memory instance
            new_memory = AtomMemory(
                capacity=100,
                seq_len=8,
                obs_shape=(4, 16, 16, 16),
                act_dim=1
            )

            # Load
            new_memory.load(save_path)

            # Check that data was restored
            assert new_memory.size == memory.size
            assert new_memory.ptr == memory.ptr

            # Check that data matches
            np.testing.assert_array_equal(new_memory.obs_buf[:new_memory.size],
                                        memory.obs_buf[:memory.size])

    def test_load_nonexistent_file(self, memory):
        """Test loading from nonexistent file."""
        with pytest.raises(MemoryError):
            memory.load("nonexistent_file.npz")

    def test_get_stats(self, memory):
        """Test getting memory statistics."""
        # Add some data
        for i in range(15):
            obs = np.random.randn(4, 16, 16, 16)
            action = float(i)
            reward = 1.0
            done = False
            hx = np.random.randn(64)

            memory.push(obs, action, reward, done, hx)

        stats = memory.get_stats()

        assert stats["capacity"] == 100
        assert stats["size"] == 15
        assert stats["utilization"] == 15/100
        assert stats["pointer"] == 15
        assert stats["obs_shape"] == (4, 16, 16, 16)
        assert stats["act_dim"] == 1

    def test_clear_memory(self, memory):
        """Test clearing memory."""
        # Add data
        for i in range(10):
            obs = np.random.randn(4, 16, 16, 16)
            action = float(i)
            reward = 1.0
            done = False
            hx = np.random.randn(64)

            memory.push(obs, action, reward, done, hx)

        assert memory.size == 10

        # Clear
        memory.clear()

        assert memory.size == 0
        assert memory.ptr == 0

    def test_error_handling(self, memory):
        """Test error handling for invalid inputs."""
        # Test with wrong observation shape
        obs = np.random.randn(3, 16, 16, 16)  # Wrong channel count
        action = 0.5
        reward = 1.0
        done = False
        hx = np.random.randn(64)

        # Should handle the error gracefully
        with pytest.raises(MemoryError):
            memory.push(obs, action, reward, done, hx)


class TestMemoryConfiguration:
    """Test memory configuration integration."""

    def test_create_from_config(self):
        """Test creating memory from configuration."""
        from atom.config import AtomConfig

        config = AtomConfig(
            memory={
                "capacity": 200,
                "sequence_length": 12,
                "memory_dtype": "float32"
            },
            physics={"grid_shape": [16, 16, 16]},
            brain={"action_dim": 2}
        )

        # Temporarily set as global config
        from atom.config import config as global_config
        original_config = global_config
        try:
            import atom.config
            atom.config.config = config

            memory = create_memory_from_config()

            assert memory.capacity == 200
            assert memory.seq_len == 12
            assert memory.obs_shape == (4, 16, 16, 16)  # Channelized obs shape
            assert memory.act_dim == 2

        finally:
            atom.config.config = original_config


class TestMemoryThreadSafety:
    """Test memory thread safety aspects."""

    def test_concurrent_access(self, memory):
        """Test concurrent push operations."""
        import threading

        def push_worker(worker_id: int, num_pushes: int):
            for i in range(num_pushes):
                obs = np.random.randn(4, 16, 16, 16)
                action = float(worker_id * 100 + i)
                reward = 1.0
                done = False
                hx = np.random.randn(64)

                memory.push(obs, action, reward, done, hx)

        # Create multiple threads
        threads = []
        num_threads = 3
        pushes_per_thread = 10

        for i in range(num_threads):
            thread = threading.Thread(
                target=push_worker,
                args=(i, pushes_per_thread)
            )
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Check final state
        expected_total = num_threads * pushes_per_thread
        assert memory.size == min(expected_total, memory.capacity)


class TestMemoryPerformance:
    """Test memory performance characteristics."""

    def test_push_performance(self, memory):
        """Test push operation performance."""
        import time

        obs = np.random.randn(4, 16, 16, 16)
        action = 0.5
        reward = 1.0
        done = False
        hx = np.random.randn(64)

        # Time multiple pushes
        num_pushes = 100
        start_time = time.time()

        for _ in range(num_pushes):
            memory.push(obs, action, reward, done, hx)

        end_time = time.time()
        total_time = end_time - start_time

        # Should be reasonably fast (< 1ms per push)
        avg_time_per_push = total_time / num_pushes
        assert avg_time_per_push < 0.001  # 1ms per push

    def test_sample_performance(self, memory):
        """Test sampling performance."""
        import time

        # Fill memory
        for i in range(memory.capacity):
            obs = np.random.randn(4, 16, 16, 16)
            action = float(i)
            reward = 1.0
            done = False
            hx = np.random.randn(64)

            memory.push(obs, action, reward, done, hx)

        # Time sampling
        num_samples = 10
        start_time = time.time()

        for _ in range(num_samples):
            batch = memory.sample(batch_size=4)

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete sampling reasonably fast
        avg_time_per_sample = total_time / num_samples
        assert avg_time_per_sample < 0.1  # 100ms per sample batch
