"""
ATOM MIND: MEMORY (Hippocampus)
-------------------------------
Efficient Ring Buffer with Sequence Replay (Liquid-Compatible).
Optimized for RAM usage (Float16) and LTC temporal consistency.

NVIDIA-grade fixes (drop-in, same API):
- Ring-correct sampling using step_id continuity (wrap-around safe).
- Save/load now includes PPO buffers (log_prob/value) with backward compatibility.
- Stronger shape validation for PPO fields to prevent silent corruption.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import numpy as np
import torch

from atom.config import get_config
from atom.exceptions import MemoryError
from atom.logging import get_logger

logger = get_logger("memory")


def _to_numpy(x: Any) -> Any:
    """Best-effort conversion of torch tensors to numpy arrays; pass-through for numpy/scalars."""
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return x


def _as_1d_float32(x: Any) -> np.ndarray:
    """Convert scalar/array-like to 1D float32 numpy array."""
    x = _to_numpy(x)
    if np.isscalar(x):
        return np.array([x], dtype=np.float32)
    arr = np.asarray(x, dtype=np.float32)
    return arr.reshape(-1)


class AtomMemory:
    """Efficient ring buffer for experience replay."""

    def __init__(
        self,
        capacity: int = 10000,
        seq_len: int = 16,
        obs_shape: Tuple[int, ...] = (4, 32, 32, 32),
        act_dim: int = 1,
        hx_dim: int = 64,
    ):
        """
        Args:
            capacity: Max steps to store (Pre-allocated RAM).
            seq_len: Sequence length for LTC Unroll.
            obs_shape: Shape of observation tensors.
            act_dim: Dimension of action vectors.
            hx_dim: Dimension of the liquid hidden state.
        """
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        if act_dim <= 0:
            raise ValueError("act_dim must be > 0")
        if hx_dim <= 0:
            raise ValueError("hx_dim must be > 0")

        self.capacity = int(capacity)
        self.seq_len = int(seq_len)
        self.obs_shape = tuple(obs_shape)
        self.act_dim = int(act_dim)
        self.hx_dim = int(hx_dim)

        self.ptr = 0
        self.size = 0

        # Monotonic step counter to make ring sampling temporally correct.
        # Buffer stores the step_id for each slot; unwritten slots are -1.
        self._next_step_id = 0
        self.step_id_buf = np.full((self.capacity,), -1, dtype=np.int64)

        # PRE-ALLOCATE MEMORY (Contiguous RAM blocks)
        logger.info(f"Allocating {self.capacity} steps (Obs: {self.obs_shape})...")
        self.obs_buf = np.zeros((self.capacity, *self.obs_shape), dtype=np.float16)

        # Float32 for physics scalars / PPO bookkeeping
        self.act_buf = np.zeros((self.capacity, self.act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((self.capacity, 1), dtype=np.float32)
        self.done_buf = np.zeros((self.capacity, 1), dtype=np.float32)
        self.hx_buf = np.zeros((self.capacity, self.hx_dim), dtype=np.float32)  # Liquid state preservation

        # PPO-specific buffers (importance sampling)
        # Keep original shape for drop-in compatibility.
        self.log_prob_buf = np.zeros((self.capacity, 1), dtype=np.float32)
        self.value_buf = np.zeros((self.capacity, 1), dtype=np.float32)

        # Scientist Theory (10D) & Trust (1D verified)
        self.theory_buf = np.zeros((self.capacity, 10), dtype=np.float32)
        self.trust_buf = np.zeros((self.capacity, 1), dtype=np.float32)

        logger.info(f"Initialized AtomMemory: capacity={self.capacity}, seq_len={self.seq_len}")

    def push(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor, float],
        reward: float,
        done: bool,
        hx: Union[np.ndarray, torch.Tensor],
        log_prob: Optional[Union[np.ndarray, torch.Tensor]] = None,
        value: Optional[Union[np.ndarray, torch.Tensor]] = None,
        theory: Optional[Union[np.ndarray, torch.Tensor, float]] = 0.0,
        trust: Optional[Union[np.ndarray, torch.Tensor, float]] = 0.0,
    ) -> None:
        """Store one transition in the pre-allocated ring buffer."""
        try:
            idx = self.ptr

            obs = _to_numpy(obs)
            hx = _to_numpy(hx)

            # Validate / coerce action
            action_arr = _as_1d_float32(action)
            if action_arr.shape[0] != self.act_dim:
                if self.act_dim == 1 and action_arr.shape[0] == 1:
                    pass
                else:
                    raise ValueError(f"action has shape {action_arr.shape}, expected ({self.act_dim},)")

            # Reward/done
            rew_val = float(reward)
            done_val = 1.0 if bool(done) else 0.0

            # PPO fields (optional)
            logp_arr: Optional[np.ndarray] = None
            if log_prob is not None:
                logp_arr = _as_1d_float32(log_prob)
                # FIX (audit bug #4): Accept scalar log_prob for ANY act_dim.
                # Training loop sums log_prob across action dims before pushing,
                # and the buffer stores a single scalar anyway (log_prob_buf[:, 0]).
                # Old code rejected scalar when act_dim > 1, blocking multi-action envs.
                if logp_arr.shape[0] == 1:
                    pass  # scalar log_prob: always accepted
                elif logp_arr.shape[0] == self.act_dim:
                    pass  # per-dimension log_prob: accepted, will be summed at storage
                else:
                    raise ValueError(
                        f"log_prob has shape {logp_arr.shape}, expected (1,) or ({self.act_dim},)"
                    )

            val_arr: Optional[np.ndarray] = None
            if value is not None:
                val_arr = _as_1d_float32(value)
                if val_arr.shape[0] != 1:
                    raise ValueError(f"value has shape {val_arr.shape}, expected (1,)")

            # Theory/Trust validation (V1=1D, V2=10D/1D)
            theory_arr = _as_1d_float32(theory)
            trust_arr = _as_1d_float32(trust)
            if theory_arr.shape[0] not in [1, 10]:
                raise ValueError(f"theory has shape {theory_arr.shape}, expected (1,) or (10,)")
            if trust_arr.shape[0] != 1:
                raise ValueError(f"trust has shape {trust_arr.shape}, expected (1,)")

            # Store
            self.obs_buf[idx] = obs
            self.act_buf[idx] = action_arr
            self.rew_buf[idx, 0] = rew_val
            self.done_buf[idx, 0] = done_val
            self.hx_buf[idx] = hx

            if logp_arr is not None:
                # Always store as summed scalar (training loop sums before push).
                self.log_prob_buf[idx, 0] = float(logp_arr.sum())

            if val_arr is not None:
                self.value_buf[idx, 0] = float(val_arr[0])

            # Scientist signals (V1 scalar or V2 10D Structural)
            theory_val = _as_1d_float32(theory)
            if theory_val.shape[0] == 10:
                self.theory_buf[idx] = theory_val
            else:
                # Fallback / Padding for V1 compatibility
                self.theory_buf[idx, 0] = theory_val[0]
            
            trust_val = float(_as_1d_float32(trust)[0])
            self.trust_buf[idx, 0] = trust_val

            # Step id bookkeeping (defines temporal continuity)
            self.step_id_buf[idx] = self._next_step_id
            self._next_step_id += 1

            # Advance ring
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

        except Exception as e:
            logger.error(f"Error pushing to memory: {e}")
            raise MemoryError(f"Memory push failed: {e}") from e

    def sample(self, batch_size: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Sample SEQUENCES for the Liquid Brain.
        Returns a dict of numpy arrays.

        Keys preserved (drop-in):
          obs, action, reward, done, hx, old_log_prob, old_value, theory, trust, next_obs, next_hx
        """
        try:
            if self.size <= self.seq_len:
                logger.warning(f"Memory size {self.size} too small for seq_len {self.seq_len}")
                return None

            if batch_size <= 0:
                raise ValueError("batch_size must be > 0")

            valid_starts = self._compute_valid_starts()
            if valid_starts.size == 0:
                logger.warning("No valid sequences found for sampling")
                return None

            # Sample starts (with replacement if needed)
            replace = valid_starts.size < batch_size
            starts = np.random.choice(valid_starts, size=batch_size, replace=replace)

            # Gather indices for each sequence with wrap-around
            t_offsets = np.arange(self.seq_len, dtype=np.int64)
            seq_indices = (starts[:, None] + t_offsets[None, :]) % self.capacity  # (B, T)

            # next index for bootstrap (may be across episode boundary; caller masks via done)
            next_indices = (starts + self.seq_len) % self.capacity  # (B,)

            # Batch gather
            batch_obs = self.obs_buf[seq_indices]  # (B, T, *obs_shape)
            batch_act = self.act_buf[seq_indices]  # (B, T, act_dim)
            batch_rew = self.rew_buf[seq_indices]  # (B, T, 1)
            batch_done = self.done_buf[seq_indices]  # (B, T, 1)

            # Initial hx at sequence start
            batch_hx = self.hx_buf[starts]  # (B, hx_dim)

            # PPO fields
            batch_log_prob = self.log_prob_buf[seq_indices]  # (B, T, act_dim)
            batch_value = self.value_buf[seq_indices]  # (B, T, 1)

            batch_theory = self.theory_buf[seq_indices]  # (B, T, 1)
            batch_trust = self.trust_buf[seq_indices]  # (B, T, 1)

            batch_next_obs = self.obs_buf[next_indices]  # (B, *obs_shape)
            batch_next_hx = self.hx_buf[next_indices]  # (B, hx_dim)

            return {
                "obs": batch_obs,
                "action": batch_act,
                "reward": batch_rew,
                "done": batch_done,
                "hx": batch_hx,
                "old_log_prob": batch_log_prob,
                "old_value": batch_value,
                "theory": batch_theory,
                "trust": batch_trust,
                "next_obs": batch_next_obs,
                "next_hx": batch_next_hx,
            }

        except Exception as e:
            logger.error(f"Error sampling from memory: {e}")
            raise MemoryError(f"Memory sampling failed: {e}") from e

    def _compute_valid_starts(self) -> np.ndarray:
        """
        Compute valid sequence start indices based on:
        - step_id continuity (true temporal adjacency, ring-safe)
        - do not cross episode boundary inside the sequence (done allowed only at last step)
        - avoid reading unwritten slots (step_id == -1)
        - avoid landing next_obs on an unwritten slot unless terminal at end
        """
        # If buffer not full, the valid physical indices are [0, size-1] (no wrap yet).
        # If full, all indices are potentially valid, and step_id_buf defines continuity.
        if self.size < self.capacity:
            candidate = np.arange(0, self.size, dtype=np.int64)
        else:
            candidate = np.arange(0, self.capacity, dtype=np.int64)

        valid: List[int] = []
        t_offsets = np.arange(self.seq_len, dtype=np.int64)

        for s in candidate:
            seq = (s + t_offsets) % self.capacity  # (T,)
            sid = self.step_id_buf[seq]
            if np.any(sid < 0):
                continue

            # Must be consecutive in time: sid[t] = sid[0] + t
            if not np.all(sid == (sid[0] + t_offsets)):
                continue

            # Episode boundary: done cannot occur in the middle.
            if np.any(self.done_buf[seq[:-1], 0] > 0.5):
                continue

            # next index validity:
            next_i = (s + self.seq_len) % self.capacity
            done_last = bool(self.done_buf[seq[-1], 0] > 0.5)
            if not done_last:
                # If not terminal, next step should exist and be consecutive.
                if self.step_id_buf[next_i] != sid[-1] + 1:
                    continue
            else:
                # terminal: next step may be a reset state; still must be written to be meaningful.
                if self.step_id_buf[next_i] < 0:
                    continue

            valid.append(int(s))

        return np.asarray(valid, dtype=np.int64)

    def _is_valid_sequence(self, idx: int) -> bool:
        """
        Backward-compatible validity hook.

        Your old code used ptr/size logic and forbade wrap-around. :contentReference[oaicite:11]{index=11}
        We keep this method for compatibility, but route through the new ring-correct rule.
        """
        try:
            idx = int(idx)
            if idx < 0 or idx >= self.capacity:
                return False
            # New truth: validity is defined by step_id continuity + done masking.
            valid_starts = self._compute_valid_starts()
            return bool(np.any(valid_starts == idx))
        except Exception:
            return False

    def save(self, filepath: Union[str, Path] = "atom_memory.npz") -> None:
        """Save memory buffers to disk (now includes PPO buffers; backward compatible)."""
        filepath = Path(filepath)
        logger.info(f"Saving {self.size} steps to {filepath}...")

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Save ALL buffers (full capacity) to preserve ring structure precisely.
            # This is safer than saving only [:size] when ptr!=size due to wrap-around.
            np.savez_compressed(
                filepath,
                capacity=np.array([self.capacity], dtype=np.int64),
                seq_len=np.array([self.seq_len], dtype=np.int64),
                act_dim=np.array([self.act_dim], dtype=np.int64),
                hx_dim=np.array([self.hx_dim], dtype=np.int64),

                obs=self.obs_buf,
                act=self.act_buf,
                rew=self.rew_buf,
                done=self.done_buf,
                hx=self.hx_buf,

                # PPO buffers (critical for resume correctness)
                log_prob=self.log_prob_buf,
                value=self.value_buf,

                # Scientist fields
                theory=self.theory_buf,
                trust=self.trust_buf,

                # Ring metadata
                ptr=np.array([self.ptr], dtype=np.int64),
                size=np.array([self.size], dtype=np.int64),

                # Temporal continuity metadata
                step_id=self.step_id_buf,
                next_step_id=np.array([self._next_step_id], dtype=np.int64),
            )
            logger.info("Memory save complete")

        except Exception as e:
            logger.error(f"Memory save failed: {e}")
            raise MemoryError(f"Memory save failed: {e}") from e

    def load(self, filepath: Union[str, Path] = "atom_memory.npz") -> None:
        """Load memory buffers from disk (backward compatible with old save files)."""
        filepath = Path(filepath)

        if not filepath.exists():
            msg = f"No valid save file found at {filepath}"
            logger.error(msg)
            raise MemoryError(msg)

        logger.info(f"Loading from {filepath}...")
        try:
            data = np.load(filepath)

            # Restore metadata
            saved_size = int(data["size"][0]) if "size" in data else 0
            saved_ptr = int(data["ptr"][0]) if "ptr" in data else 0

            if saved_size > self.capacity:
                logger.warning("Save file larger than capacity. Truncating.")
                saved_size = self.capacity

            self.size = saved_size
            self.ptr = saved_ptr % self.capacity

            # Old format saved only prefix slices; new format saves full capacity.
            # Handle both.
            if "obs" in data and data["obs"].shape[0] == self.capacity:
                # New format: full buffers
                self.obs_buf[:] = data["obs"]
                self.act_buf[:] = data["act"]
                self.rew_buf[:] = data["rew"]
                self.done_buf[:] = data["done"]
                self.hx_buf[:] = data["hx"]
            else:
                # Old format: only valid prefix (your previous implementation) :contentReference[oaicite:12]{index=12}
                self.obs_buf[: self.size] = data["obs"][: self.size]
                self.act_buf[: self.size] = data["act"][: self.size]
                self.rew_buf[: self.size] = data["rew"][: self.size]
                self.done_buf[: self.size] = data["done"][: self.size]
                self.hx_buf[: self.size] = data["hx"][: self.size]

            # Scientist fields (backward compatible)
            if "theory" in data:
                if data["theory"].shape[0] == self.capacity:
                    self.theory_buf[:] = data["theory"]
                else:
                    self.theory_buf[: self.size] = data["theory"][: self.size]
            if "trust" in data:
                if data["trust"].shape[0] == self.capacity:
                    self.trust_buf[:] = data["trust"]
                else:
                    self.trust_buf[: self.size] = data["trust"][: self.size]

            # PPO fields (new; if missing in old files, keep zeros)
            if "log_prob" in data:
                if data["log_prob"].shape[0] == self.capacity:
                    self.log_prob_buf[:] = data["log_prob"]
                else:
                    self.log_prob_buf[: self.size] = data["log_prob"][: self.size]
            if "value" in data:
                if data["value"].shape[0] == self.capacity:
                    self.value_buf[:] = data["value"]
                else:
                    self.value_buf[: self.size] = data["value"][: self.size]

            # Temporal continuity (new; if missing, reconstruct best-effort)
            if "step_id" in data:
                if data["step_id"].shape[0] == self.capacity:
                    self.step_id_buf[:] = data["step_id"]
                else:
                    self.step_id_buf[:] = -1
                    self.step_id_buf[: self.size] = data["step_id"][: self.size]
            else:
                # Best-effort reconstruction for old files: assign consecutive IDs to prefix region.
                # This cannot reconstruct true time ordering if ptr indicates wrap.
                self.step_id_buf[:] = -1
                if self.size > 0:
                    self.step_id_buf[: self.size] = np.arange(self.size, dtype=np.int64)

            if "next_step_id" in data:
                self._next_step_id = int(data["next_step_id"][0])
            else:
                # Conservative: next id = max(step_id)+1
                mx = int(np.max(self.step_id_buf)) if np.any(self.step_id_buf >= 0) else -1
                self._next_step_id = mx + 1

            logger.info(f"Loaded {self.size} steps")

        except Exception as e:
            logger.error(f"Memory load failed: {e}")
            raise MemoryError(f"Memory load failed: {e}") from e

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "capacity": self.capacity,
            "size": self.size,
            "utilization": self.size / self.capacity,
            "pointer": self.ptr,
            "obs_shape": self.obs_shape,
            "act_dim": self.act_dim,
        }

    def clear(self) -> None:
        """Clear all memory buffers."""
        self.ptr = 0
        self.size = 0
        self._next_step_id = 0
        self.step_id_buf[:] = -1
        logger.info("Memory buffers cleared")


def create_memory_from_config(config: Any = None) -> AtomMemory:
    """Create an AtomMemory instance from the configuration."""
    if config is None:
        config = get_config()
    return AtomMemory(
        capacity=config.memory.capacity,
        seq_len=config.memory.sequence_length,
        obs_shape=(4, *config.physics.grid_shape),
        act_dim=config.brain.action_dim,
        hx_dim=config.brain.internal_neurons,
    )


if __name__ == "__main__":
    print(">>> ATOM MEMORY: Initializing Efficient Buffer...")
    mem = AtomMemory(capacity=100, seq_len=4)

    # Fake Data
    obs = np.random.randn(4, 32, 32, 32)
    hx = np.random.randn(64)

    for i in range(120):  # force wrap
        mem.push(obs, 0.5, 0.1, (i % 17 == 0), hx)

    print(f"   Stored {mem.size} items. Pointer: {mem.ptr}")

    batch = mem.sample(2)
    if batch:
        print(f"   Sampled Batch Obs Shape: {batch['obs'].shape}")
        print(f"   Sampled Batch HX Shape: {batch['hx'].shape}")
        print(f"   Sampled Batch old_log_prob Shape: {batch['old_log_prob'].shape}")
    else:
        print("   Batch sampling failed.")