"""
ATOM CORE: BRAIN (Controller)
-----------------------------
The Neuro-Symbolic Executive.

What this module actually does (no marketing):
- Perception: compresses a 3D fluid observation into a latent vector (Eyes).
- Cognition: adapts a low-dim "theory" signal into a controller conditioning vector.
- Reflex/Dynamics: runs a recurrent core (LTC if available; GRU fallback otherwise).
- Constraint: projects the internal state through either:
  - SymplecticManifold (Hamiltonian-ish constraint), or
  - NeuralSkeleton (low-rank linear manifold with delayed Hebbian updates).
- Control: outputs Gaussian policy parameters (mu, std) + critic value + stress proxy.

NOTE ON SHAPES:
- obs_frame: (B, 4, X, Y, Z) is assumed throughout this repo.
- visual_latent: (B, vision_dim) (NOT hardcoded 256).
- theory_signal: (B, theory_dim) (often 1).
- last_action: (B, action_dim) (often 1).
"""

from __future__ import annotations

from typing import Optional, Tuple, Any

import torch
import torch.nn as nn

from atom.config import get_config
from atom.exceptions import BrainError, InitializationError
from atom.logging import get_logger

logger = get_logger("brain")

# --- Optional dependency: NCPS / LTC ---
try:
    from ncps.torch import LTC
    from ncps.wirings import AutoNCP

    HAS_LTC = True
except ImportError:
    logger.warning("'ncps' library not found. Falling back to GRU.")
    logger.info("Run 'pip install ncps' for full Liquid Physics capabilities.")
    HAS_LTC = False

# Eyes (Hodge/FNO stack)
from .eyes2 import AtomEyes

# --- Optional dependency: Symplectic constraint ---
try:
    from .symplectic import SymplecticManifold, HamiltonianNet

    HAS_SYMPLECTIC = True
except ImportError:
    HAS_SYMPLECTIC = False


# =============================================================================
# 1) THE SKELETON (Safety Constraint): Low-rank linear manifold + delayed Hebbian
# =============================================================================
class NeuralSkeleton(nn.Module):
    """
    Low-rank linear manifold projector with delayed Hebbian-style updates.

    - components: (num_bones, num_inputs), frozen (NOT optimized by Adam)
    - forward():
        y = x @ components^T
        x_recon = y @ components
        stress = mean((x - x_recon)^2)
        accumulates growth_buffer += y^T @ (x - x_recon) / B   (no-grad)
    - apply_growth():
        components += plasticity * (possibly allreduced) growth_buffer
        re-orthogonalize rows via Gram-Schmidt
    """

    def __init__(self, num_inputs: int, num_bones: int = 8, plasticity: float = 0.01):
        super().__init__()
        if num_inputs <= 0:
            raise InitializationError(f"NeuralSkeleton: num_inputs must be > 0, got {num_inputs}")
        if num_bones <= 0:
            raise InitializationError(f"NeuralSkeleton: num_bones must be > 0, got {num_bones}")

        self.num_bones = int(num_bones)
        self.plasticity = float(plasticity)

        # Frozen manifold basis (rows)
        self.components = nn.Parameter(torch.randn(self.num_bones, num_inputs), requires_grad=False)
        with torch.no_grad():
            nn.init.orthogonal_(self.components)

        # Accumulate delayed updates here
        self.register_buffer("growth_buffer", torch.zeros_like(self.components))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, num_inputs)

        Returns:
            x_recon: (B, num_inputs)
            stress:  (B, 1)
        """
        if x.dim() != 2:
            raise BrainError(f"NeuralSkeleton.forward expected x as (B, D), got shape={tuple(x.shape)}")

        # Project -> reconstruct
        y = torch.mm(x, self.components.t())          # (B, K)
        x_recon = torch.mm(y, self.components)        # (B, D)

        # Reconstruction error as "stress" proxy
        stress = torch.mean((x - x_recon) ** 2, dim=1, keepdim=True)  # (B, 1)

        # Delayed Hebbian accumulation (NO gradient / BPTT-safe)
        if self.training:
            with torch.no_grad():
                delta = torch.mm(y.t(), (x - x_recon))  # (K, D)
                self.growth_buffer.add_(delta / max(int(x.shape[0]), 1))

        return x_recon, stress

    def apply_growth(self) -> None:
        """
        Apply accumulated growth_buffer to components, then re-orthogonalize.

        DDP note:
        If torch.distributed is initialized, we all-reduce the buffer so ranks
        don't diverge. This keeps multi-GPU behavior sane.
        """
        with torch.no_grad():
            gb = self.growth_buffer

            # --- DDP synchronization (best-effort, no hard dependency) ---
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(gb, op=torch.distributed.ReduceOp.SUM)
                world = torch.distributed.get_world_size()
                if world > 1:
                    gb.div_(world)

            # Apply update
            self.components.add_(self.plasticity * gb)
            gb.zero_()

            # Gram-Schmidt re-orthogonalization of row vectors
            for i in range(self.num_bones):
                # Remove projections onto previous bones
                for j in range(i):
                    proj = torch.dot(self.components[i], self.components[j])
                    self.components[i].sub_(proj * self.components[j])

                # Normalize
                norm = torch.norm(self.components[i])
                self.components[i].div_(norm + 1e-8)


# =============================================================================
# 2) THE BRAIN (Controller): Eyes -> Liquid Core -> Constraint -> Policy/Value
# =============================================================================
class AtomBrain(nn.Module):
    """
    Merges perception, theory conditioning, recurrent dynamics, and constraints.

    Public API preserved:
    - __init__(..., vision_dim, internal_neurons, action_dim, theory_dim, bones,
               use_symplectic, vision_mode, config)
    - forward(obs_frame, theory_signal, last_action, hx=None, theory_confidence=None)
    - apply_hebbian_growth()
    """

    def __init__(
        self,
        vision_dim: int = 64,
        internal_neurons: int = 16,
        action_dim: int = 1,
        theory_dim: int = 10,
        bones: int = 8,
        use_symplectic: bool = True,
        vision_mode: str = "fno",
        config: Any = None,
    ):
        super().__init__()

        if config is None:
            config = get_config()

        if vision_dim <= 0:
            raise InitializationError(f"AtomBrain: vision_dim must be > 0, got {vision_dim}")
        if internal_neurons <= 0:
            raise InitializationError(f"AtomBrain: internal_neurons must be > 0, got {internal_neurons}")
        if action_dim <= 0:
            raise InitializationError(f"AtomBrain: action_dim must be > 0, got {action_dim}")
        if theory_dim <= 0:
            raise InitializationError(f"AtomBrain: theory_dim must be > 0, got {theory_dim}")

        self.vision_dim = int(vision_dim)
        self.internal_neurons = int(internal_neurons)
        self.action_dim = int(action_dim)
        self.theory_dim = int(theory_dim)
        self.vision_mode = str(vision_mode)

        # Gate symplectic usage by availability
        self.use_symplectic = bool(use_symplectic) and HAS_SYMPLECTIC
        if bool(use_symplectic) and not HAS_SYMPLECTIC:
            logger.warning(
                "Symplectic manifold requested but 'symplectic' module not available. "
                "Falling back to NeuralSkeleton. Energy conservation claims do NOT apply."
            )

        # ---------------------------------------------------------------------
        # A) PERCEPTION (Eyes)
        # ---------------------------------------------------------------------
        if self.vision_mode == "fno":
            # eyes2.AtomEyes is expected to provide .embed()
            self.eyes = AtomEyes(
                modes=config.eyes.fno_modes,
                width=config.eyes.fno_width,
                depth=config.eyes.fno_depth,
                embedding_dim=self.vision_dim,
            )
        else:
            # Baseline ablation: pooled linear projection, robust to any grid size.
            self.eyes = nn.Sequential(
                nn.AdaptiveAvgPool3d((4, 4, 4)),
                nn.Flatten(),
                nn.Linear(4 * 4 * 4 * 4, self.vision_dim),
            )

            # Provide .embed() for compatibility with the FNO path
            self.eyes.embed = lambda d: self.eyes(d)

        # ---------------------------------------------------------------------
        # B) COGNITION (Theory adapter)
        # ---------------------------------------------------------------------
        self.theory_adapter = nn.Sequential(
            nn.Linear(int(theory_dim), 16),
            nn.Tanh(),
        )

        # ---------------------------------------------------------------------
        # C) LIQUID CORE (LTC preferred; GRU fallback)
        # ---------------------------------------------------------------------
        input_size = self.vision_dim + 16 + self.action_dim
        self.ltc_norm = nn.LayerNorm(input_size)

        if HAS_LTC:
            wiring = AutoNCP(self.internal_neurons, self.action_dim)
            self.liquid = LTC(input_size, wiring, batch_first=True)
        else:
            self.liquid = nn.GRU(input_size, self.internal_neurons, batch_first=True)
            # Keep this for backward compatibility (even if unused elsewhere)
            self.motor_head_gru = nn.Linear(self.internal_neurons, self.action_dim)

        # ---------------------------------------------------------------------
        # D) CONSTRAINT (Symplectic manifold vs NeuralSkeleton)
        # ---------------------------------------------------------------------
        if self.use_symplectic:
            # These are still "magic numbers" — but we keep them to stay drop-in.
            # If you want them config-driven, we can do that safely next.
            self.skeleton = SymplecticManifold(
                state_dim=self.internal_neurons,
                hidden_dim=64,
                dt=0.1,
                dissipation=0.001,
            )
            logger.info("Using Symplectic Manifold (energy-controlled; small dissipation enabled)")
        else:
            self.skeleton = NeuralSkeleton(self.internal_neurons, num_bones=int(bones))

        # Physics monitor for logging only (energy-like scalar)
        self.phys_monitor = HamiltonianNet(state_dim=self.internal_neurons, hidden_dim=64) if HAS_SYMPLECTIC else None

        # ---------------------------------------------------------------------
        # E) POLICY + VALUE HEADS
        # ---------------------------------------------------------------------
        self.actor_mu = nn.Linear(self.internal_neurons, self.action_dim)
        # FIX (audit bug #1): actor_log_std was a bare nn.Parameter, invisible to
        # optimizer groups built from named submodules (eyes, actor_mu, critic, etc.).
        # This froze exploration noise for the entire training run.
        # Wrapping in nn.ParameterDict (an nn.Module subclass) makes it discoverable
        # via self.actor_std_param.parameters() in training_loop optimizer groups.
        self.actor_std_param = nn.ParameterDict({
            "log_std": nn.Parameter(torch.zeros(1, self.action_dim) - 0.5)
        })

        # IMPORTANT FIX (stability): critic now conditions on the SAME state as actor by default.
        # This avoids policy/value state mismatch that can destabilize PPO. (See audit.) :contentReference[oaicite:3]{index=3}
        self.critic = nn.Linear(self.internal_neurons, 1)

        logger.info(
            f"Initialized AtomBrain: vision_dim={self.vision_dim}, internal_neurons={self.internal_neurons}, "
            f"action_dim={self.action_dim}, vision_mode={self.vision_mode}, use_symplectic={self.use_symplectic}"
        )

    @property
    def actor_log_std(self) -> nn.Parameter:
        """Backward-compatible accessor. Parameter lives in self.actor_std_param."""
        return self.actor_std_param["log_std"]

    @staticmethod
    def _normalize_recurrent_state(h: torch.Tensor, batch_size: int, hidden_dim: int) -> torch.Tensor:
        """
        Normalize recurrent hidden state shapes to (B, H).

        Handles common variants:
        - (B, H)
        - (1, B, H)   (typical RNN/LTC)
        - (T, B, H)   (unexpected but we handle by taking last T)
        """
        if h is None:
            raise BrainError("Internal error: _normalize_recurrent_state received None")

        if h.dim() == 2:
            # (B, H)
            return h
        if h.dim() == 3:
            # (N, B, H) or (1, B, H)
            # Take last along first dim.
            if h.shape[1] != batch_size or h.shape[2] != hidden_dim:
                # Best-effort: attempt to reshape common (B, 1, H) style bugs
                # but do not silently accept incompatible shapes.
                raise BrainError(f"Unexpected recurrent state shape={tuple(h.shape)} (expected (*,B,H)=(*,{batch_size},{hidden_dim}))")
            return h[-1]
        raise BrainError(f"Unexpected recurrent state rank={h.dim()} shape={tuple(h.shape)}")


    def _coerce_theory_signal(
        self,
        theory_signal: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # Normalize theory inputs to (B, theory_dim) with explicit compatibility rules.
        if not torch.is_tensor(theory_signal):
            theory_signal = torch.as_tensor(theory_signal, device=device, dtype=dtype)
        else:
            theory_signal = theory_signal.to(device=device, dtype=dtype)

        if theory_signal.dim() == 1:
            if batch_size == 1 and theory_signal.numel() == self.theory_dim:
                theory_signal = theory_signal.view(1, self.theory_dim)
            elif theory_signal.numel() == batch_size:
                theory_signal = theory_signal.view(batch_size, 1)
            else:
                theory_signal = theory_signal.view(1, -1)
        elif theory_signal.dim() != 2:
            raise BrainError(
                f"theory_signal must be rank-1 or rank-2, got shape={tuple(theory_signal.shape)}"
            )

        if theory_signal.shape[0] != batch_size:
            if theory_signal.shape[0] == 1 and batch_size > 1:
                theory_signal = theory_signal.expand(batch_size, -1)
            else:
                raise BrainError(
                    f"theory_signal batch mismatch: expected B={batch_size}, got shape={tuple(theory_signal.shape)}"
                )

        if theory_signal.shape[1] == self.theory_dim:
            return theory_signal

        # Backward-compatible scalar path for the V2 10D theory adapter.
        if self.theory_dim == 10 and theory_signal.shape[1] == 1:
            padded = torch.zeros((batch_size, 10), device=device, dtype=theory_signal.dtype)
            padded[:, :1] = theory_signal
            return padded

        raise BrainError(
            f"theory_signal feature mismatch: expected {self.theory_dim}, got {theory_signal.shape[1]}"
        )

    def forward(
        self,
        obs_frame: torch.Tensor,
        theory_signal: torch.Tensor,
        last_action: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
        theory_confidence: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The Thinking Loop.

        Args:
            obs_frame: (B, 4, X, Y, Z) raw fluid state
            theory_signal: (B, theory_dim) scalar/vector from Scientist
            last_action: (B, action_dim) previous action (proprioception)
            hx: recurrent hidden state (implementation-dependent)
            theory_confidence: (B, 1) trust score in [0,1] for constraint blending

        Returns:
            (mu, std): policy parameters, both (B, action_dim)
            value: (B, 1)
            new_hx: recurrent state (library-dependent shape)
            phys_stress: (B, 1) energy-like deviation or reconstruction stress
        """
        try:
            if obs_frame.dim() != 5:
                raise BrainError(f"AtomBrain.forward expected obs_frame as (B,4,X,Y,Z), got shape={tuple(obs_frame.shape)}")
            batch_size = int(obs_frame.shape[0])

            if last_action.dim() == 1:
                if self.action_dim != 1 or last_action.shape[0] != batch_size:
                    raise BrainError(
                        f"last_action shape mismatch: expected (B,{self.action_dim}), got {tuple(last_action.shape)}"
                    )
                last_action = last_action.view(batch_size, 1)
            elif last_action.dim() != 2:
                raise BrainError(
                    f"last_action must be rank-1 or rank-2, got shape={tuple(last_action.shape)}"
                )

            if last_action.shape != (batch_size, self.action_dim):
                raise BrainError(
                    f"last_action shape mismatch: expected (B,{self.action_dim}), got {tuple(last_action.shape)}"
                )

            last_action = last_action.to(device=obs_frame.device, dtype=obs_frame.dtype)
            theory_signal = self._coerce_theory_signal(
                theory_signal,
                batch_size=batch_size,
                device=obs_frame.device,
                dtype=obs_frame.dtype,
            )

            # 1) SEE
            visual_latent = self.eyes.embed(obs_frame)  # (B, vision_dim)

            # 2) REASON
            theory_vec = self.theory_adapter(theory_signal)  # (B, 16)

            # 3) FUSE
            combined = torch.cat([visual_latent, theory_vec, last_action], dim=-1)  # (B, input_size)
            combined = self.ltc_norm(combined)
            combined = combined.unsqueeze(1)  # (B, 1, input_size)

            # 4) THINK (recurrent dynamics)
            if HAS_LTC:
                x_out, new_hx = self.liquid(combined, hx)
                liquid_state = self._normalize_recurrent_state(new_hx, batch_size, self.internal_neurons)
            else:
                x_out, new_hx = self.liquid(combined, hx)
                liquid_state = x_out[:, -1, :]  # (B, H)

            # 5) CONSTRAIN
            constrained_state, stress = self.skeleton(liquid_state)  # both (B, H) and (B,1)

            # 6) TRUST GATE
            if theory_confidence is None:
                # Keep backward compatibility with your current behavior.
                # If you want a safer default, we can switch to alpha=0 (no constraint) or alpha=clamped(mean).
                bone_state = constrained_state
            else:
                alpha = theory_confidence.view(batch_size, 1).clamp(0.0, 1.0)
                bone_state = (1.0 - alpha) * liquid_state + alpha * constrained_state

            # 7) ACT
            mu = self.actor_mu(bone_state)  # (B, action_dim)

            # Guard against weird training explosions producing NaNs/Infs
            mu = torch.nan_to_num(mu, nan=0.0, posinf=1.0, neginf=-1.0)

            std = torch.exp(self.actor_std_param["log_std"]).expand_as(mu)
            std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=0.1).clamp_min(1e-6)

            # 8) VALUE  (FIX: align critic input with policy state)
            value = self.critic(bone_state)  # (B, 1)

            # 9) PHYSICAL STRESS (logging proxy)
            if self.phys_monitor is not None:
                current_energy = self.phys_monitor(liquid_state)  # (B,1)
                target = getattr(
                    self.skeleton,
                    "target_energy",
                    torch.tensor([1.0], device=obs_frame.device, dtype=current_energy.dtype),
                )
                target = target.to(device=current_energy.device, dtype=current_energy.dtype)
                phys_stress = (current_energy - target).pow(2).mean(dim=-1, keepdim=True)
            else:
                phys_stress = stress

            return (mu, std), value, new_hx, phys_stress

        except Exception as e:
            logger.error(f"Error in AtomBrain forward pass: {e}")
            raise BrainError(f"Brain forward pass failed: {e}") from e

    def apply_hebbian_growth(self) -> None:
        """Trigger the constraint's delayed update (NeuralSkeleton: basis update, Symplectic: target update)."""
        try:
            # SymplecticManifold in your repo already implements apply_growth() for compatibility. :contentReference[oaicite:4]{index=4}
            self.skeleton.apply_growth()
        except Exception as e:
            logger.error(f"Error applying Hebbian growth: {e}")
            raise BrainError(f"Hebbian growth failed: {e}") from e


def create_brain_from_config(
    config: Any = None,
    use_symplectic: Optional[bool] = None,
    vision_mode: Optional[str] = None,
) -> AtomBrain:
    """Create an AtomBrain instance from the configuration."""
    if config is None:
        config = get_config()

    symplectic = use_symplectic if use_symplectic is not None else True
    v_mode = vision_mode if vision_mode is not None else "fno"

    return AtomBrain(
        vision_dim=config.brain.vision_dim,
        theory_dim=config.brain.theory_dim,
        action_dim=config.brain.action_dim,
        internal_neurons=config.brain.internal_neurons,
        bones=config.brain.skeleton_bones,
        use_symplectic=symplectic,
        vision_mode=v_mode,
        config=config,
    )


if __name__ == "__main__":
    print(">>> ATOM BRAIN: Initializing Controller...")
    brain = create_brain_from_config()

    # Don't assume Eyes exposes width in all modes.
    eyes_width = getattr(brain.eyes, "width", None)
    if eyes_width is None:
        print(f"   Structure: Eyes({brain.vision_mode}) -> Liquid({brain.internal_neurons}) -> Motor")
    else:
        print(f"   Structure: Eyes(width={eyes_width}) -> Liquid({brain.internal_neurons}) -> Motor")

    B = 2
    dummy_obs = torch.randn(B, 4, 32, 32, 32)
    dummy_theory = torch.zeros(B, brain.theory_dim)
    dummy_theory[:, 0] = torch.tensor([0.5, -0.5])
    dummy_action = torch.zeros(B, brain.action_dim)

    print("   Running Forward Pass...")
    (mu, std), val, hx, stress = brain(dummy_obs, dummy_theory, dummy_action)

    print(f"   Action Mean: {mu.shape} (expected [{B}, {brain.action_dim}])")
    print(f"   Action Std:  {std.shape} (expected [{B}, {brain.action_dim}])")
    print(f"   Stress:      {stress.mean().item():.6f}")
    print(f"   Value Est:   {val.mean().item():.6f}")
    print("   ✅ Liquid Time-Constants Active." if HAS_LTC else "   ⚠️ Running in GRU Fallback Mode.")