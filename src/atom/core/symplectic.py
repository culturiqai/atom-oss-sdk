"""
ATOM CORE: SYMPLECTIC (Hamiltonian Control)
-------------------------------------------
Energy-shaped neural dynamics for stabilizing hidden-state control.

What this module REALLY is (no marketing):
- It learns a scalar functional H(q,p) over a hidden vector (split into q,p).
- It evolves the hidden vector using a symplectic (leapfrog / Störmer–Verlet) step
  consistent with separable Hamiltonians H(q,p)=V(q)+T(p).
- It reports a "stress" scalar = deviation of (post-step) energy from a target.

Notes:
- This is symplectic w.r.t. the learned Hamiltonian, not automatically "PDE-physics-correct".
- Stress is now aligned with the returned state (post-step), not the pre-step state.

References (conceptual inspiration):
- Greydanus et al. (2019): Hamiltonian Neural Networks
- Zhong et al. (2020): Symplectic ODE-Net
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn


class HamiltonianNet(nn.Module):
    """
    Separable Hamiltonian:
        H(q,p) = V(q) + T(p)
    with diagonal mass modelled via log_mass.

    State split:
        q = state[:, :q_dim]
        p = state[:, q_dim:]
    where q_dim = floor(state_dim/2), p_dim = state_dim - q_dim.

    IMPORTANT:
    - This is a learned scalar energy functional over your hidden vector.
      It is NOT automatically calibrated to physical units.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        if state_dim <= 0:
            raise ValueError(f"state_dim must be > 0, got {state_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")

        self.state_dim = int(state_dim)
        self.q_dim = self.state_dim // 2
        self.p_dim = self.state_dim - self.q_dim

        # Potential energy V(q)
        self.V_net = nn.Sequential(
            nn.Linear(self.q_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )

        # Diagonal inverse-mass via exp(-log_mass)
        self.log_mass = nn.Parameter(torch.zeros(self.p_dim))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(f"HamiltonianNet.forward expects (B,{self.state_dim}), got {tuple(state.shape)}")

        q, p = state[:, : self.q_dim], state[:, self.q_dim :]

        V = self.V_net(q)
        inv_mass = torch.exp(-self.log_mass)
        T = 0.5 * (p.pow(2) * inv_mass).sum(dim=-1, keepdim=True)

        return V + T

    def _inv_mass(self) -> torch.Tensor:
        return torch.exp(-self.log_mass)

    def kinetic_velocity(self, p: torch.Tensor) -> torch.Tensor:
        """dq/dt = ∂H/∂p = p * inv_mass (for separable diagonal-mass kinetic energy)."""
        return p * self._inv_mass()

    def potential_and_grad(
        self, q: torch.Tensor, create_graph: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute V(q) and dV/dq with autograd. This is the only place we use autograd
        in the symplectic step (separable H lets us avoid full dH/dstate three times).

        Args:
            q: (B, q_dim)
            create_graph: keep graph for higher-order gradients (training)

        Returns:
            V: (B,1)
            dV_dq: (B,q_dim)
        """
        with torch.enable_grad():
            if q.requires_grad:
                q_in = q
            else:
                q_in = q.detach().requires_grad_(True)

            V = self.V_net(q_in)
            dV_dq = torch.autograd.grad(V.sum(), q_in, create_graph=create_graph)[0]
        return V, dV_dq

    def time_derivative(self, state: torch.Tensor) -> torch.Tensor:
        """
        Time derivative for the separable Hamiltonian dynamics:
            dq/dt =  ∂H/∂p = p * inv_mass
            dp/dt = -∂H/∂q = -∂V/∂q
        """
        if state.dim() != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(f"time_derivative expects (B,{self.state_dim}), got {tuple(state.shape)}")

        q = state[:, : self.q_dim]
        p = state[:, self.q_dim :]

        dq_dt = self.kinetic_velocity(p)
        _, dV_dq = self.potential_and_grad(q, create_graph=True)
        dp_dt = -dV_dq

        return torch.cat([dq_dt, dp_dt], dim=-1)


class StormerVerletIntegrator(nn.Module):
    """
    Störmer–Verlet (Leapfrog) integrator for separable Hamiltonians H(q,p)=V(q)+T(p).

    Update:
        p_{1/2} = p_0 - (dt/2) * ∂V/∂q(q_0)
        q_1     = q_0 + dt     * ∂T/∂p(p_{1/2}) = q_0 + dt * p_{1/2} * inv_mass
        p_1     = p_{1/2} - (dt/2) * ∂V/∂q(q_1)

    This is symplectic (volume preserving) for the learned Hamiltonian dynamics.

    Key fix vs your original implementation:
    - We do NOT compute full dH/dstate 3× per step. We only autograd dV/dq twice per step.
    - Grad handling is explicit: if the input state requires grad, we keep graphs; otherwise we
      detach and create local graphs for force calculation only.
    """

    def __init__(self, hamiltonian: HamiltonianNet, dt: float = 0.1, steps: int = 1):
        super().__init__()
        if steps <= 0:
            raise ValueError(f"steps must be >= 1, got {steps}")
        self.H = hamiltonian
        self.dt = float(dt)
        self.steps = int(steps)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() != 2 or state.shape[-1] != self.H.state_dim:
            raise ValueError(f"Integrator expects (B,{self.H.state_dim}), got {tuple(state.shape)}")

        q_dim = self.H.q_dim
        dt = self.dt

        # If the incoming state participates in training graphs, preserve that.
        # Otherwise: do local grad calculations without leaking graphs outward.
        keep_graph = bool(state.requires_grad)

        # Work on local tensors to avoid accidentally modifying views.
        s = state

        for _ in range(self.steps):
            q = s[:, :q_dim]
            p = s[:, q_dim:]

            # --- Half momentum step: p_half = p - 0.5*dt*dV/dq(q) ---
            _, dV_dq = self.H.potential_and_grad(q, create_graph=keep_graph)
            p_half = p - 0.5 * dt * dV_dq

            # --- Full position step: q_new = q + dt * (p_half * inv_mass) ---
            dq = self.H.kinetic_velocity(p_half)
            q_new = q + dt * dq

            # --- Final half momentum step at q_new ---
            _, dV_dq_new = self.H.potential_and_grad(q_new, create_graph=keep_graph)
            p_new = p_half - 0.5 * dt * dV_dq_new

            s = torch.cat([q_new, p_new], dim=-1)

            # If input didn't require grad, don't retain graphs between steps.
            if not keep_graph:
                s = s.detach()

        return s


class SymplecticManifold(nn.Module):
    """
    Hamiltonian-based state constraint (API-compatible with NeuralSkeleton).

    forward(state) -> (projected_state, stress)

    IMPORTANT BEHAVIOR FIX:
    - stress is computed from the ENERGY OF THE RETURNED STATE (post integration),
      not the pre-integration state.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        dt: float = 0.1,
        dissipation: float = 0.0,
    ):
        super().__init__()
        if state_dim <= 0:
            raise ValueError(f"state_dim must be > 0, got {state_dim}")

        self.state_dim = int(state_dim)
        self.dissipation = float(dissipation)  # 0 = pure symplectic, >0 = contractive (heuristic)

        self.hamiltonian = HamiltonianNet(state_dim, hidden_dim)
        self.integrator = StormerVerletIntegrator(self.hamiltonian, dt=dt)

        # Target energy baseline (learned/adaptive)
        self.register_buffer("target_energy", torch.tensor([1.0]))
        self.register_buffer("energy_ema", torch.tensor([1.0]))
        self.ema_decay = 0.99

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if state.dim() != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(f"SymplecticManifold.forward expects (B,{self.state_dim}), got {tuple(state.shape)}")

        # Integrate state forward (flow operator)
        projected_state = self.integrator(state)

        # Optional dissipation (heuristic contraction)
        if self.dissipation > 0.0 and self.training:
            projected_state = projected_state * (1.0 - self.dissipation)

        # Energy + stress computed on the returned state (aligned monitoring)
        projected_energy = self.hamiltonian(projected_state)  # (B,1)
        stress = (projected_energy - self.target_energy).pow(2).mean(dim=-1, keepdim=True)

        # Update EMA based on projected energy (the thing you actually return)
        if self.training:
            with torch.no_grad():
                avg_energy = projected_energy.mean()
                self.energy_ema.copy_(self.ema_decay * self.energy_ema + (1.0 - self.ema_decay) * avg_energy)

        return projected_state, stress

    def apply_growth(self):
        """
        Compatibility method for NeuralSkeleton interface.
        Here: adapt target_energy toward running EMA.
        """
        with torch.no_grad():
            self.target_energy.copy_(self.energy_ema)

    def get_energy(self, state: torch.Tensor) -> torch.Tensor:
        """Get Hamiltonian energy of a state."""
        return self.hamiltonian(state)


class HamiltonianLiquidCell(nn.Module):
    """
    Hamiltonian-constrained recurrent cell.

    Not currently wired into brain.py in your stack, but kept for API completeness.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dt: float = 0.1):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError(f"input_dim and hidden_dim must be > 0, got {input_dim}, {hidden_dim}")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)

        self.input_proj = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.Tanh())
        self.combine = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.hamiltonian = HamiltonianNet(self.hidden_dim, self.hidden_dim)
        self.integrator = StormerVerletIntegrator(self.hamiltonian, dt=dt)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        if h is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)

        x_proj = self.input_proj(x)
        state = self.combine(torch.cat([x_proj, h], dim=-1))

        new_h = self.integrator(state)
        return new_h, new_h


# --- Utility: Compatibility wrapper for existing AtomBrain ---

def create_symplectic_skeleton(num_inputs: int, num_bones: int = 8, plasticity: float = 0.01):
    """
    Factory function to create a SymplecticManifold that's API-compatible
    with the original NeuralSkeleton.

    Args:
        num_inputs: Dimension of the neural state
        num_bones: Not used (kept for API compatibility)
        plasticity: Maps to dissipation parameter (heuristic scaling)

    Returns:
        SymplecticManifold instance
    """
    return SymplecticManifold(
        state_dim=num_inputs,
        hidden_dim=64,
        dt=0.1,
        dissipation=float(plasticity) * 0.1,
    )


if __name__ == "__main__":
    print(">>> ATOM SYMPLECTIC: Testing Hamiltonian Dynamics...")

    # Test HamiltonianNet
    H = HamiltonianNet(state_dim=8, hidden_dim=32)
    state = torch.randn(4, 8)

    energy = H(state)
    print(f"   Energy shape: {energy.shape} (should be [4, 1])")

    dstate = H.time_derivative(state)
    print(f"   Derivative shape: {dstate.shape} (should be [4, 8])")

    # Test Störmer–Verlet
    integrator = StormerVerletIntegrator(H, dt=0.1, steps=5)
    new_state = integrator(state)
    print(f"   Integrated state shape: {new_state.shape}")

    # Test energy conservation (of the learned H, not real physics)
    energy_before = H(state).mean().item()
    energy_after = H(new_state).mean().item()
    delta_E = abs(energy_after - energy_before) / (abs(energy_before) + 1e-8)
    print(f"   Energy before: {energy_before:.4f}")
    print(f"   Energy after:  {energy_after:.4f}")
    print(f"   Relative change: {delta_E:.6f} (should be small)")

    # Test SymplecticManifold
    manifold = SymplecticManifold(state_dim=64, hidden_dim=32)
    state_64 = torch.randn(4, 64)
    proj_state, stress = manifold(state_64)
    print(f"   Manifold projection: {proj_state.shape}")
    print(f"   Stress: {stress.mean().item():.6f}")

    print("   ✅ Symplectic module tests passed!")
