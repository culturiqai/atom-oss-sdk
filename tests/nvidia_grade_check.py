import torch
import torch.nn.functional as F
import numpy as np
import time
from atom.core.legacy.eyes import HelmholtzHead3d
from atom.core.symplectic import HamiltonianNet, StormerVerletIntegrator

def check_helmholtz():
    print("--- 1. NVIDIA-GRADE EYES CHECK: Helmholtz Solenoidality ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Grid parameters
    X, Y, Z = 64, 32, 24
    grid_h = 1.0
    
    # Initialize Head
    head = HelmholtzHead3d(in_width=128).to(device)
    dummy_input = torch.randn(1, X, Y, Z, 128).to(device)
    
    # Forward Pass
    uvw = head(dummy_input, grid_h=grid_h) # (B, 3, X, Y, Z)
    u, v, w = uvw[:, 0], uvw[:, 1], uvw[:, 2]
    
    # Numeric Divergence Check
    # div(u) = du/dx + dv/dy + dw/dz
    def grad(f, dim):
        pad_shape = [0]*6
        if dim == 1: pad_shape = (0,0, 0,0, 1,1)
        if dim == 2: pad_shape = (0,0, 1,1, 0,0)
        if dim == 3: pad_shape = (1,1, 0,0, 0,0)
        f_pad = F.pad(f, pad_shape, mode='circular')
        if dim == 1: return (f_pad[:, 2:, :, :] - f_pad[:, :-2, :, :]) / (2 * grid_h)
        if dim == 2: return (f_pad[:, :, 2:, :] - f_pad[:, :, :-2, :]) / (2 * grid_h)
        if dim == 3: return (f_pad[:, :, :, 2:] - f_pad[:, :, :, :-2]) / (2 * grid_h)
    
    du_dx = grad(u, 1)
    dv_dy = grad(v, 2)
    dw_dz = grad(w, 3)
    
    divergence = du_dx + dv_dy + dw_dz
    mean_div = torch.abs(divergence).mean().item()
    max_div = torch.abs(divergence).max().item()
    
    print(f"Mean Divergence: {mean_div:.2e}")
    print(f"Max Divergence:  {max_div:.2e}")
    
    # Solenoidal check (should be near machine epsilon for float32)
    is_solenoidal = mean_div < 1e-7
    print(f"Result: {'PASSED' if is_solenoidal else 'FAILED'}")
    return mean_div

def check_symplectic():
    print("\n--- 2. NVIDIA-GRADE BRAIN CHECK: Hamiltonian Drift ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    state_dim = 64
    H = HamiltonianNet(state_dim=state_dim, hidden_dim=128).to(device)
    integrator = StormerVerletIntegrator(H, dt=0.01, steps=1).to(device)
    
    # Initial state
    state = torch.randn(1, state_dim).to(device)
    E0 = H(state).item()
    
    # Long-term integration
    steps = 10000
    energies = []
    
    start_time = time.time()
    for i in range(steps):
        state = integrator(state)
        if i % 100 == 0:
            energies.append(H(state).item())
            
    total_time = time.time() - start_time
    
    final_E = H(state).item()
    max_drift = max([abs(e - E0) for e in energies])
    rel_drift = max_drift / (abs(E0) + 1e-8)
    
    print(f"Integration Steps: {steps}")
    print(f"Execution Time:    {total_time:.2f}s")
    print(f"Initial Energy:    {E0:.6f}")
    print(f"Final Energy:      {final_E:.6f}")
    print(f"Max Abs Drift:     {max_drift:.2e}")
    print(f"Max Rel Drift:     {rel_drift:.2e}")
    
    # Symplectic Check (should be O(h^2) and bounded)
    is_stable = rel_drift < 0.01 # Relaxed for random untrained Hamiltonian
    print(f"Result: {'PASSED' if is_stable else 'FAILED'}")
    return rel_drift

if __name__ == "__main__":
    m_div = check_helmholtz()
    s_drift = check_symplectic()
    
    print("\n--- FINAL VERDICT ---")
    print(f"Helmholtz Solenoidality: {m_div:.2e}")
    print(f"Symplectic Energy Drift: {s_drift:.2e}")
