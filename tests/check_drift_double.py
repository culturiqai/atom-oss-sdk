import torch
import numpy as np
import time
from atom.core.symplectic import HamiltonianNet, StormerVerletIntegrator

def check_symplectic_double():
    print("--- DOUBLE PRECISION BRAIN CHECK: Hamiltonian Drift ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use float64 for machine-precision check
    dtype = torch.float64
    state_dim = 16 # Reduced for focused check
    
    H = HamiltonianNet(state_dim=state_dim, hidden_dim=64).to(device).to(dtype)
    integrator = StormerVerletIntegrator(H, dt=0.001, steps=1).to(device).to(dtype)
    
    # Initial state
    state = torch.randn(1, state_dim, device=device, dtype=dtype)
    E0 = H(state).item()
    
    # Long-term integration
    steps = 1000
    energies = []
    
    for i in range(steps):
        state = integrator(state)
        # Check every step for drift
        energies.append(H(state).item())
            
    final_E = H(state).item()
    max_drift = max([abs(e - E0) for e in energies])
    rel_drift = max_drift / (abs(E0) + 1e-15)
    
    print(f"DType:             {dtype}")
    print(f"Steps:             {steps}")
    print(f"Max Abs Drift:     {max_drift:.2e}")
    print(f"Max Rel Drift:     {rel_drift:.2e}")
    
    return max_drift

if __name__ == "__main__":
    check_symplectic_double()
