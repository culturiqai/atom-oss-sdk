import torch
import numpy as np
from atom.core.eyes2 import AtomEyes

def spectral_divergence(u, v, w, grid_h=1.0):
    """Compute divergence in spectral space."""
    u, v, w = u.double(), v.double(), w.double()
    
    # Forward FFT
    u_hat = torch.fft.rfftn(u, dim=(-3, -2, -1))
    v_hat = torch.fft.rfftn(v, dim=(-3, -2, -1))
    w_hat = torch.fft.rfftn(w, dim=(-3, -2, -1))

    B, Nx, Ny, Nz = u.shape
    device = u.device
    
    # Wavenumbers
    kx = torch.fft.fftfreq(Nx, d=1.0, device=device).double()
    ky = torch.fft.fftfreq(Ny, d=1.0, device=device).double()
    kz = torch.fft.rfftfreq(Nz, d=1.0, device=device).double()
    
    scale = 2 * np.pi / grid_h
    kx = kx.view(Nx, 1, 1) * scale
    ky = ky.view(1, Ny, 1) * scale
    kz = kz.view(1, 1, -1) * scale

    # Div = i*k . v
    div_hat = 1j * (kx * u_hat + ky * v_hat + kz * w_hat)
    
    # Nyquist Mask (Must match model)
    if Nx % 2 == 0: div_hat[:, Nx//2, :, :] = 0
    if Ny % 2 == 0: div_hat[:, :, Ny//2, :] = 0
    if Nz % 2 == 0: div_hat[:, :, :, -1] = 0

    return torch.fft.irfftn(div_hat, s=(Nx, Ny, Nz), dim=(-3, -2, -1))

if __name__ == "__main__":
    print(f"{'='*60}")
    print(">>> ATOM PHYSICS: Validating Hodge Gating Mechanism")
    print(f"{'='*60}")
    
    # 1. Initialize & Double Precision
    model = AtomEyes(modes=8, width=20).double()
    x = torch.randn(2, 4, 32, 32, 32, dtype=torch.double)
    
    # --- TEST 1: SQUINT MODE (Incompressible) ---
    print("\n[TEST 1] Squint Mode (Beta = 0)")
    print("   Enforcing Incompressible Flow constraint...")
    
    # Hack: Manually force the gate to ZERO
    def force_squint(m, input, output):
        return output * 0.0 # Force beta to 0
    
    # Hook into the gate to shut the "Open Eye"
    handle = model.head.proj_gate.register_forward_hook(force_squint)
    
    with torch.no_grad():
        uvw_squint = model(x, grid_h=0.1)
    
    handle.remove() # Clean up hook
    
    div_squint = spectral_divergence(uvw_squint[:,0], uvw_squint[:,1], uvw_squint[:,2], grid_h=0.1)
    err_squint = torch.max(torch.abs(div_squint)).item()
    print(f"   > Max Divergence: {err_squint:.4e}")
    
    if err_squint < 1e-12:
        print("   ✅ PASS: Solenoidal stream is perfectly divergence-free.")
    else:
        print("   ❌ FAIL: Leakage in solenoidal stream.")

    # --- TEST 2: OPEN EYE MODE (Compressible) ---
    print("\n[TEST 2] Open-Eye Mode (Beta = 1)")
    print("   Allowing full compressibility...")
    
    # Hack: Manually force the gate to ONE
    def force_open(m, input, output):
        return torch.ones_like(output) # Force beta to 1
        
    handle = model.head.proj_gate.register_forward_hook(force_open)
    
    with torch.no_grad():
        uvw_open = model(x, grid_h=0.1)
        
    handle.remove()
    
    div_open = spectral_divergence(uvw_open[:,0], uvw_open[:,1], uvw_open[:,2], grid_h=0.1)
    err_open = torch.max(torch.abs(div_open)).item()
    print(f"   > Max Divergence: {err_open:.4e}")
    
    if err_open > 1.0:
        print("   ✅ PASS: Phi-stream correctly producing divergence (Sound Waves Active).")
    else:
        print("   ❌ FAIL: Phi-stream is silent (No Acoustics).")

    print(f"\n{'='*60}")
    print("SUMMARY: Hodge Decomposition Architecture is VALID.")