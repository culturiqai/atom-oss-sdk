"""
ATOM CORE: EYES (Perception)
----------------------------
3D Fourier Neural Operator (FNO)-style perception backbone + potential-based decoder.

Important reality check (for your paper/docs):
- This module is translation-equivariant (via FFT structure). It is NOT rotation/SE(3)-equivariant.
- "Hodge decomposition" here is a decoder parameterization: v = curl(psi) + beta * grad(phi).
  It is not a true Hodge projection operator applied to an arbitrary input vector field.

I/O contract (preserved):
- AtomEyes.forward(x) expects x shaped (B, 4, X, Y, Z) and returns velocity (B, 3, X, Y, Z)
- AtomEyes.embed(x) returns latent embedding (B, embedding_dim)
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from atom.config import get_config
from atom.exceptions import EyesError
from atom.logging import get_logger

logger = get_logger("eyes")


# --- 1. LOW RANK HYPER-LAYERS ---
class ComplexLowRankLinear(nn.Module):
    """
    Memory-efficient HyperNetwork (Reserved for future use).

    Notes:
    - Currently unused in AtomEyes. Kept for API stability / future wiring.
    - Produces a real-valued tensor reshaped to out_shape.
    """
    def __init__(self, in_dim: int, out_shape: Tuple[int, ...]):
        super().__init__()
        self.out_shape = out_shape
        self.flat_dim = int(np.prod(out_shape))
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, self.flat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.net(x)
        return w.view(-1, *self.out_shape)


# --- 2. 3D SPECTRAL CONVOLUTION ---
class SpectralConv3d(nn.Module):
    """
    3D Fourier Layer: convolution in frequency domain (FNO-style).

    Implementation detail:
    - Uses rfftn over spatial dims.
    - Retains low-frequency modes in z (one-sided because rFFT),
      and retains low-frequency blocks in x/y by gathering:
        [0:mx] and [-mx:] for x, and [0:my] and [-my:] for y
      into a "corner block" (2*mx, 2*my, mz),
      then multiplies by complex weights and scatters back.

    Contract:
    - Input:  (B, in_channels, X, Y, Z) real
    - Output: (B, out_channels, X, Y, Z) real
    """
    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int, modes_z: int):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(f"in_channels/out_channels must be positive, got {in_channels}/{out_channels}")
        if modes_x <= 0 or modes_y <= 0 or modes_z <= 0:
            raise ValueError(f"modes_x/modes_y/modes_z must be positive, got {modes_x}/{modes_y}/{modes_z}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z

        # Complex weights over gathered corner block:
        # (in, out, 2*modes_x, 2*modes_y, modes_z)
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, 2 * modes_x, 2 * modes_y, modes_z, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul3d(inp: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # inp: (B, I, X, Y, Z) complex
        # w:   (I, O, X, Y, Z) complex
        # out: (B, O, X, Y, Z) complex
        return torch.einsum("bixyz,ioxyz->boxyz", inp, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise EyesError(f"SpectralConv3d expected 5D tensor (B,C,X,Y,Z), got shape {tuple(x.shape)}")

        bsz, _, nx, ny, nz = x.shape

        # FFT (real -> complex)
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))  # (B, I, nx, ny, nz_r)
        nz_r = x_ft.size(-1)

        # Output spectrum buffer
        out_ft = torch.zeros(
            bsz,
            self.out_channels,
            nx,
            ny,
            nz_r,
            dtype=x_ft.dtype,
            device=x.device,
        )

        # Modes selected (bounded by grid)
        mx = min(self.modes_x, nx // 2)
        my = min(self.modes_y, ny // 2)
        mz = min(self.modes_z, nz_r)

        # If grid is tiny / degenerate, return zeros cleanly.
        # This avoids the catastrophic "-0:" slicing bug.
        if mx <= 0 or my <= 0 or mz <= 0:
            return torch.fft.irfftn(out_ft, s=(nx, ny, nz), dim=(-3, -2, -1))

        # Dtype matching must actually be used (your old code computed current_weights then ignored it).
        w = self.weights
        if w.dtype != x_ft.dtype:
            # .to keeps device by default; dtype-only cast
            w = w.to(dtype=x_ft.dtype)

        # Gather corners: x in [0:mx] U [-mx:], y in [0:my] U [-my:]
        x_xpos = x_ft[:, :, :mx, :, :mz]
        x_xneg = x_ft[:, :, -mx:, :, :mz]
        x_x = torch.cat([x_xpos, x_xneg], dim=2)  # (B, I, 2*mx, ny, mz)

        x_ypos = x_x[:, :, :, :my, :]
        x_yneg = x_x[:, :, :, -my:, :]
        x_corners = torch.cat([x_ypos, x_yneg], dim=3)  # (B, I, 2*mx, 2*my, mz)

        # Slice weights to actual gathered sizes
        w_sl = w[:, :, : 2 * mx, : 2 * my, :mz]  # (I, O, 2*mx, 2*my, mz)

        out_corners = self.compl_mul3d(x_corners, w_sl)  # (B, O, 2*mx, 2*my, mz)

        # Scatter back into frequency tensor
        # Quadrants in x/y:
        out_ft[:, :, :mx, :my, :mz] = out_corners[:, :, :mx, :my, :]
        out_ft[:, :, :mx, -my:, :mz] = out_corners[:, :, :mx, my:, :]
        out_ft[:, :, -mx:, :my, :mz] = out_corners[:, :, mx:, :my, :]
        out_ft[:, :, -mx:, -my:, :mz] = out_corners[:, :, mx:, my:, :]

        # iFFT (complex -> real)
        x_out = torch.fft.irfftn(out_ft, s=(nx, ny, nz), dim=(-3, -2, -1))
        return x_out


# --- 3. PHYSICS HEADS (HODGE PARAMETERIZATION) ---
class HodgeHead3d(nn.Module):
    """
    Decoder: latent -> velocity via potential parameterization.

    v = curl(psi) + beta(x) * grad(phi)

    - psi: vector potential (3 components) -> curl yields solenoidal component.
    - phi: scalar potential (1 component) -> grad yields irrotational component.
    - beta: gate in [0,1] (1 component) -> spatially blends compressible contribution.

    NOTE:
    - This is not a true Hodge projection of an arbitrary vector field.
    - It is a structured decoder that *can* represent a Hodge-like split.
    """
    def __init__(self, in_width: int):
        super().__init__()
        self.proj_psi = nn.Linear(in_width, 3)
        self.proj_phi = nn.Linear(in_width, 1)
        self.proj_gate = nn.Sequential(
            nn.Linear(in_width, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _apply_nyquist_mask_inplace(
        hats: torch.Tensor, nx: int, ny: int, nz: int
    ) -> torch.Tensor:
        """
        Zero Nyquist planes (when sizes are even) to avoid edge-case symmetry issues.

        hats: (B, C, nx, ny, nz_r) complex
        """
        # nz_r is last dim of rfft output
        # Nyquist indices exist only for even sizes.
        if nx % 2 == 0:
            hats[:, :, nx // 2, :, :] = 0
        if ny % 2 == 0:
            hats[:, :, :, ny // 2, :] = 0
        if nz % 2 == 0:
            # rfft stores Nyquist at last index for the rfft axis
            hats[:, :, :, :, -1] = 0
        return hats

    def forward(self, x: torch.Tensor, grid_h: float = 1.0) -> torch.Tensor:
        """
        Args:
            x: latent tensor (B, X, Y, Z, C)
            grid_h: grid spacing for spectral derivatives
        Returns:
            uvw: velocity field (B, 3, X, Y, Z)
        """
        if x.ndim != 5:
            raise EyesError(f"HodgeHead3d expected (B,X,Y,Z,C), got shape {tuple(x.shape)}")
        if grid_h <= 0:
            raise EyesError(f"grid_h must be positive, got {grid_h}")

        # Project potentials + gate in physical space
        psi = self.proj_psi(x).permute(0, 4, 1, 2, 3)   # (B, 3, nx, ny, nz)
        phi = self.proj_phi(x).permute(0, 4, 1, 2, 3)   # (B, 1, nx, ny, nz)
        beta = self.proj_gate(x).permute(0, 4, 1, 2, 3) # (B, 1, nx, ny, nz)

        bsz, _, nx, ny, nz = psi.shape
        device = psi.device
        real_dtype = psi.dtype

        # Forward FFT
        psi_hat = torch.fft.rfftn(psi, dim=(-3, -2, -1))  # (B, 3, nx, ny, nz_r)
        phi_hat = torch.fft.rfftn(phi, dim=(-3, -2, -1))  # (B, 1, nx, ny, nz_r)

        # Wavenumbers (type-matched)
        kx = torch.fft.fftfreq(nx, d=1.0, device=device).to(real_dtype)
        ky = torch.fft.fftfreq(ny, d=1.0, device=device).to(real_dtype)
        kz = torch.fft.rfftfreq(nz, d=1.0, device=device).to(real_dtype)

        # Scale to physical grid spacing
        k_scale = (2.0 * np.pi) / grid_h
        kx = kx.view(nx, 1, 1) * k_scale
        ky = ky.view(1, ny, 1) * k_scale
        kz = kz.view(1, 1, -1) * k_scale

        # Curl(psi) in Fourier space
        ax_hat, ay_hat, az_hat = psi_hat[:, 0], psi_hat[:, 1], psi_hat[:, 2]
        uc_hat = 1j * (ky * az_hat - kz * ay_hat)
        vc_hat = 1j * (kz * ax_hat - kx * az_hat)
        wc_hat = 1j * (kx * ay_hat - ky * ax_hat)

        # Grad(phi) in Fourier space
        p_hat = phi_hat[:, 0]
        ug_hat = 1j * (kx * p_hat)
        vg_hat = 1j * (ky * p_hat)
        wg_hat = 1j * (kz * p_hat)

        # Stack then mask Nyquist planes (in-place)
        uvw_curl_hat = torch.stack([uc_hat, vc_hat, wc_hat], dim=1)  # (B,3,nx,ny,nz_r)
        uvw_grad_hat = torch.stack([ug_hat, vg_hat, wg_hat], dim=1)  # (B,3,nx,ny,nz_r)

        self._apply_nyquist_mask_inplace(uvw_curl_hat, nx, ny, nz)
        self._apply_nyquist_mask_inplace(uvw_grad_hat, nx, ny, nz)

        # Inverse FFT
        uvw_curl = torch.fft.irfftn(uvw_curl_hat, s=(nx, ny, nz), dim=(-3, -2, -1))
        uvw_grad = torch.fft.irfftn(uvw_grad_hat, s=(nx, ny, nz), dim=(-3, -2, -1))

        # Regime-adaptive combination
        uvw_total = uvw_curl + (beta * uvw_grad)  # (B,3,nx,ny,nz)
        return uvw_total


# --- 4. THE MAIN EYE ---
class AtomEyes(nn.Module):
    """
    FNO-ish perception backbone + HodgeHead3d decoder.

    Expected input:
      x: (B, 4, X, Y, Z)

    Output:
      uvw: (B, 3, X, Y, Z)
    """
    def __init__(self, modes: int = 12, width: int = 32, depth: int = 4, embedding_dim: Optional[int] = None):
        super().__init__()
        if modes <= 0 or width <= 0 or depth <= 0:
            raise ValueError(f"modes/width/depth must be positive, got {modes}/{width}/{depth}")

        self.width = width
        self.modes = modes

        config = get_config()
        self.embedding_dim = embedding_dim if embedding_dim is not None else getattr(config.eyes, "embedding_dim", 256)

        # Input is 4 channels by contract.
        self.lifting = nn.Linear(4, width)

        self.spectral_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(depth):
            self.spectral_layers.append(SpectralConv3d(width, width, modes, modes, modes))
            self.skip_layers.append(nn.Conv3d(width, width, 1))
            self.norms.append(nn.GroupNorm(num_groups=4, num_channels=width))

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.embedding_proj = nn.Linear(width, self.embedding_dim)

        self.dec1 = nn.Linear(width, 128)
        self.dec2 = nn.Linear(128, 128)

        self.head = HodgeHead3d(128)

        logger.info(f"Initialized AtomEyes (Hodge-Net): modes={modes}, width={width}, depth={depth}")

    @staticmethod
    def _validate_input(x: torch.Tensor) -> None:
        if x.ndim != 5:
            raise EyesError(f"AtomEyes expects 5D input (B,4,X,Y,Z). Got shape {tuple(x.shape)}")
        if x.size(1) != 4:
            raise EyesError(f"AtomEyes expects channel dim = 4. Got x.shape[1] = {x.size(1)}")

    def forward(self, x: torch.Tensor, grid_h: float = 1.0) -> torch.Tensor:
        try:
            self._validate_input(x)
            if grid_h <= 0:
                raise EyesError(f"grid_h must be positive, got {grid_h}")

            # Move channels-last for linear lifting over channel dim, then back to channels-first.
            x = x.permute(0, 2, 3, 4, 1)  # (B, X, Y, Z, 4)
            x = self.lifting(x)           # (B, X, Y, Z, width)
            x = x.permute(0, 4, 1, 2, 3)  # (B, width, X, Y, Z)

            # FNO blocks
            for spec, skip, norm in zip(self.spectral_layers, self.skip_layers, self.norms):
                x_spec = spec(x)
                x_skip = skip(x)
                x = F.gelu(x_spec + x_skip)
                x = norm(x)

            # Decode to velocity
            x_out = x.permute(0, 2, 3, 4, 1)  # (B, X, Y, Z, width)
            x_out = F.gelu(self.dec1(x_out))
            x_out = F.gelu(self.dec2(x_out))

            return self.head(x_out, grid_h=grid_h)

        except Exception as e:
            logger.error(f"Error in AtomEyes forward pass: {e}")
            raise EyesError(f"Eyes forward pass failed: {e}") from e

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Produces a global latent embedding (B, embedding_dim) via global average pooling
        over the final backbone feature map.
        """
        try:
            self._validate_input(x)

            x = x.permute(0, 2, 3, 4, 1)  # (B, X, Y, Z, 4)
            x = self.lifting(x)
            x = x.permute(0, 4, 1, 2, 3)  # (B, width, X, Y, Z)

            for spec, skip, norm in zip(self.spectral_layers, self.skip_layers, self.norms):
                x_spec = spec(x)
                x_skip = skip(x)
                x = F.gelu(x_spec + x_skip)
                x = norm(x)

            latent = self.gap(x).view(x.shape[0], -1)  # (B, width)
            latent = self.embedding_proj(latent)       # (B, embedding_dim)
            return latent

        except Exception as e:
            logger.error(f"Error in AtomEyes embed: {e}")
            raise EyesError(f"Eyes embedding failed: {e}") from e


def create_eyes_from_config(config: Any = None) -> AtomEyes:
    if config is None:
        config = get_config()
    return AtomEyes(modes=config.eyes.fno_modes, width=config.eyes.fno_width, depth=config.eyes.fno_depth)


if __name__ == "__main__":
    print(">>> ATOM EYES (HODGE-NET): Initializing (Robust Dtype Mode)...")
    model = AtomEyes(modes=8, width=20)

    # Sanity check in double precision
    dummy = torch.randn(2, 4, 32, 32, 32, dtype=torch.double)
    model = model.double()

    out = model(dummy)

    print(f"Output Shape: {out.shape}")
    print(f"Output Dtype: {out.dtype}")
    print(">>> Hodge Decomposition Integration Successful.")
