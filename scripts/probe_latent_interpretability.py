#!/usr/bin/env python3
"""Probe Eyes2 latent saliency against a selected latent index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from atom.config import AtomConfig
from atom.sim.training_loop import AtomOrchestrator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Eyes2 latent saliency.")
    parser.add_argument("--latent-index", type=int, default=2, help="Latent index to probe (>=0).")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Simulation warmup steps.")
    parser.add_argument(
        "--output",
        type=str,
        default="challenge_results/latent_probe.png",
        help="Output image path.",
    )
    return parser.parse_args()


def probe(latent_index: int, warmup_steps: int, output_path: Path) -> Path:
    if latent_index < 0:
        raise ValueError("latent_index must be non-negative")
    if warmup_steps < 1:
        raise ValueError("warmup_steps must be >= 1")

    config = AtomConfig(
        experiment_name=f"interpretability_probe_latent_{latent_index}",
        seed=42,
        physics={
            "grid_shape": [64, 32, 24],
            "world_type": "cylinder",
            "reynolds_number": 1000.0,
        },
    )
    orchestrator = AtomOrchestrator(config)
    eyes = orchestrator.brain.eyes
    world = orchestrator.world
    eyes.eval()

    obs, _ = world.reset()
    for _ in range(int(warmup_steps)):
        obs, _, _, _ = world.step(np.zeros(1, dtype=np.float32))

    obs_t = orchestrator._tensorify(obs).clone().detach().requires_grad_(True)
    latent = eyes.embed(obs_t)
    latent_dim = int(latent.shape[-1])
    if latent_index >= latent_dim:
        raise ValueError(
            f"latent_index={latent_index} out of range for latent_dim={latent_dim}"
        )

    target = torch.mean(latent[:, latent_index])
    target.backward()
    grad = obs_t.grad
    if grad is None:
        raise RuntimeError("Gradient is None; unable to compute saliency map.")

    saliency = grad.detach().abs().sum(dim=1)[0].cpu().numpy()
    slice_idx = saliency.shape[2] // 2

    u_mag = np.sqrt(obs[0, 0] ** 2 + obs[0, 1] ** 2 + obs[0, 2] ** 2)
    s_slice = saliency[:, :, slice_idx].T
    s_slice = (s_slice - s_slice.min()) / (s_slice.max() - s_slice.min() + 1e-8)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    im1 = axes[0].imshow(
        u_mag[:, :, slice_idx].T,
        origin="lower",
        cmap="magma",
        extent=[0, u_mag.shape[0], 0, u_mag.shape[1]],
    )
    axes[0].set_title("Velocity Magnitude Reference", fontsize=13, fontweight="bold")
    plt.colorbar(im1, ax=axes[0], label="velocity magnitude")

    im2 = axes[1].imshow(
        s_slice,
        origin="lower",
        cmap="inferno",
        extent=[0, s_slice.shape[1], 0, s_slice.shape[0]],
    )
    axes[1].set_title(
        f"Eyes2 Gradient Saliency for latent_{latent_index}",
        fontsize=13,
        fontweight="bold",
    )
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    plt.colorbar(im2, ax=axes[1], label="normalized gradient importance")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    args = _parse_args()
    out = probe(
        latent_index=int(args.latent_index),
        warmup_steps=int(args.warmup_steps),
        output_path=Path(args.output),
    )
    print(f"saved_probe={out}")


if __name__ == "__main__":
    main()
