"""
ATOM Challenge: Active Flow Control (Cylinder)
==============================================
Scenario: Mitigating Von Kármán Vortex Shedding at Re=1000.
Mission: Stabilize unsteady wake using active jet control.
Goal: Discover the symbolic relationship between control action and flow stability.
"""

import os
import time
import numpy as np
import torch
import json
import cv2
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

from atom.config import AtomConfig, reload_config
from atom.sim.training_loop import AtomOrchestrator
from atom.logging import setup_logging, get_logger
from atom.visualization import get_visualizer

# =============================================================================
# CONFIGURATION
# =============================================================================

def get_cylinder_config(max_steps: int = 1000, headless: bool = False) -> AtomConfig:
    """Create configuration for cylinder flow control challenge."""
    return AtomConfig(
        experiment_name="active_flow_control_challenge",
        seed=42,
        hardware={"device": "auto", "enable_x64_precision": True}, # Auto-detects CUDA/MPS/CPU
        physics={
            "grid_shape": [64, 32, 24], 
            "reynolds_number": 1000.0,
            "world_type": "cylinder"
        },
        brain={
            "vision_dim": 256,
            "internal_neurons": 64,
            "batch_size": 8,
            "learning_rate_actor": 1e-4,
            "use_symplectic": True
        },
        eyes={
            "fno_modes": 8,
            "fno_width": 20
        },
        memory={
            "sequence_length": 4,
            "capacity": 500
        },
        training={
            "max_steps": max_steps,
            "sleep_interval": 100
        }
    )

# =============================================================================
# RUNNER
# =============================================================================

def run_challenge(max_steps: int = 1000, headless: bool = False):
    """Execute the Cylinder Flow Control Challenge."""
    
    config = get_cylinder_config(max_steps, headless)
    setup_logging(config)
    log = get_logger("challenge")

    print("\n🏁 ATOM Aerodynamic Challenge: Active Flow Control")
    print("="*60)
    print("Scenario: Mitigating Von Kármán Vortex Shedding at Re=1000")
    print("Control: Trailing-edge synthetic jets")
    print("Objective: STABILIZE UNSTEADY LIFT\n")

    # 1. Initialize Orchestrator
    orchestrator = AtomOrchestrator(config)
    visualizer = get_visualizer(output_dir=Path("challenge_results"))
    
    # 2. Demonstration Loop
    history = {"reward": [], "lift": [], "stress": [], "theory_score": []}
    
    print(f"🚀 Launching ATOM vs. Chaos (Grid: {config.physics.grid_shape})...")
    
    obs, mask = orchestrator.world.reset()
    hx = None
    last_action = torch.zeros(1, 1).to(config.get_device())
    
    try:
        for step in range(config.training.max_steps):
            start_t = time.time()
            
            # --- WAKE PHASE: INTERACTION ---
            obs_t = orchestrator._tensorify(obs)
            
            # 1. Scientist Intuition (Symbolic Law)
            theory_intuition, theory_conf = orchestrator.scientist.predict_theory(last_action)
            
            # Ensure Tensors (B, 1)
            theory_intuition = torch.as_tensor(theory_intuition, dtype=torch.float32, device=config.get_device()).view(obs_t.shape[0], 1)
            theory_conf = torch.as_tensor(theory_conf, dtype=torch.float32, device=config.get_device()).view(obs_t.shape[0], 1)
            
            # 2. Brain Forward
            (mu, std), value, hx_new, stress = orchestrator.brain(
                obs_t, theory_intuition, last_action, hx, theory_confidence=theory_conf
            )
            
            # 3. Action Selection
            dist = torch.distributions.Normal(mu, std)
            action_phys = torch.clamp(dist.sample(), -1.0, 1.0)
            
            # 4. Environment Step
            next_obs, reward, done, info = orchestrator.world.step(action_phys.detach().cpu().numpy().flatten())
            lift = info.get("lift", 0.0)
            
            # --- RECORD JOURNEY ---
            history["reward"].append(reward)
            history["lift"].append(lift)
            history["stress"].append(stress.mean().item())
            history["theory_score"].append(orchestrator.scientist.best_law_score if orchestrator.scientist.best_law_score != float('inf') else 1.0)
            
            # 5. Fast Learning
            orchestrator.memory.push(obs_t, action_phys, reward, done, hx_new)
            
            # 6. Feed Scientist
            with torch.no_grad():
                visual_embed = orchestrator.brain.eyes.embed(obs_t)
                speed_stats = torch.tensor([torch.norm(obs_t[:, :3], dim=1).mean(), torch.norm(obs_t[:, :3], dim=1).std()], device=obs_t.device)
                features = torch.cat([action_phys.flatten(), speed_stats, visual_embed[0, :8]])
                orchestrator.scientist.observe(features.cpu().numpy(), lift)
                
            # --- BRAIN OPTIMIZATION ---
            if orchestrator.memory.size >= config.brain.batch_size + config.memory.sequence_length:
                _ = orchestrator._update_brain()
            
            # --- SLEEP PHASE ---
            if step > 0 and step % config.training.sleep_interval == 0:
                print(f"\n💤 Step {step}: Scientist Pondering Experience (Law Discovery)...")
                if orchestrator.scientist.best_law:
                    print(f"   📜 Current Best Flow Law: {orchestrator.scientist.best_law}")
                else:
                    print("   🔍 Still searching for the underlying physics...")
                
            # Logging
            dt = time.time() - start_t
            if step % 10 == 0:
                print(f"📍 Step {step:3d} | Reward: {reward:8.2f} | Lift: {lift:8.2f} | Stress: {stress.mean().item():.4f} | {1/dt:.1f} FPS")

            # --- LIVE VIEW ---
            if not headless:
                try:
                    # Quick visualization (Mid-Z slice magnitude)
                    u_mag = np.sqrt(obs[0]**2 + obs[1]**2 + obs[2]**2)
                    slice_2d = u_mag[:, :, u_mag.shape[2]//2]
                    disp = (slice_2d.T * 255 / (np.max(slice_2d) + 1e-6)).astype(np.uint8)
                    disp = cv2.applyColorMap(disp, cv2.COLORMAP_MAGMA)
                    disp = cv2.resize(disp, (640, 320), interpolation=cv2.INTER_LINEAR)
                    
                    cv2.putText(disp, f"Step: {step} | Reward: {reward:.2f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(disp, f"Law: {orchestrator.scientist.best_law[:40] if orchestrator.scientist.best_law else 'Searching...'}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    cv2.imshow("ATOM: Active Flow Control (Live)", disp)
                    cv2.waitKey(1)
                except Exception as e:
                    pass

            # Advance state
            obs = next_obs
            hx = hx_new.detach()
            last_action = action_phys
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        if not headless:
            cv2.destroyAllWindows()
            
    print("\n🏁 Challenge Finished!")
    
    # 3. Final Analysis
    final_dir = Path("challenge_results")
    final_dir.mkdir(exist_ok=True)
    
    visualizer.visualize_training_progress(history, "challenge_learning_curve.png")
    
    report = {
        "final_reward": np.mean(history["reward"][-10:]) if history["reward"] else 0.0,
        "lift_stabilization": (np.std(history["lift"][-10:]) / (np.std(history["lift"][:10]) + 1e-6)) if history["lift"] else 1.0,
        "best_theory": orchestrator.scientist.get_best_theory(),
        "total_steps": config.training.max_steps
    }
    
    with open(final_dir / "challenge_audit.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"\n📊 Challenge Report Saved to {final_dir}/")
    print(f"   Lift Variance Reduction: {100*(1-report['lift_stabilization']):.1f}%")
    print(f"   Final Scientific Verdict: {report['best_theory']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    run_challenge(args.steps, args.headless)
