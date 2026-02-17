"""
ATOM Challenge: Custom Geometry (Platform Benchmark)
====================================================
The Full Neuro-Symbolic Stack applied to User Geometry.

Scenario: "Virtual Wind Tunnel" Optimization.
Mission: The Brain must learn to steer the inlet flow vector (Angle of Attack) 
         to minimize drag/turbulence on the custom object.
Goal: Discover aerodynamic laws specific to this unique geometry.
"""

import os
import time
import numpy as np
import torch
import json
import cv2
import argparse
from pathlib import Path

from atom.config import AtomConfig
from atom.sim.training_loop import AtomOrchestrator
from atom.logging import setup_logging, get_logger
from atom.visualization import get_visualizer

def run_custom_challenge(stl_path: str, max_steps: int = 1000, headless: bool = False):
    """Execute the Custom Geometry Challenge using the Full ATOM Stack."""
    
    # 1. Setup Config
    config = AtomConfig(
        experiment_name=f"custom_{os.path.basename(stl_path).split('.')[0]}",
        hardware={"device": "auto", "enable_x64_precision": True},
        physics={
            "grid_shape": [128, 64, 64], 
            "reynolds_number": 2000.0,
            "world_type": "custom",
            "geometry_path": stl_path
        },
        brain={
            "vision_dim": 256,
            "action_dim": 2, # Control Vy, Vz (Steering)
            "internal_neurons": 64,
            "batch_size": 8,
            "learning_rate_actor": 1e-4,
        },
        training={
            "max_steps": max_steps,
            "sleep_interval": 50 # Advertise frequent discovery
        }
    )
    
    setup_logging(config)
    log = get_logger("challenge")

    print(f"\n🎨 ATOM Platform Challenge: {os.path.basename(stl_path)}")
    print("="*60)
    print("Scenario: Adaptive Wind Tunnel Optimization")
    print("Brain: Learning optimal flow angles to minimize drag.")
    print("Scientist: Deriving geometry-specific aerodynamic laws.")
    
    # 2. Initialize Orchestrator (The Full Stack)
    try:
        orchestrator = AtomOrchestrator(config)
    except Exception as e:
        print(f"❌ Failed to initialize ATOM Platform: {e}")
        return

    visualizer = get_visualizer(output_dir=Path("challenge_results"))
    
    # 3. Training Loop
    history = {"reward": [], "drag": [], "theory_score": []}
    
    print(f"🚀 Launching Neuro-Symbolic Agent on {config.get_device().upper()}...")
    
    obs, mask = orchestrator.world.reset()
    hx = None
    last_action = torch.zeros(1, 2).to(config.get_device()) # 2D Action
    
    # Video Setup
    video_path = "custom_demo.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None
    
    try:
        for step in range(config.training.max_steps):
            start_t = time.time()
            
            # --- BRAIN CYCLE ---
            obs_t = orchestrator._tensorify(obs)
            
            # 1. Scientist Intuition
            theory_intuition, theory_conf = orchestrator.scientist.predict_theory(last_action)
            
            # Reshape for Brain
            theory_intuition = torch.as_tensor(theory_intuition, dtype=torch.float32, device=config.get_device()).view(obs_t.shape[0], 1)
            theory_conf = torch.as_tensor(theory_conf, dtype=torch.float32, device=config.get_device()).view(obs_t.shape[0], 1)
            
            # 2. Brain Forward
            (mu, std), value, hx_new, stress = orchestrator.brain(
                obs_t, theory_intuition, last_action, hx, theory_confidence=theory_conf
            )
            
            # 3. Action
            dist = torch.distributions.Normal(mu, std)
            action_phys = torch.clamp(dist.sample(), -1.0, 1.0)
            
            # 4. World Step
            next_obs, reward, done, info = orchestrator.world.step(action_phys.detach().cpu().numpy().flatten())
            drag = info.get("drag", 0.0)
            
            # --- MEMORY & LEARNING ---
            orchestrator.memory.push(obs_t, action_phys, reward, done, hx_new)
            
            history["reward"].append(reward)
            history["drag"].append(drag)
            
            # Feed Scientist
            with torch.no_grad():
                visual_embed = orchestrator.brain.eyes.embed(obs_t)
                speed_stats = torch.tensor([torch.norm(obs_t[:, :3], dim=1).mean(), torch.norm(obs_t[:, :3], dim=1).std()], device=obs_t.device)
                features = torch.cat([action_phys.flatten(), speed_stats, visual_embed[0, :8]])
                orchestrator.scientist.observe(features.cpu().numpy(), drag)
                
            # Train Brain
            if orchestrator.memory.size >= config.brain.batch_size + config.memory.sequence_length:
                _ = orchestrator._update_brain()
                
            # Sleep/Dream
            if step > 0 and step % config.training.sleep_interval == 0:
                print(f"\n💤 Step {step}: Scientist Pondering...")
                if orchestrator.scientist.best_law:
                    print(f"   📜 Discovered Law: {orchestrator.scientist.best_law}")

            # --- VISUALIZATION ---
            dt = time.time() - start_t
            if step % 10 == 0:
                 print(f"📍 Step {step:3d} | Reward: {reward:8.2f} | Action: {action_phys[0].cpu().numpy()} | {1/dt:.1f} FPS")
            
            # Live View / Video
            # Ensure obs is pure numpy first
            obs_np = np.array(obs) 
            # Obs is (Batch, Channel, X, Y, Z). Batch=1.
            # Channels: 0=ux, 1=uy, 2=uz, 3=rho
            # We want magnitude u = sqrt(ux^2 + uy^2 + uz^2)
            u_mag = np.sqrt(obs_np[0, 0]**2 + obs_np[0, 1]**2 + obs_np[0, 2]**2)
            
            slice_2d = u_mag[:, :, u_mag.shape[2]//2]
            
            # Normalize to 0-255
            norm_factor = np.max(slice_2d) + 1e-6
            disp = (slice_2d.T * 255.0 / norm_factor).astype(np.uint8)
            
            # Enforce contiguous array for OpenCV
            disp = np.ascontiguousarray(disp)
            
            # Debug shape if needed (commented out)
            # print(f"DEBUG: disp shape {disp.shape}, dtype {disp.dtype}")
            
            disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
            disp = cv2.resize(disp, (800, 400), interpolation=cv2.INTER_LINEAR)
            
            # Overlay
            cv2.putText(disp, f"Geo: {os.path.basename(stl_path)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            law_text = orchestrator.scientist.best_law[:50] if orchestrator.scientist.best_law else "Hypothesizing..."
            cv2.putText(disp, f"Law: {law_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            
            if video_writer is None:
                h, w = disp.shape[:2]
                video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
            video_writer.write(disp)
            
            if not headless:
                try:
                    cv2.imshow("ATOM Platform", disp)
                    cv2.waitKey(1)
                except:
                    pass
            
            # Advance
            obs = next_obs
            hx = hx_new.detach()
            last_action = action_phys
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    finally:
        if not headless:
            cv2.destroyAllWindows()
        if video_writer:
            video_writer.release()
            print(f"🎥 Video saved: {video_path}")
            
    print("\n🏁 Custom Challenge Complete.")
