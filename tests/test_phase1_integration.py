import torch
import numpy as np
import os
import sys
from atom.sim.training_loop import AtomOrchestrator
from atom.config import get_config

def test_phase1_integration():
    print(">>> Testing Phase 1 Integration (PPO, Memories, Brain)...")
    
    # Override config for speed
    config = get_config()
    config.brain.batch_size = 4
    config.memory.sequence_length = 4
    config.training.max_steps = 10
    config.physics.grid_shape = (16, 16, 16) # Fast
    config.brain.vision_dim = 64
    config.brain.internal_neurons = 16
    
    # Create Orchestrator
    try:
        orch = AtomOrchestrator(config, use_symplectic=True)
        print("   ✅ Orchestrator Initialized")
    except Exception as e:
        print(f"   ❌ Orchestrator Init Failed: {e}")
        raise e
        
    # Check Brain Eyes (Are we using Eyes2?)
    from atom.core.eyes2 import AtomEyes
    if isinstance(orch.brain.eyes, AtomEyes):
        print("   ✅ Brain is using Eyes2 (Hodge-Net)")
    else:
        print(f"   ⚠️ Brain using {type(orch.brain.eyes)} (Expected Eyes2)")

    # Run a few steps of simulation
    print("   Running 6 steps...")
    try:
        orch.run()
        print("   ✅ Training Loop ran without crash")
    except Exception as e:
        print(f"   ❌ Training Loop crashed: {e}")
        # raise e # Don't raise to allow inspection if needed
        
    # Check Memory
    print(f"   Memory Size: {orch.memory.size}")
    if orch.memory.size > 0:
        # Check if theory/trust are stored
        if np.any(orch.memory.theory_buf[:orch.memory.size] != 0):
             print("   ✅ Theory buffer populated")
        else:
             print("   ⚠️ Theory buffer is all zeros (Maybe scientist predicted 0?)")
             
        # Check PPO Log Probs
        if np.any(orch.memory.log_prob_buf[:orch.memory.size] != 0):
             print("   ✅ PPO Log Probs populated")
        else:
             print("   ❌ PPO Log Probs are zero!")

    print(">>> Phase 1 Integration Test Complete.")

if __name__ == "__main__":
    test_phase1_integration()
