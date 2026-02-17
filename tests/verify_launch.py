#!/usr/bin/env python3
"""
ATOM End-to-End Verification Suite (NVIDIA Grade Audit)
=======================================================
Simulates the experience of a new developer onboarding to ATOM.
Verifies:
1. Environment Setup & Importability
2. Unified CLI Functionality (Supersonic & Cylinder)
3. Hardware Agnosticism (Auto-detection)
4. Repository Hygiene (No loose scripts)
5. Documentation Integrity (Whitepaper images)

Usage:
    python3 tests/verify_launch.py
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# Colors for professional output
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def log(msg, status="INFO"):
    symbol = "ℹ️"
    color = BLUE
    if status == "SUCCESS":
        symbol = "✅"
        color = GREEN
    elif status == "ERROR":
        symbol = "❌"
        color = RED
    elif status == "WARN":
        symbol = "⚠️"
        color = YELLOW
    print(f"{color}{symbol}  {msg}{RESET}")

def run_command(cmd, cwd=None, env=None):
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            env=env,
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def check_structure():
    log("Checking Repository Structure...", "INFO")
    
    root = Path(".")
    required = [
        "src/atom/challenges/supersonic.py",
        "src/atom/challenges/cylinder.py",
        "install.sh",
        "docs/ATOM_Whitepaper.md",
        "docs/assets"
    ]
    
    forbidden = [
        "scripts/atom_challenge.py",
        "examples/discover_3d.py"
    ]
    
    all_passed = True
    for p in required:
        if not (root / p).exists():
            log(f"Missing required file: {p}", "ERROR")
            all_passed = False
            
    for p in forbidden:
        if (root / p).exists():
            log(f"Found forbidden legacy file (should be cleaned): {p}", "ERROR")
            all_passed = False
            
    if all_passed:
        log("Repository structure is clean and standarized.", "SUCCESS")
    return all_passed

def check_cli_run(challenge_name):
    log(f"Verifying 'atom run {challenge_name}' (Headless)...", "INFO")
    
    # We must ensure PYTHONPATH includes src since we aren't pip installed in this test env
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    
    # Run for just 10 steps to verify pipeline works
    cmd = [get_python_exe(), "-m", "atom.cli", "run", challenge_name, "--steps", "10", "--headless"]
    
    success, output = run_command(cmd, env=env)
    
    if success:
        log(f"{challenge_name} challenge executed successfully.", "SUCCESS")
        # Check for results artifact
        if (Path("challenge_results/challenge_audit.json")).exists():
            log(f"Results artifact generated for {challenge_name}.", "SUCCESS")
        else:
            log(f"No results found for {challenge_name}!", "WARN")
        return True
    else:
        log(f"{challenge_name} failed!\nError:\n{output}", "ERROR")
        return False

def check_hardware_config():
    log("Verifying Hardware Agnosticism...", "INFO")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    
    cmd = [get_python_exe(), "-c", "from atom.config import config; print(f'PLATFORM: {config.get_device()}')"]
    success, output = run_command(cmd, env=env)
    
    if success:
        device = output.strip().split(": ")[1]
        log(f"Auto-detected Hardware: {device}", "SUCCESS")
        if device == "cpu" and TORCH_AVAILABLE and torch.cuda.is_available():
             log("Warning: CUDA verified available but config chose CPU!", "WARN")
        return True
    else:
        log("Failed to load config!", "ERROR")
        return False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
def get_python_exe():
    # Prefer venv if valid
    venv_python = Path("venv/bin/python")
    if venv_python.exists() and os.access(venv_python, os.X_OK):
        return str(venv_python)
    return sys.executable


def main():
    print(f"\n{BLUE}======================================================{RESET}")
    print(f"{BLUE}      ATOM OPEN SOURCE READINESS AUDIT (E2E)          {RESET}")
    print(f"{BLUE}======================================================{RESET}\n")
    
    results = []
    
    # 1. Structure
    results.append(check_structure())
    
    # 2. Hardware
    results.append(check_hardware_config())
    
    # 3. Supersonic Run
    results.append(check_cli_run("supersonic"))
    
    # 4. Cylinder Run
    results.append(check_cli_run("cylinder"))
    
    print(f"\n{BLUE}======================================================{RESET}")
    if all(results):
        print(f"{GREEN}PASSED: System is NVIDIA-Grade Ready for Launch. 🚀{RESET}")
        sys.exit(0)
    else:
        print(f"{RED}FAILED: Issues detected. See logs above.{RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()
