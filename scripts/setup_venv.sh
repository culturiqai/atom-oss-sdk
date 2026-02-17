#!/bin/bash

# ATOM Virtual Environment Setup Script
# Optimized for MacBook Pro development and testing

set -e

echo "🚀 Setting up ATOM Virtual Environment for MacBook Pro"
echo "======================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${BLUE}Checking Python version...${NC}"
python3 --version

# Create virtual environment
echo -e "${BLUE}Creating virtual environment...${NC}"
python3 -m venv venv

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip

# Install core dependencies
echo -e "${BLUE}Installing core dependencies...${NC}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy matplotlib plotly pandas h5py

# Install ML/AI dependencies
echo -e "${BLUE}Installing ML/AI dependencies...${NC}"
pip install jax jaxlib
pip install sympy

# Install development dependencies
echo -e "${BLUE}Installing development dependencies...${NC}"
pip install pytest pytest-cov pytest-xdist pytest-benchmark
pip install black isort flake8 mypy pre-commit
pip install jupyter ipykernel

# Install project in development mode
echo -e "${BLUE}Installing ATOM in development mode...${NC}"
pip install -e ".[dev]"

# Optional: Install symbolic regression (may fail on some systems)
echo -e "${YELLOW}Attempting to install PySR (symbolic regression)...${NC}"
pip install pysr 2>/dev/null || echo -e "${RED}PySR installation failed - symbolic reasoning will be mocked${NC}"

# Optional: Install physics simulation (may fail on some systems)
echo -e "${YELLOW}Attempting to install XLB (physics simulation)...${NC}"
pip install xlb trimesh 2>/dev/null || echo -e "${RED}XLB installation failed - physics simulation will be mocked${NC}"

# Optional: Install neural circuit policies
echo -e "${YELLOW}Attempting to install NCPS (neural circuits)...${NC}"
pip install ncps 2>/dev/null || echo -e "${RED}NCPS installation failed - LTC networks will use GRU fallback${NC}"

# Verify installation
echo -e "${BLUE}Verifying installation...${NC}"
python -c "
import sys
print(f'Python version: {sys.version}')
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
except ImportError:
    print('PyTorch not available')

try:
    import jax
    print(f'JAX version: {jax.__version__}')
except ImportError:
    print('JAX not available')

try:
    import atom
    print(f'ATOM version: {atom.__version__}')
    print('ATOM import successful!')
except ImportError as e:
    print(f'ATOM import failed: {e}')
    exit(1)
"

echo -e "${GREEN}Virtual environment setup complete!${NC}"
echo ""
echo -e "${YELLOW}To activate the environment in new shells:${NC}"
echo "source venv/bin/activate"
echo ""
echo -e "${YELLOW}To run comprehensive tests:${NC}"
echo "python scripts/run_comprehensive_tests.py"
echo ""
echo -e "${YELLOW}To run benchmark:${NC}"
echo "python scripts/benchmark_atom.py"
echo ""
echo -e "${GREEN}Ready for ATOM testing and benchmarking! 🚀${NC}"