#!/bin/bash
set -e

echo "╔══════════════════════════════════════╗"
echo "║        PROMPT FORGE INSTALLER        ║"
echo "║  Type what you want → perfect image  ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Install from python.org"
    exit 1
fi

PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python $PYVER"

# Create venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "✅ Virtual environment ready"

# Install dependencies
echo ""
echo "Installing dependencies (this may take 5-10 minutes)..."
pip install -q --upgrade pip

# PyTorch (CPU by default, GPU users can reinstall with CUDA)
echo "  Installing PyTorch..."
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || \
pip install -q torch torchvision

# Core dependencies
echo "  Installing core packages..."
pip install -q -r requirements.txt

# Florence-2 specific versions
echo "  Installing Florence-2 dependencies..."
pip install -q "transformers>=4.40,<4.50" "numpy<2" einops timm

echo "✅ Dependencies installed"

# Check ComfyUI
echo ""
echo "═══════════════════════════════════════"
echo "COMFYUI SETUP"
echo "═══════════════════════════════════════"
echo ""
echo "Prompt Forge needs ComfyUI running for image generation."
echo ""

read -p "Is ComfyUI already running? [y/N] " comfyui_running
if [[ "$comfyui_running" =~ ^[Yy]$ ]]; then
    read -p "ComfyUI URL [http://localhost:8188]: " comfyui_url
    comfyui_url=${comfyui_url:-http://localhost:8188}
else
    echo ""
    echo "Install ComfyUI:"
    echo "  1. git clone https://github.com/comfyanonymous/ComfyUI"
    echo "  2. cd ComfyUI"
    echo "  3. pip install -r requirements.txt"
    echo "  4. python main.py --listen 0.0.0.0"
    echo ""
    echo "Download these models into ComfyUI/models/:"
    echo "  checkpoints/ideogram4_fp8_scaled.safetensors"
    echo "  clip/qwen3vl_8b_fp8_scaled.safetensors"
    echo "  vae/flux2-vae.safetensors"
    echo ""
    comfyui_url="http://localhost:8188"
fi

# Write .env
cat > .env << EOF
# ComfyUI
COMFYUI_URL=$comfyui_url

# Models
FLORENCE_MODEL=microsoft/Florence-2-base-ft
DEFAULT_CHECKPOINT=ideogram4_fp8_scaled.safetensors
DEFAULT_VAE=flux2-vae.safetensors

# Forge
MAX_ITERATIONS=5
CONVERGENCE_THRESHOLD=0.55
FORGE_PORT=7861

# Database
DB_PATH=data/library/compositions.db
EOF

echo "✅ Configuration saved to .env"

# Create directories
mkdir -p data/outputs data/library

echo ""
echo "═══════════════════════════════════════"
echo "  READY!"
echo "═══════════════════════════════════════"
echo ""
echo "Start Prompt Forge:"
echo ""
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "  venv\\Scripts\\activate"
else
    echo "  source venv/bin/activate"
fi
echo "  python -m uvicorn server.main:app --host 0.0.0.0 --port 7861"
echo ""
echo "Then open http://localhost:7861"
echo ""

read -p "Start now? [Y/n] " start_now
if [[ ! "$start_now" =~ ^[Nn]$ ]]; then
    echo ""
    echo "Starting Prompt Forge..."
    python -m uvicorn server.main:app --host 0.0.0.0 --port 7861
fi
