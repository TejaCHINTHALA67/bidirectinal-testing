#!/bin/bash
# ==============================================================================
# Bidirectional RAG: Cloud Experiment Setup Script
# IEEE Access Publication - Full Setup
# ==============================================================================

set -e  # Exit on error

echo "=============================================="
echo "Bidirectional RAG: Cloud Experiment Setup"
echo "=============================================="

# 1. System Updates
echo "[1/8] Updating system packages..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git wget curl unzip

# 2. Install NVIDIA drivers (if GPU available)
echo "[2/8] Checking for GPU..."
if lspci | grep -i nvidia; then
    echo "NVIDIA GPU detected. Installing drivers..."
    sudo apt install -y nvidia-driver-535 nvidia-cuda-toolkit
    nvidia-smi
else
    echo "No NVIDIA GPU detected. Running on CPU only."
fi

# 3. Install Ollama
echo "[3/8] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh
sleep 5

# 4. Start Ollama and pull model
echo "[4/8] Starting Ollama and pulling llama3.2:3b..."
ollama serve &
sleep 10
ollama pull llama3.2:3b

# 5. Clone repository (or use existing)
echo "[5/8] Setting up repository..."
if [ -d "bidirectional-rag-research" ]; then
    echo "Repository already exists. Pulling latest..."
    cd bidirectional-rag-research
    git pull
else
    echo "Cloning repository..."
    git clone https://github.com/TejaCHINTHALA67/bidirectional-rag.git bidirectional-rag-research
    cd bidirectional-rag-research
fi

# 6. Create virtual environment and install dependencies
echo "[6/8] Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 7. Pre-cache corpus
echo "[7/8] Pre-caching Wikipedia corpus..."
python scripts/precache_corpus.py

# 8. Verify setup
echo "[8/8] Verifying setup..."
python -c "import chromadb; import sentence_transformers; print('ChromaDB:', chromadb.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "To run experiments:"
echo "  source venv/bin/activate"
echo "  python main.py --corpus_type realistic --systems all --seeds 42 43 44 45 46 47 48 49 50 51 --num_queries 1000 --max_workers 4"
echo ""
echo "Estimated runtime: 24-48 hours for full experiments"
echo "=============================================="
