#!/bin/bash

# AI Trading System Setup Script for Dell i7
echo "🚀 Setting up AI Trading System on Dell i7..."

# Check system resources
echo "🔍 Checking system specifications..."
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
echo "Total RAM: ${TOTAL_MEM}GB"
CPU_CORES=$(nproc)
echo "CPU Cores: ${CPU_CORES}"

# Create virtual environment
echo "📦 Creating Python environment..."
python3.9 -m venv trading_ai
source trading_ai/bin/activate

# Install optimized PyTorch for CPU
echo "🔧 Installing optimized PyTorch for CPU..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p data/cache models/trained_models logs signals

# Download pre-trained models
echo "🤖 Downloading pre-trained models..."
python -c "from models.neural_networks.transformer_model import download_pretrained; download_pretrained()"

# Setup database
echo "💾 Setting up database..."
python -c "from data.database.schemas import init_db; init_db()"

# Test the system
echo "🧪 Running tests..."
pytest tests/ -v

echo "✅ Setup complete! Activate with: source trading_ai/bin/activate"
echo "🎯 Start with: python main.py --config environments/local_i7_setup.yaml"
