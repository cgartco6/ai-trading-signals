#!/bin/bash

# AI Trading System Setup Script for Dell i7
echo "🚀 Setting up AI Trading System on Dell i7..."

# Check system resources
echo "🔍 Checking system specifications..."
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
echo "Total RAM: ${TOTAL_MEM}GB"
CPU_CORES=$(nproc)
echo "CPU Cores: ${CPU_CORES}"

# Verify Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: ${PYTHON_VERSION}"

if [ $(echo "$PYTHON_VERSION < 3.8" | bc -l) -eq 1 ]; then
    echo "❌ Python 3.8 or higher required"
    exit 1
fi

# Create virtual environment
echo "📦 Creating Python environment..."
python3 -m venv trading_ai
source trading_ai/bin/activate

# Install optimized PyTorch for CPU
echo "🔧 Installing optimized PyTorch for CPU..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies
echo "🔧 Installing development dependencies..."
pip install pytest pytest-cov black flake8 mypy

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p data/cache models/trained_models logs signals tests/fixtures

# Download pre-trained models (if available)
echo "🤖 Setting up AI models..."
python -c "
try:
    from models.neural_networks.transformer_model import TransformerTradingModel
    from models.neural_networks.lstm_attention import LSTMAttentionModel
    print('AI models initialized successfully')
except Exception as e:
    print(f'Model setup: {e}')
"

# Setup database
echo "💾 Setting up database..."
python -c "
try:
    from data.database.schemas import init_db
    init_db()
    print('Database initialized successfully')
except Exception as e:
    print(f'Database setup: {e}')
"

# Create environment file
echo "🔧 Creating environment configuration..."
cp .env.example .env

# Set up pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pip install pre-commit
pre-commit install

# Test the system
echo "🧪 Running tests..."
python -m pytest tests/ -v --cov=agents --cov=models --cov-report=html

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Edit .env file with your Telegram credentials"
echo "2. Activate environment: source trading_ai/bin/activate"
echo "3. Start system: python main.py --config environments/local_i7_setup.yaml"
echo "4. Monitor logs: tail -f logs/trading_system.log"
echo ""
echo "📚 For more information, see docs/user_guide.md"
