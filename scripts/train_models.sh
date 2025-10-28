#!/bin/bash

# AI Model Training Script

set -e

echo "🤖 Starting AI Model Training..."

# Activate virtual environment
if [ -d "trading_ai" ]; then
    source trading_ai/bin/activate
fi

# Create models directory
mkdir -p models/trained_models

# Train Transformer model
echo "🔧 Training Transformer model..."
python -c "
import torch
from models.neural_networks.transformer_model import TransformerTradingModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    model = TransformerTradingModel(feature_size=128, num_layers=4)
    
    # Save initial model
    torch.save(model.state_dict(), 'models/trained_models/transformer_base.pth')
    logger.info('Transformer model saved successfully')
    
except Exception as e:
    logger.error(f'Transformer training failed: {e}')
    raise
"

# Train LSTM model
echo "🔧 Training LSTM Attention model..."
python -c "
import torch
from models.neural_networks.lstm_attention import LSTMAttentionModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    model = LSTMAttentionModel(input_size=50, hidden_size=128)
    
    # Save initial model
    torch.save(model.state_dict(), 'models/trained_models/lstm_attention_base.pth')
    logger.info('LSTM Attention model saved successfully')
    
except Exception as e:
    logger.error(f'LSTM training failed: {e}')
    raise
"

# Create ensemble model
echo "🔧 Creating ensemble model..."
python -c "
import torch
from models.neural_networks.ensemble_model import EnsembleModel
from models.neural_networks.transformer_model import TransformerTradingModel
from models.neural_networks.lstm_attention import LSTMAttentionModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Load base models
    transformer = TransformerTradingModel(feature_size=128)
    lstm = LSTMAttentionModel(input_size=128)
    
    # Create ensemble
    ensemble = EnsembleModel([transformer, lstm], weights=[0.6, 0.4])
    
    # Save ensemble
    torch.save(ensemble.state_dict(), 'models/trained_models/ensemble_base.pth')
    logger.info('Ensemble model saved successfully')
    
except Exception as e:
    logger.error(f'Ensemble creation failed: {e}')
    raise
"

echo "✅ Model training completed successfully!"
echo ""
echo "📁 Models saved in: models/trained_models/"
echo "🔧 Next: Run the system with: python main.py"
