"""
Deep Predictive AI Agent with Neural Networks
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional
import asyncio
import logging
from datetime import datetime
from ..models.neural_networks.transformer_model import TransformerTradingModel
from ..models.neural_networks.lstm_attention import LSTMAttentionModel
from ..models.neural_networks.ensemble_model import EnsembleModel

class DeepPredictiveAgent:
    """Deep learning agent for price prediction using multiple neural networks"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all deep learning models"""
        try:
            # Transformer model for sequence prediction
            self.models['transformer'] = TransformerTradingModel(
                feature_size=128,
                num_layers=4,
                nhead=8
            )
            
            # LSTM with Attention for temporal patterns
            self.models['lstm_attention'] = LSTMAttentionModel(
                input_size=50,
                hidden_size=128,
                num_layers=2
            )
            
            # Ensemble model to combine predictions
            self.models['ensemble'] = EnsembleModel(
                models=list(self.models.values())[:2],  # Exclude ensemble itself
                weights=[0.5, 0.5]
            )
            
            # Load pre-trained weights if available
            self._load_model_weights()
            
            # Set models to evaluation mode
            for model in self.models.values():
                model.to(self.device)
                model.eval()
                
            self.logger.info("Deep learning models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            raise
    
    def _load_model_weights(self):
        """Load pre-trained model weights"""
        try:
            # This would load from models/trained_models/
            # For now, we'll initialize with random weights
            self.logger.info("No pre-trained weights found, using initialized models")
        except Exception as e:
            self.logger.warning(f"Could not load model weights: {e}")
    
    async def analyze_symbol(self, symbol: str, timeframe: str, market_regime) -> Optional[Dict]:
        """Analyze symbol using deep learning models"""
        try:
            # Get market data
            from ..data.market_data import MarketDataFetcher
            data_fetcher = MarketDataFetcher(self.config)
            df = await data_fetcher.get_symbol_data(symbol, timeframe, limit=100)
            
            if df is None or len(df) < 50:
                return None
            
            # Prepare features
            from ..data.feature_engineering import AdvancedFeatureEngine
            features = AdvancedFeatureEngine.create_advanced_features(df)
            
            # Convert to tensor
            feature_tensor = torch.FloatTensor(features.values[-50:]).unsqueeze(0).to(self.device)
            
            # Get predictions from all models
            predictions = {}
            for name, model in self.models.items():
                with torch.no_grad():
                    output = model(feature_tensor)
                    pred = torch.softmax(output, dim=1)
                    confidence, action_idx = torch.max(pred, 1)
                    
                action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
                predictions[name] = {
                    'action': action_map[action_idx.item()],
                    'confidence': confidence.item()
                }
            
            # Ensemble the model predictions
            final_prediction = self._ensemble_model_predictions(predictions)
            
            if final_prediction['action'] != 'HOLD':
                # Calculate price levels
                current_price = df['close'].iloc[-1]
                price_levels = self._calculate_price_levels(df, final_prediction['action'])
                
                return {
                    'action': final_prediction['action'],
                    'confidence': final_prediction['confidence'],
                    'entry_price': current_price,
                    'targets': price_levels['targets'],
                    'stop_loss': price_levels['stop_loss'],
                    'model_breakdown': predictions
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Deep learning analysis failed for {symbol}: {e}")
            return None
    
    def _ensemble_model_predictions(self, predictions: Dict) -> Dict:
        """Combine predictions from multiple models"""
        action_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        model_weights = {'transformer': 0.4, 'lstm_attention': 0.4, 'ensemble': 0.2}
        
        for model_name, prediction in predictions.items():
            weight = model_weights.get(model_name, 0.1)
            action = prediction['action']
            confidence = prediction['confidence']
            
            action_scores[action] += confidence * weight
        
        # Determine final action
        final_action = max(action_scores, key=action_scores.get)
        final_confidence = action_scores[final_action]
        
        return {
            'action': final_action,
            'confidence': final_confidence
        }
    
    def _calculate_price_levels(self, df, action: str) -> Dict:
        """Calculate entry, target, and stop loss levels"""
        current_price = df['close'].iloc[-1]
        atr = self._calculate_atr(df, period=14)
        
        if action == 'BUY':
            stop_loss = current_price - (2 * atr)
            targets = [
                current_price + (1 * atr),
                current_price + (2 * atr),
                current_price + (3 * atr)
            ]
        else:  # SELL
            stop_loss = current_price + (2 * atr)
            targets = [
                current_price - (1 * atr),
                current_price - (2 * atr),
                current_price - (3 * atr)
            ]
        
        return {
            'targets': targets,
            'stop_loss': stop_loss
        }
    
    def _calculate_atr(self, df, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = true_range.rolling(period).mean().iloc[-1]
        
        return atr if not np.isnan(atr) else current_price * 0.02
    
    async def analyze_market_regime(self):
        """Analyze market regime using deep learning"""
        # Implement regime detection using neural networks
        # For now, return a basic regime
        from .master_orchestrator import MarketRegime
        return MarketRegime.TRENDING
    
    async def retrain_models(self, new_data):
        """Retrain models with new data"""
        # Implement model retraining logic
        pass
