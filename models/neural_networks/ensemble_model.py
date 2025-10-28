"""
Ensemble Model that combines multiple AI models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class EnsembleModel(nn.Module):
    """Ensemble model that combines predictions from multiple models"""
    
    def __init__(self, models: List[nn.Module], weights: List[float] = None):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        
        if weights is None:
            # Equal weights if not specified
            weights = [1.0 / len(models)] * len(models)
        
        # Normalize weights
        weight_sum = sum(weights)
        self.weights = [w / weight_sum for w in weights]
        
        # Learnable weighting (optional)
        self.learnable_weights = nn.Parameter(torch.ones(len(models)))
        
        self.ensemble_fc = nn.Linear(3 * len(models), 3)  # Combine all predictions
    
    def forward(self, x):
        model_outputs = []
        
        # Get predictions from all models
        for model in self.models:
            output = model(x)
            model_outputs.append(output)
        
        # Stack outputs
        stacked_outputs = torch.stack(model_outputs, dim=1)  # (batch, num_models, 3)
        
        # Apply learned weights
        weights = F.softmax(self.learnable_weights, dim=0)
        weighted_outputs = stacked_outputs * weights.unsqueeze(0).unsqueeze(-1)
        
        # Combine predictions
        combined = torch.sum(weighted_outputs, dim=1)  # (batch, 3)
        
        return combined
    
    def get_model_contributions(self, x):
        """Get contribution of each model to the final prediction"""
        model_outputs = []
        
        with torch.no_grad():
            for model in self.models:
                output = model(x)
                model_outputs.append(F.softmax(output, dim=1))
        
        stacked_outputs = torch.stack(model_outputs, dim=1)
        weights = F.softmax(self.learnable_weights, dim=0)
        
        contributions = {}
        for i, model in enumerate(self.models):
            contributions[f'model_{i}'] = {
                'weight': weights[i].item(),
                'prediction': model_outputs[i].cpu().numpy(),
                'contribution': (weights[i] * model_outputs[i]).cpu().numpy()
            }
        
        return contributions

if __name__ == "__main__":
    # Test with dummy models
    from .transformer_model import TransformerTradingModel
    from .lstm_attention import LSTMAttentionModel
    
    # Create dummy models
    model1 = TransformerTradingModel(feature_size=128)
    model2 = LSTMAttentionModel(input_size=128)
    
    # Create ensemble
    ensemble = EnsembleModel([model1, model2], weights=[0.6, 0.4])
    
    # Test forward pass
    sample_input = torch.randn(32, 50, 128)
    output = ensemble(sample_input)
    
    print(f"Ensemble input shape: {sample_input.shape}")
    print(f"Ensemble output shape: {output.shape}")
    print(f"Ensemble output: {output}")
    
    # Get model contributions
    contributions = ensemble.get_model_contributions(sample_input)
    for model_name, contrib in contributions.items():
        print(f"{model_name}: weight={contrib['weight']:.3f}")
