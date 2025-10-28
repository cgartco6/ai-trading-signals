"""
Transformer Model for Financial Time Series Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer sequences"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerTradingModel(nn.Module):
    """Transformer-based model for trading signal prediction"""
    
    def __init__(self, feature_size=128, num_layers=4, nhead=8, 
                 dim_feedforward=512, dropout=0.1, max_seq_len=100):
        super().__init__()
        
        self.feature_size = feature_size
        self.num_layers = num_layers
        self.nhead = nhead
        
        # Input projection
        self.input_projection = nn.Linear(feature_size, feature_size)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(feature_size, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Attention pooling
        self.attention_pool = nn.MultiheadAttention(feature_size, 1, dropout=dropout)
        
        # Output layers
        self.fc1 = nn.Linear(feature_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)  # BUY, SELL, HOLD
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_size)
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        x = self.layer_norm(x)
        
        # Add positional encoding
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Transformer encoding
        transformer_out = self.transformer(x)
        
        # Attention pooling
        query = torch.mean(transformer_out, dim=1, keepdim=True)  # (batch_size, 1, feature_size)
        attn_out, attn_weights = self.attention_pool(
            query.transpose(0, 1),  # (1, batch_size, feature_size)
            transformer_out.transpose(0, 1),  # (seq_len, batch_size, feature_size)
            transformer_out.transpose(0, 1)
        )
        
        # Fully connected layers
        x = attn_out.squeeze(0)  # (batch_size, feature_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

if __name__ == "__main__":
    # Test the model
    model = TransformerTradingModel(feature_size=128, num_layers=4)
    sample_input = torch.randn(32, 50, 128)  # (batch, sequence, features)
    output = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
