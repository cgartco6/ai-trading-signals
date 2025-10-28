"""
LSTM with Attention Mechanism for Time Series Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    """Attention mechanism for LSTM outputs"""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        
        # Calculate attention weights
        attention_weights = self.attention(lstm_output)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention weights
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # (batch_size, hidden_size)
        
        return context_vector, attention_weights

class LSTMAttentionModel(nn.Module):
    """LSTM with attention for trading signal prediction"""
    
    def __init__(self, input_size=50, hidden_size=128, num_layers=2, 
                 dropout=0.2, bidirectional=True):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Attention layer
        self.attention = AttentionLayer(hidden_size * (2 if bidirectional else 1))
        
        # Fully connected layers
        fc_input_size = hidden_size * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(fc_input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)  # BUY, SELL, HOLD
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(fc_input_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        for module in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # LSTM layer
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        context_vector, attention_weights = self.attention(lstm_out)
        
        # Layer normalization
        context_vector = self.layer_norm(context_vector)
        
        # Fully connected layers
        x = F.relu(self.fc1(context_vector))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

if __name__ == "__main__":
    # Test the model
    model = LSTMAttentionModel(input_size=50, hidden_size=128)
    sample_input = torch.randn(32, 50, 50)  # (batch, sequence, features)
    output = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
