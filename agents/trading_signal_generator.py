import pandas as pd
import numpy as np
import requests
import ta
from datetime import datetime
import time
import logging

class TradingSignalGenerator:
    def __init__(self):
        self.signals = []
        
    def calculate_technical_indicators(self, df):
        """Calculate multiple technical indicators"""
        # Moving Averages
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        return df
    
    def generate_signal(self, symbol, df, market_type='crypto'):
        """Generate trading signals based on technical analysis"""
        if len(df) < 50:
            return None
            
        df = self.calculate_technical_indicators(df)
        current_price = df['close'].iloc[-1]
        signal = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'price': current_price,
            'signal': 'HOLD',
            'confidence': 0,
            'indicators': {}
        }
        
        # Signal logic
        buy_signals = 0
        total_signals = 0
        
        # 1. Moving Average Crossover
        if df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1] and df['sma_20'].iloc[-2] <= df['sma_50'].iloc[-2]:
            buy_signals += 1
            signal['indicators']['ma_cross'] = 'BULLISH'
        total_signals += 1
        
        # 2. RSI Analysis
        if df['rsi'].iloc[-1] < 30:
            buy_signals += 1
            signal['indicators']['rsi'] = 'OVERSOLD'
        elif df['rsi'].iloc[-1] > 70:
            signal['indicators']['rsi'] = 'OVERBOUGHT'
        else:
            buy_signals += 0.5
        total_signals += 1
        
        # 3. MACD Analysis
        if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] and df['macd_histogram'].iloc[-1] > 0:
            buy_signals += 1
            signal['indicators']['macd'] = 'BULLISH'
        total_signals += 1
        
        # 4. Bollinger Bands
        if df['close'].iloc[-1] < df['bb_lower'].iloc[-1]:
            buy_signals += 1
            signal['indicators']['bollinger'] = 'OVERSOLD'
        elif df['close'].iloc[-1] > df['bb_upper'].iloc[-1]:
            signal['indicators']['bollinger'] = 'OVERBOUGHT'
        else:
            buy_signals += 0.5
        total_signals += 1
        
        # 5. Stochastic
        if df['stoch_k'].iloc[-1] < 20 and df['stoch_d'].iloc[-1] < 20:
            buy_signals += 1
            signal['indicators']['stochastic'] = 'OVERSOLD'
        elif df['stoch_k'].iloc[-1] > 80 and df['stoch_d'].iloc[-1] > 80:
            signal['indicators']['stochastic'] = 'OVERBOUGHT'
        total_signals += 1
        
        # Calculate confidence and determine signal
        confidence = (buy_signals / total_signals) * 100
        
        if confidence >= 60:
            signal['signal'] = 'BUY'
            signal['confidence'] = confidence
        elif confidence <= 40:
            signal['signal'] = 'SELL'
            signal['confidence'] = 100 - confidence
        else:
            return None
            
        return signal
