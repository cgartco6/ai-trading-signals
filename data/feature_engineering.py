"""
Advanced Feature Engineering for Trading
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import ta
from scipy import stats
import logging

class AdvancedFeatureEngine:
    """Creates sophisticated features for AI models"""
    
    @staticmethod
    def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive feature set"""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Price-based features
            price_features = AdvancedFeatureEngine._create_price_features(df)
            features = pd.concat([features, price_features], axis=1)
            
            # Volume-based features
            volume_features = AdvancedFeatureEngine._create_volume_features(df)
            features = pd.concat([features, volume_features], axis=1)
            
            # Technical indicators
            technical_features = AdvancedFeatureEngine._create_technical_indicators(df)
            features = pd.concat([features, technical_features], axis=1)
            
            # Statistical features
            statistical_features = AdvancedFeatureEngine._create_statistical_features(df)
            features = pd.concat([features, statistical_features], axis=1)
            
            # Cyclical features
            cyclical_features = AdvancedFeatureEngine._create_cyclical_features(df)
            features = pd.concat([features, cyclical_features], axis=1)
            
            # Fill NaN values
            features = features.ffill().bfill()
            
            return features
            
        except Exception as e:
            logging.error(f"Feature engineering failed: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def _create_price_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features"""
        features = pd.DataFrame(index=df.index)
        
        # Basic price transformations
        features['price_return'] = df['close'].pct_change()
        features['log_return'] = np.log(df['close'] / df['close'].shift(1))
        features['price_range'] = (df['high'] - df['low']) / df['close']
        features['body_size'] = abs(df['close'] - df['open']) / df['close']
        
        # Rolling statistics
        for window in [5, 10, 20]:
            features[f'rolling_mean_{window}'] = df['close'].rolling(window).mean()
            features[f'rolling_std_{window}'] = df['close'].rolling(window).std()
            features[f'rolling_min_{window}'] = df['close'].rolling(window).min()
            features[f'rolling_max_{window}'] = df['close'].rolling(window).max()
            
            # Z-score
            features[f'z_score_{window}'] = (
                (df['close'] - features[f'rolling_mean_{window}']) / 
                features[f'rolling_std_{window}']
            )
        
        # Momentum features
        features['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        features['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        features['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Price acceleration
        features['acceleration_5'] = features['momentum_5'] - features['momentum_5'].shift(5)
        features['acceleration_10'] = features['momentum_10'] - features['momentum_10'].shift(10)
        
        return features
    
    @staticmethod
    def _create_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features"""
        features = pd.DataFrame(index=df.index)
        
        if 'volume' not in df.columns:
            return features
        
        # Basic volume features
        features['volume_return'] = df['volume'].pct_change()
        features['volume_ma_5'] = df['volume'].rolling(5).mean()
        features['volume_ma_20'] = df['volume'].rolling(20).mean()
        
        # Volume ratio
        features['volume_ratio'] = df['volume'] / features['volume_ma_20']
        
        # Volume-price correlation
        features['volume_price_correlation_10'] = (
            df['close'].rolling(10).corr(df['volume'])
        )
        
        # On-Balance Volume
        features['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        # Volume Weighted Average Price
        features['vwap'] = (
            (df['close'] * df['volume']).rolling(20).sum() / 
            df['volume'].rolling(20).sum()
        )
        
        # Money Flow Index
        features['mfi'] = ta.volume.money_flow_index(
            df['high'], df['low'], df['close'], df['volume'], window=14
        )
        
        return features
    
    @staticmethod
    def _create_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features"""
        features = pd.DataFrame(index=df.index)
        
        # RSI
        features['rsi'] = ta.momentum.rsi(df['close'], window=14)
        features['rsi_ma'] = features['rsi'].rolling(5).mean()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        features['macd'] = macd.macd()
        features['macd_signal'] = macd.macd_signal()
        features['macd_histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        features['bb_upper'] = bollinger.bollinger_hband()
        features['bb_lower'] = bollinger.bollinger_lband()
        features['bb_middle'] = bollinger.bollinger_mavg()
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
        features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        features['stoch_k'] = stoch.stoch()
        features['stoch_d'] = stoch.stoch_signal()
        
        # ADX
        features['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
        
        # ATR
        features['atr'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close']
        ).average_true_range()
        
        # Ichimoku Cloud (simplified)
        ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
        features['ichimoku_a'] = ichimoku.ichimoku_a()
        features['ichimoku_b'] = ichimoku.ichimoku_b()
        
        # Parabolic SAR
        features['psar'] = ta.trend.PSARIndicator(
            df['high'], df['low'], df['close']
        ).psar()
        
        # Commodity Channel Index
        features['cci'] = ta.trend.CCIIndicator(
            df['high'], df['low'], df['close']
        ).cci()
        
        return features
    
    @staticmethod
    def _create_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features"""
        features = pd.DataFrame(index=df.index)
        
        returns = df['close'].pct_change().dropna()
        
        # Volatility measures
        features['volatility_5'] = returns.rolling(5).std()
        features['volatility_20'] = returns.rolling(20).std()
        features['volatility_ratio'] = features['volatility_5'] / features['volatility_20']
        
        # Skewness and Kurtosis
        features['skewness_10'] = returns.rolling(10).skew()
        features['kurtosis_10'] = returns.rolling(10).kurtosis()
        
        # Hurst Exponent (simplified)
        features['hurst_20'] = AdvancedFeatureEngine._calculate_hurst(df['close'], 20)
        
        # Autocorrelation
        features['autocorr_1'] = returns.rolling(20).apply(
            lambda x: x.autocorr(lag=1), raw=False
        )
        features['autocorr_5'] = returns.rolling(20).apply(
            lambda x: x.autocorr(lag=5), raw=False
        )
        
        # Shannon Entropy (simplified)
        features['entropy_10'] = returns.rolling(10).apply(
            lambda x: stats.entropy(np.histogram(x, bins=5)[0] + 1e-8), raw=False
        )
        
        return features
    
    @staticmethod
    def _create_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical time-based features"""
        features = pd.DataFrame(index=df.index)
        
        # Time-based features
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['day_of_month'] = df.index.day
        features['week_of_year'] = df.index.isocalendar().week
        features['month'] = df.index.month
        
        # Cyclical encoding for hour
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        
        # Cyclical encoding for day of week
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        # Cyclical encoding for month
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        
        # Trading session markers
        features['asian_session'] = ((features['hour'] >= 0) & (features['hour'] < 8)).astype(int)
        features['european_session'] = ((features['hour'] >= 8) & (features['hour'] < 16)).astype(int)
        features['us_session'] = ((features['hour'] >= 16) | (features['hour'] < 0)).astype(int)
        
        return features
    
    @staticmethod
    def _calculate_hurst(ts: pd.Series, window: int) -> pd.Series:
        """Calculate Hurst exponent (simplified version)"""
        def hurst_calc(x):
            if len(x) < window:
                return np.nan
            
            # Simplified Hurst calculation
            lags = range(2, min(20, len(x)))
            tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
        
        return ts.rolling(window).apply(hurst_calc, raw=True)

# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='1h')
    np.random.seed(42)
    
    sample_df = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(100) * 0.1),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.1) + 0.5,
        'low': 100 + np.cumsum(np.random.randn(100) * 0.1) - 0.5,
        'close': 100 + np.cumsum(np.random.randn(100) * 0.1),
        'volume': np.random.lognormal(10, 1, 100)
    }, index=dates)
    
    # Generate features
    features = AdvancedFeatureEngine.create_advanced_features(sample_df)
    print(f"Generated {len(features.columns)} features")
    print(f"Feature names: {list(features.columns)}")
