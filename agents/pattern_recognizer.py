"""
Pattern Recognition Agent for Technical Chart Patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import talib
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN

class PatternRecognitionAgent:
    """AI agent for recognizing technical chart patterns"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.patterns_db = {}
        self._initialize_pattern_library()
    
    def _initialize_pattern_library(self):
        """Initialize known technical patterns"""
        self.patterns_db = {
            'double_top': self._detect_double_top,
            'double_bottom': self._detect_double_bottom,
            'head_shoulders': self._detect_head_shoulders,
            'triangle': self._detect_triangle,
            'flag': self._detect_flag,
            'wedge': self._detect_wedge
        }
    
    async def analyze_symbol(self, symbol: str, timeframe: str, market_regime) -> Optional[Dict]:
        """Analyze symbol for technical patterns"""
        try:
            from ..data.market_data import MarketDataFetcher
            data_fetcher = MarketDataFetcher(self.config)
            df = await data_fetcher.get_symbol_data(symbol, timeframe, limit=100)
            
            if df is None or len(df) < 50:
                return None
            
            # Detect all patterns
            detected_patterns = []
            for pattern_name, pattern_func in self.patterns_db.items():
                result = pattern_func(df)
                if result['detected']:
                    detected_patterns.append({
                        'name': pattern_name,
                        'confidence': result['confidence'],
                        'action': result['action'],
                        'target_price': result.get('target_price'),
                        'stop_loss': result.get('stop_loss')
                    })
            
            if not detected_patterns:
                return None
            
            # Find the highest confidence pattern
            best_pattern = max(detected_patterns, key=lambda x: x['confidence'])
            
            return {
                'action': best_pattern['action'],
                'confidence': best_pattern['confidence'],
                'pattern_name': best_pattern['name'],
                'target_price': best_pattern.get('target_price'),
                'stop_loss': best_pattern.get('stop_loss'),
                'all_patterns': detected_patterns
            }
            
        except Exception as e:
            self.logger.error(f"Pattern recognition failed for {symbol}: {e}")
            return None
    
    def _detect_double_top(self, df: pd.DataFrame) -> Dict:
        """Detect double top pattern"""
        try:
            high = df['high'].values
            low = df['low'].values
            
            # Find peaks in the high prices
            peaks, _ = find_peaks(high, distance=10, prominence=np.std(high)*0.5)
            
            if len(peaks) < 2:
                return {'detected': False, 'confidence': 0}
            
            # Check if last two peaks form a double top
            last_peak = peaks[-1]
            second_last_peak = peaks[-2]
            
            last_peak_price = high[last_peak]
            second_last_peak_price = high[second_last_peak]
            
            # Prices should be similar (within 1%)
            price_similarity = abs(last_peak_price - second_last_peak_price) / second_last_peak_price
            
            if price_similarity < 0.01:
                # Check for neckline break
                trough = np.min(low[second_last_peak:last_peak])
                current_price = df['close'].iloc[-1]
                
                if current_price < trough:
                    confidence = 0.8 - (price_similarity * 10)
                    target_price = trough - (last_peak_price - trough)
                    
                    return {
                        'detected': True,
                        'confidence': min(confidence, 0.9),
                        'action': 'SELL',
                        'target_price': target_price,
                        'stop_loss': last_peak_price * 1.01
                    }
            
            return {'detected': False, 'confidence': 0}
            
        except Exception as e:
            self.logger.warning(f"Double top detection failed: {e}")
            return {'detected': False, 'confidence': 0}
    
    def _detect_double_bottom(self, df: pd.DataFrame) -> Dict:
        """Detect double bottom pattern"""
        try:
            high = df['high'].values
            low = df['low'].values
            
            # Find troughs in the low prices
            troughs, _ = find_peaks(-low, distance=10, prominence=np.std(low)*0.5)
            
            if len(troughs) < 2:
                return {'detected': False, 'confidence': 0}
            
            # Check if last two troughs form a double bottom
            last_trough = troughs[-1]
            second_last_trough = troughs[-2]
            
            last_trough_price = low[last_trough]
            second_last_trough_price = low[second_last_trough]
            
            # Prices should be similar (within 1%)
            price_similarity = abs(last_trough_price - second_last_trough_price) / second_last_trough_price
            
            if price_similarity < 0.01:
                # Check for neckline break
                peak = np.max(high[second_last_trough:last_trough])
                current_price = df['close'].iloc[-1]
                
                if current_price > peak:
                    confidence = 0.8 - (price_similarity * 10)
                    target_price = peak + (peak - last_trough_price)
                    
                    return {
                        'detected': True,
                        'confidence': min(confidence, 0.9),
                        'action': 'BUY',
                        'target_price': target_price,
                        'stop_loss': last_trough_price * 0.99
                    }
            
            return {'detected': False, 'confidence': 0}
            
        except Exception as e:
            self.logger.warning(f"Double bottom detection failed: {e}")
            return {'detected': False, 'confidence': 0}
    
    def _detect_head_shoulders(self, df: pd.DataFrame) -> Dict:
        """Detect head and shoulders pattern"""
        try:
            high = df['high'].values
            
            # Find peaks for potential head and shoulders
            peaks, properties = find_peaks(high, distance=15, prominence=np.std(high)*0.3)
            
            if len(peaks) < 3:
                return {'detected': False, 'confidence': 0}
            
            # Check last three peaks for head and shoulders pattern
            right_shoulder = peaks[-1]
            head = peaks[-2]
            left_shoulder = peaks[-3]
            
            # Head should be higher than shoulders
            if (high[head] > high[left_shoulder] and 
                high[head] > high[right_shoulder] and
                abs(high[left_shoulder] - high[right_shoulder]) / high[left_shoulder] < 0.02):
                
                # Calculate neckline
                neckline = (df['low'].iloc[left_shoulder] + df['low'].iloc[right_shoulder]) / 2
                current_price = df['close'].iloc[-1]
                
                if current_price < neckline:
                    confidence = 0.75
                    target_price = neckline - (high[head] - neckline)
                    
                    return {
                        'detected': True,
                        'confidence': confidence,
                        'action': 'SELL',
                        'target_price': target_price,
                        'stop_loss': high[head] * 1.01
                    }
            
            return {'detected': False, 'confidence': 0}
            
        except Exception as e:
            self.logger.warning(f"Head and shoulders detection failed: {e}")
            return {'detected': False, 'confidence': 0}
    
    def _detect_triangle(self, df: pd.DataFrame) -> Dict:
        """Detect triangle pattern (symmetrical, ascending, descending)"""
        try:
            high = df['high'].values
            low = df['low'].values
            
            # Use linear regression to find trend lines
            from scipy import stats
            
            x = np.arange(len(high))
            
            # Upper trend line (resistance)
            high_slope, high_intercept, _, _, _ = stats.linregress(x[-20:], high[-20:])
            
            # Lower trend line (support)
            low_slope, low_intercept, _, _, _ = stats.linregress(x[-20:], low[-20:])
            
            # Check for converging trends (triangle)
            if (high_slope < 0 and low_slope > 0) or (abs(high_slope - low_slope) > 0.001):
                
                current_price = df['close'].iloc[-1]
                upper_line = high_slope * len(high) + high_intercept
                lower_line = low_slope * len(low) + low_intercept
                
                # Price is near breakout
                if (current_price > upper_line * 0.98 and current_price < upper_line * 1.02) or \
                   (current_price > lower_line * 0.98 and current_price < lower_line * 1.02):
                    
                    # Determine breakout direction
                    if high_slope < low_slope:  # Symmetrical triangle
                        # Wait for confirmation
                        return {'detected': False, 'confidence': 0}
                    elif high_slope < 0:  # Descending triangle - bearish
                        confidence = 0.7
                        target_price = lower_line - (upper_line - lower_line)
                        
                        return {
                            'detected': True,
                            'confidence': confidence,
                            'action': 'SELL',
                            'target_price': target_price,
                            'stop_loss': upper_line * 1.02
                        }
                    else:  # Ascending triangle - bullish
                        confidence = 0.7
                        target_price = upper_line + (upper_line - lower_line)
                        
                        return {
                            'detected': True,
                            'confidence': confidence,
                            'action': 'BUY',
                            'target_price': target_price,
                            'stop_loss': lower_line * 0.98
                        }
            
            return {'detected': False, 'confidence': 0}
            
        except Exception as e:
            self.logger.warning(f"Triangle detection failed: {e}")
            return {'detected': False, 'confidence': 0}
    
    def _detect_flag(self, df: pd.DataFrame) -> Dict:
        """Detect flag and pennant patterns"""
        # Implementation for flag pattern detection
        return {'detected': False, 'confidence': 0}
    
    def _detect_wedge(self, df: pd.DataFrame) -> Dict:
        """Detect rising and falling wedge patterns"""
        # Implementation for wedge pattern detection
        return {'detected': False, 'confidence': 0}
    
    async def analyze_market_regime(self):
        """Analyze market regime based on pattern prevalence"""
        from .master_orchestrator import MarketRegime
        # Pattern-based regime analysis
        return MarketRegime.TRENDING
