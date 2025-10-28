"""
Market Psychology Agent - Analyzes market sentiment and behavioral patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import asyncio

class MarketPsychologyAgent:
    """AI agent that analyzes market psychology and behavioral finance patterns"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.market_mood = 'NEUTRAL'
        self.herding_indicator = 0.5
        self.fear_greed_history = []
        
    async def analyze_symbol(self, symbol: str, timeframe: str, market_regime) -> Optional[Dict]:
        """Analyze market psychology for a symbol"""
        try:
            from ..data.market_data import MarketDataFetcher
            data_fetcher = MarketDataFetcher(self.config)
            df = await data_fetcher.get_symbol_data(symbol, timeframe, limit=100)
            
            if df is None:
                return None
            
            # Analyze various psychological indicators
            psychology_metrics = await asyncio.gather(
                self._analyze_herd_behavior(df, symbol),
                self._analyze_fear_greed(df),
                self._analyze_market_sentiment_extremes(df),
                self._analyze_contrarian_indicators(df),
                return_exceptions=True
            )
            
            valid_metrics = [m for m in psychology_metrics if not isinstance(m, Exception) and m is not None]
            
            if not valid_metrics:
                return None
            
            # Calculate overall psychology score
            psych_score = np.mean([m['score'] for m in valid_metrics])
            confidence = np.mean([m.get('confidence', 0.5) for m in valid_metrics])
            
            # Convert to trading action
            action, action_confidence = self._psychology_to_action(psych_score, confidence, market_regime)
            
            if action != 'HOLD':
                return {
                    'action': action,
                    'confidence': action_confidence,
                    'psychology_score': psych_score,
                    'market_mood': self.market_mood,
                    'psychology_metrics': valid_metrics
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Market psychology analysis failed for {symbol}: {e}")
            return None
    
    async def _analyze_herd_behavior(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Analyze herd behavior in the market"""
        try:
            # Use volume and price correlation to detect herding
            volume = df['volume'].values
            price_changes = df['close'].pct_change().values
            
            # Calculate volume-price correlation
            valid_data = ~np.isnan(price_changes)
            if np.sum(valid_data) > 10:
                correlation = np.corrcoef(volume[valid_data][1:], 
                                        price_changes[valid_data][1:])[0,1]
            else:
                correlation = 0
            
            # High positive correlation suggests herding
            herd_strength = abs(correlation)
            
            # Recent volume surge detection
            recent_volume = volume[-5:].mean()
            historical_volume = volume[:-5].mean() if len(volume) > 5 else recent_volume
            
            volume_surge = recent_volume / historical_volume if historical_volume > 0 else 1
            
            # Combine indicators
            herd_score = min(1.0, (herd_strength * 0.6 + min(2.0, volume_surge) * 0.4))
            
            # High herd score often precedes reversals (contrarian indicator)
            return {
                'metric': 'herd_behavior',
                'score': herd_score,
                'confidence': 0.7,
                'volume_correlation': correlation,
                'volume_surge': volume_surge
            }
            
        except Exception as e:
            self.logger.warning(f"Herd behavior analysis failed: {e}")
            return {'metric': 'herd_behavior', 'score': 0.5, 'confidence': 0.5}
    
    async def _analyze_fear_greed(self, df: pd.DataFrame) -> Dict:
        """Analyze fear and greed indicators"""
        try:
            prices = df['close'].values
            returns = df['close'].pct_change().dropna().values
            
            # Volatility-based fear indicator
            volatility = np.std(returns[-10:]) if len(returns) >= 10 else np.std(returns)
            fear_from_vol = min(1.0, volatility * 20)  # Scale volatility
            
            # Drawdown-based fear
            current_price = prices[-1]
            recent_high = np.max(prices[-20:])
            drawdown = (recent_high - current_price) / recent_high
            fear_from_drawdown = min(1.0, drawdown * 3)  # Scale drawdown
            
            # Momentum-based greed
            short_momentum = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
            long_momentum = (prices[-1] - prices[-20]) / prices[-20] if len(prices) >= 20 else 0
            
            greed_from_momentum = min(1.0, max(0, (short_momentum + long_momentum) * 10))
            
            # Combined fear-greed score (0=fear, 1=greed)
            fear_greed_score = (fear_from_vol * 0.3 + fear_from_drawdown * 0.3 + 
                              greed_from_momentum * 0.4)
            
            # Extreme fear/greed often signals reversals
            if fear_greed_score < 0.3:
                self.market_mood = 'FEAR'
                # Extreme fear -> potential buying opportunity
                contrarian_score = 1.0 - fear_greed_score
            elif fear_greed_score > 0.7:
                self.market_mood = 'GREED'
                # Extreme greed -> potential selling opportunity
                contrarian_score = fear_greed_score
            else:
                self.market_mood = 'NEUTRAL'
                contrarian_score = 0.5
            
            self.fear_greed_history.append(fear_greed_score)
            if len(self.fear_greed_history) > 50:
                self.fear_greed_history.pop(0)
            
            return {
                'metric': 'fear_greed',
                'score': contrarian_score,
                'confidence': 0.8,
                'fear_greed_index': fear_greed_score,
                'market_mood': self.market_mood
            }
            
        except Exception as e:
            self.logger.warning(f"Fear-greed analysis failed: {e}")
            return {'metric': 'fear_greed', 'score': 0.5, 'confidence': 0.5}
    
    async def _analyze_market_sentiment_extremes(self, df: pd.DataFrame) -> Dict:
        """Detect sentiment extremes that often precede reversals"""
        try:
            # Use RSI and other oscillators to detect extremes
            from ..data.feature_engineering import AdvancedFeatureEngine
            features = AdvancedFeatureEngine.create_advanced_features(df)
            
            if 'rsi' not in features.columns:
                return {'metric': 'sentiment_extremes', 'score': 0.5, 'confidence': 0.5}
            
            rsi = features['rsi'].iloc[-1]
            
            # RSI extremes
            if rsi < 30:
                # Oversold - potential buying opportunity
                oversold_score = (30 - rsi) / 30
                return {
                    'metric': 'sentiment_extremes',
                    'score': 1.0 - oversold_score,  # Higher score for more extreme oversold
                    'confidence': 0.8,
                    'rsi_level': rsi,
                    'condition': 'OVERSOLD'
                }
            elif rsi > 70:
                # Overbought - potential selling opportunity
                overbought_score = (rsi - 70) / 30
                return {
                    'metric': 'sentiment_extremes',
                    'score': overbought_score,  # Higher score for more extreme overbought
                    'confidence': 0.8,
                    'rsi_level': rsi,
                    'condition': 'OVERBOUGHT'
                }
            else:
                return {
                    'metric': 'sentiment_extremes',
                    'score': 0.5,
                    'confidence': 0.6,
                    'rsi_level': rsi,
                    'condition': 'NEUTRAL'
                }
                
        except Exception as e:
            self.logger.warning(f"Sentiment extremes analysis failed: {e}")
            return {'metric': 'sentiment_extremes', 'score': 0.5, 'confidence': 0.5}
    
    async def _analyze_contrarian_indicators(self, df: pd.DataFrame) -> Dict:
        """Analyze contrarian indicators for market turns"""
        try:
            prices = df['close'].values
            volumes = df['volume'].values
            
            # Price-volume divergence
            price_trend = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
            volume_trend = (volumes[-1] - volumes[-10]) / volumes[-10] if len(volumes) >= 10 else 0
            
            # Bullish divergence: price down, volume up -> potential bottom
            # Bearish divergence: price up, volume down -> potential top
            if price_trend < 0 and volume_trend > 0.1:
                divergence_score = 0.8  # Bullish divergence
            elif price_trend > 0 and volume_trend < -0.1:
                divergence_score = 0.8  # Bearish divergence
            else:
                divergence_score = 0.5
            
            # Confidence based on divergence strength
            confidence = min(0.9, abs(price_trend) * 10 + abs(volume_trend))
            
            return {
                'metric': 'contrarian',
                'score': divergence_score,
                'confidence': confidence,
                'price_trend': price_trend,
                'volume_trend': volume_trend
            }
            
        except Exception as e:
            self.logger.warning(f"Contrarian analysis failed: {e}")
            return {'metric': 'contrarian', 'score': 0.5, 'confidence': 0.5}
    
    def _psychology_to_action(self, psychology_score: float, confidence: float, market_regime) -> tuple:
        """Convert psychology score to trading action"""
        # Psychology scores > 0.7 indicate strong contrarian opportunities
        # Scores < 0.3 indicate following the trend might be better
        
        if psychology_score > 0.7 and confidence > 0.6:
            # Strong contrarian signal - often opposite to current trend
            if self.market_mood == 'FEAR':
                return 'BUY', confidence
            elif self.market_mood == 'GREED':
                return 'SELL', confidence
            else:
                return 'HOLD', 0
        elif psychology_score < 0.3 and confidence > 0.6:
            # Weak psychology signal - follow the trend
            if market_regime.value in ['TRENDING', 'RECOVERY']:
                return 'BUY', confidence * 0.8
            elif market_regime.value in ['CRASH', 'VOLATILE']:
                return 'SELL', confidence * 0.8
            else:
                return 'HOLD', 0
        else:
            return 'HOLD', 0
    
    async def analyze_market_regime(self):
        """Analyze market regime from psychology perspective"""
        from .master_orchestrator import MarketRegime
        
        if len(self.fear_greed_history) < 10:
            return MarketRegime.TRENDING
        
        avg_fear_greed = np.mean(self.fear_greed_history)
        
        if avg_fear_greed < 0.3:
            return MarketRegime.CRASH
        elif avg_fear_greed > 0.7:
            return MarketRegime.TRENDING
        elif 0.4 <= avg_fear_greed <= 0.6:
            return MarketRegime.RANGING
        else:
            return MarketRegime.VOLATILE
