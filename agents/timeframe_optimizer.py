"""
Timeframe Optimization Agent - Finds optimal trading timeframes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import asyncio

class TimeframeOptimizationAgent:
    """AI agent for optimal timeframe selection based on market conditions"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.timeframe_hierarchy = ['1m', '5m', '15m', '1h', '4h', '1d', '1w']
        self.optimal_timeframes = {}
        self.timeframe_scores = {}
        
    async def get_optimal_timeframe(self, market_regime) -> str:
        """Get optimal timeframe for current market regime"""
        try:
            # Analyze all timeframes for the current regime
            timeframe_analysis = await asyncio.gather(
                *[self._analyze_timeframe(tf, market_regime) for tf in self.timeframe_hierarchy],
                return_exceptions=True
            )
            
            # Filter valid results
            valid_analysis = []
            for i, result in enumerate(timeframe_analysis):
                if not isinstance(result, Exception) and result is not None:
                    valid_analysis.append((self.timeframe_hierarchy[i], result))
            
            if not valid_analysis:
                return '1h'  # Default fallback
            
            # Find timeframe with highest score
            best_timeframe = max(valid_analysis, key=lambda x: x[1]['score'])
            
            self.optimal_timeframes[market_regime.value] = best_timeframe[0]
            self.logger.info(f"Optimal timeframe for {market_regime.value}: {best_timeframe[0]} (score: {best_timeframe[1]['score']:.2f})")
            
            return best_timeframe[0]
            
        except Exception as e:
            self.logger.error(f"Timeframe optimization failed: {e}")
            return '1h'  # Default fallback
    
    async def _analyze_timeframe(self, timeframe: str, market_regime) -> Optional[Dict]:
        """Analyze a specific timeframe for trading suitability"""
        try:
            from ..data.market_data import MarketDataFetcher
            data_fetcher = MarketDataFetcher(self.config)
            
            # Get data for multiple symbols to assess timeframe generally
            test_symbols = ['BTCUSDT', 'EURUSD']  # Crypto and Forex representatives
            
            symbol_scores = []
            for symbol in test_symbols:
                df = await data_fetcher.get_symbol_data(symbol, timeframe, limit=100)
                if df is not None:
                    score = await self._calculate_timeframe_score(df, timeframe, market_regime, symbol)
                    symbol_scores.append(score)
            
            if not symbol_scores:
                return None
            
            # Average score across symbols
            avg_score = np.mean(symbol_scores)
            
            return {
                'timeframe': timeframe,
                'score': avg_score,
                'symbol_scores': dict(zip(test_symbols, symbol_scores)),
                'market_regime': market_regime.value
            }
            
        except Exception as e:
            self.logger.warning(f"Timeframe analysis failed for {timeframe}: {e}")
            return None
    
    async def _calculate_timeframe_score(self, df: pd.DataFrame, timeframe: str, market_regime, symbol: str) -> float:
        """Calculate score for a timeframe based on multiple factors"""
        try:
            if df is None or len(df) < 20:
                return 0.0
            
            scores = []
            weights = []
            
            # 1. Volatility suitability
            vol_score, vol_weight = await self._assess_volatility_suitability(df, timeframe, market_regime)
            scores.append(vol_score)
            weights.append(vol_weight)
            
            # 2. Trend clarity
            trend_score, trend_weight = await self._assess_trend_clarity(df, timeframe, market_regime)
            scores.append(trend_score)
            weights.append(trend_weight)
            
            # 3. Noise level
            noise_score, noise_weight = await self._assess_noise_level(df, timeframe)
            scores.append(noise_score)
            weights.append(noise_weight)
            
            # 4. Trading opportunities
            opportunity_score, opportunity_weight = await self._assess_trading_opportunities(df, timeframe)
            scores.append(opportunity_score)
            weights.append(opportunity_weight)
            
            # Calculate weighted average
            total_weight = sum(weights)
            if total_weight > 0:
                weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
            else:
                weighted_score = 0.5
            
            # Adjust for market regime
            regime_adjusted_score = self._adjust_for_regime(weighted_score, market_regime, timeframe)
            
            return regime_adjusted_score
            
        except Exception as e:
            self.logger.warning(f"Timeframe score calculation failed: {e}")
            return 0.5
    
    async def _assess_volatility_suitability(self, df: pd.DataFrame, timeframe: str, market_regime) -> Tuple[float, float]:
        """Assess if volatility is suitable for the timeframe"""
        try:
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Ideal volatility ranges for different timeframes
            ideal_vol_ranges = {
                '1m': (0.8, 1.5),   # High frequency - needs higher volatility
                '5m': (0.6, 1.2),
                '15m': (0.4, 0.9),
                '1h': (0.3, 0.7),
                '4h': (0.2, 0.5),
                '1d': (0.1, 0.3),
                '1w': (0.05, 0.2)
            }
            
            ideal_range = ideal_vol_ranges.get(timeframe, (0.2, 0.5))
            
            if volatility < ideal_range[0]:
                # Too low volatility
                score = volatility / ideal_range[0]
            elif volatility > ideal_range[1]:
                # Too high volatility
                score = ideal_range[1] / volatility
            else:
                # Ideal volatility
                score = 1.0
            
            # Adjust weight based on regime
            weight = 0.25
            if market_regime.value in ['VOLATILE', 'CRASH']:
                weight = 0.35  # Volatility more important in volatile markets
            
            return score, weight
            
        except Exception as e:
            self.logger.warning(f"Volatility assessment failed: {e}")
            return 0.5, 0.2
    
    async def _assess_trend_clarity(self, df: pd.DataFrame, timeframe: str, market_regime) -> Tuple[float, float]:
        """Assess trend clarity in the timeframe"""
        try:
            prices = df['close'].values
            
            # Calculate multiple moving averages
            ma_short = pd.Series(prices).rolling(5).mean()
            ma_medium = pd.Series(prices).rolling(10).mean()
            ma_long = pd.Series(prices).rolling(20).mean()
            
            # Remove NaN values
            valid_idx = ~(ma_short.isna() | ma_medium.isna() | ma_long.isna())
            ma_short = ma_short[valid_idx]
            ma_medium = ma_medium[valid_idx]
            ma_long = ma_long[valid_idx]
            
            if len(ma_short) < 5:
                return 0.5, 0.2
            
            # Check alignment of moving averages
            alignment_score = 0
            if len(ma_short) > 0 and len(ma_medium) > 0 and len(ma_long) > 0:
                # Positive trend: short > medium > long
                positive_trend = (ma_short.iloc[-1] > ma_medium.iloc[-1] > ma_long.iloc[-1])
                # Negative trend: short < medium < long
                negative_trend = (ma_short.iloc[-1] < ma_medium.iloc[-1] < ma_long.iloc[-1])
                
                if positive_trend or negative_trend:
                    alignment_score = 1.0
                else:
                    # Check if at least two are aligned
                    if (ma_short.iloc[-1] > ma_medium.iloc[-1]) or (ma_medium.iloc[-1] > ma_long.iloc[-1]):
                        alignment_score = 0.7
                    else:
                        alignment_score = 0.3
            
            # R-squared of linear trend
            from scipy import stats
            x = np.arange(len(prices))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
            trend_strength = r_value ** 2
            
            combined_score = (alignment_score * 0.6 + trend_strength * 0.4)
            
            weight = 0.3
            if market_regime.value in ['TRENDING', 'RECOVERY']:
                weight = 0.4  # Trend clarity more important in trending markets
            
            return combined_score, weight
            
        except Exception as e:
            self.logger.warning(f"Trend clarity assessment failed: {e}")
            return 0.5, 0.2
    
    async def _assess_noise_level(self, df: pd.DataFrame, timeframe: str) -> Tuple[float, float]:
        """Assess noise level in the timeframe"""
        try:
            prices = df['close'].values
            returns = np.diff(prices) / prices[:-1]
            
            # Calculate noise using efficiency ratio
            if len(returns) < 2:
                return 0.5, 0.2
            
            # Market efficiency ratio (noise measurement)
            price_changes = np.diff(prices)
            total_movement = np.sum(np.abs(price_changes))
            net_movement = np.abs(prices[-1] - prices[0])
            
            if total_movement > 0:
                efficiency_ratio = net_movement / total_movement
            else:
                efficiency_ratio = 0
            
            # Higher efficiency ratio = less noise
            noise_score = efficiency_ratio
            
            # Ideal noise levels for different timeframes
            # Shorter timeframes can tolerate more noise
            ideal_noise = {
                '1m': 0.3,
                '5m': 0.4,
                '15m': 0.5,
                '1h': 0.6,
                '4h': 0.7,
                '1d': 0.8,
                '1w': 0.9
            }
            
            ideal = ideal_noise.get(timeframe, 0.6)
            adjusted_score = 1.0 - abs(noise_score - ideal)
            
            return max(0, adjusted_score), 0.2
            
        except Exception as e:
            self.logger.warning(f"Noise level assessment failed: {e}")
            return 0.5, 0.2
    
    async def _assess_trading_opportunities(self, df: pd.DataFrame, timeframe: str) -> Tuple[float, float]:
        """Assess number of trading opportunities in the timeframe"""
        try:
            # Count significant price moves
            returns = df['close'].pct_change().dropna()
            
            # Define "significant move" based on timeframe
            significance_thresholds = {
                '1m': 0.001,  # 0.1%
                '5m': 0.002,  # 0.2%
                '15m': 0.005, # 0.5%
                '1h': 0.01,   # 1%
                '4h': 0.02,   # 2%
                '1d': 0.03,   # 3%
                '1w': 0.05    # 5%
            }
            
            threshold = significance_thresholds.get(timeframe, 0.01)
            significant_moves = np.sum(np.abs(returns) > threshold)
            total_periods = len(returns)
            
            if total_periods > 0:
                opportunity_frequency = significant_moves / total_periods
            else:
                opportunity_frequency = 0
            
            # Ideal frequency depends on trading style
            # More frequent for scalping, less for position trading
            ideal_frequencies = {
                '1m': 0.4,   # 40% of periods have significant moves
                '5m': 0.3,
                '15m': 0.25,
                '1h': 0.2,
                '4h': 0.15,
                '1d': 0.1,
                '1w': 0.05
            }
            
            ideal = ideal_frequencies.get(timeframe, 0.2)
            score = 1.0 - min(1.0, abs(opportunity_frequency - ideal) / ideal)
            
            return score, 0.25
            
        except Exception as e:
            self.logger.warning(f"Trading opportunities assessment failed: {e}")
            return 0.5, 0.2
    
    def _adjust_for_regime(self, score: float, market_regime, timeframe: str) -> float:
        """Adjust timeframe score based on market regime"""
        # Different regimes favor different timeframes
        regime_adjustments = {
            'TRENDING': {
                '1h': 1.1, '4h': 1.2, '1d': 1.1, '1w': 1.0,
                '15m': 0.9, '5m': 0.8, '1m': 0.7
            },
            'RANGING': {
                '15m': 1.1, '1h': 1.0, '4h': 0.9, '1d': 0.8,
                '5m': 1.2, '1m': 1.1, '1w': 0.7
            },
            'VOLATILE': {
                '1h': 1.0, '4h': 1.1, '1d': 0.9, '1w': 0.8,
                '15m': 0.9, '5m': 0.8, '1m': 0.6
            },
            'CRASH': {
                '1d': 1.2, '4h': 1.1, '1w': 1.0, '1h': 0.9,
                '15m': 0.7, '5m': 0.6, '1m': 0.5
            },
            'RECOVERY': {
                '4h': 1.1, '1d': 1.2, '1w': 1.1, '1h': 1.0,
                '15m': 0.8, '5m': 0.7, '1m': 0.6
            }
        }
        
        adjustment = regime_adjustments.get(market_regime.value, {}).get(timeframe, 1.0)
        adjusted_score = score * adjustment
        
        return min(1.0, adjusted_score)
    
    async def analyze_symbol(self, symbol: str, timeframe: str, market_regime) -> Optional[Dict]:
        """Analyze specific symbol-timeframe combination"""
        # This agent primarily focuses on timeframe optimization
        # For individual symbol analysis, it defers to other agents
        return None
    
    async def analyze_market_regime(self):
        """Analyze market regime from timeframe perspective"""
        from .master_orchestrator import MarketRegime
        
        # Timeframe-based regime detection
        # If short timeframes are performing well -> ranging/volatile
        # If long timeframes are performing well -> trending
        
        return MarketRegime.TRENDING
