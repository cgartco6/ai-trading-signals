"""
Pair Selection Agent - Dynamically selects best trading pairs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import asyncio
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class PairSelectionAgent:
    """AI agent for dynamic pair selection based on market conditions"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.available_pairs = self._initialize_pairs()
        self.pair_metrics = {}
        self.correlation_matrix = None
        
    def _initialize_pairs(self) -> Dict[str, List[str]]:
        """Initialize available trading pairs"""
        return {
            'forex': [
                'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF',
                'AUD/USD', 'USD/CAD', 'NZD/USD', 'EUR/GBP',
                'EUR/JPY', 'GBP/JPY', 'AUD/JPY', 'EUR/AUD'
            ],
            'crypto': [
                'BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'DOT/USDT',
                'LINK/USDT', 'LTC/USDT', 'BCH/USDT', 'XRP/USDT',
                'EOS/USDT', 'XTZ/USDT', 'ATOM/USDT', 'SOL/USDT'
            ],
            'indices': [
                'SPX', 'NDX', 'DJI', 'FTSE', 'DAX', 'CAC', 'NIKKEI'
            ]
        }
    
    async def select_pairs(self, market_regime) -> List[Tuple[str, float]]:
        """Select optimal trading pairs for current market regime"""
        try:
            all_pairs = self.available_pairs['forex'] + self.available_pairs['crypto']
            
            # Analyze all pairs concurrently
            pair_analysis = await asyncio.gather(
                *[self._analyze_pair(pair, market_regime) for pair in all_pairs],
                return_exceptions=True
            )
            
            # Filter valid results
            valid_pairs = []
            for i, result in enumerate(pair_analysis):
                if not isinstance(result, Exception) and result is not None:
                    valid_pairs.append((all_pairs[i], result['score']))
            
            # Sort by opportunity score (descending)
            valid_pairs.sort(key=lambda x: x[1], reverse=True)
            
            self.logger.info(f"Selected top pairs: {[p[0] for p in valid_pairs[:5]]}")
            
            return valid_pairs
            
        except Exception as e:
            self.logger.error(f"Pair selection failed: {e}")
            # Return default pairs
            return [('BTC/USDT', 0.7), ('EUR/USD', 0.6), ('ETH/USDT', 0.65)]
    
    async def _analyze_pair(self, pair: str, market_regime) -> Optional[Dict]:
        """Analyze a single trading pair"""
        try:
            from ..data.market_data import MarketDataFetcher
            data_fetcher = MarketDataFetcher(self.config)
            
            # Get data for multiple timeframes to assess pair quality
            timeframes = ['1h', '4h', '1d']
            timeframe_data = {}
            
            for tf in timeframes:
                df = await data_fetcher.get_symbol_data(pair, tf, limit=100)
                if df is not None:
                    timeframe_data[tf] = df
            
            if not timeframe_data:
                return None
            
            # Calculate multiple quality metrics
            metrics = await asyncio.gather(
                self._calculate_volatility_quality(timeframe_data, pair),
                self._calculate_trend_quality(timeframe_data, pair),
                self._calculate_liquidity_quality(timeframe_data, pair),
                self._calculate_regime_alignment(timeframe_data, pair, market_regime),
                self._calculate_technical_setup(timeframe_data, pair),
                return_exceptions=True
            )
            
            valid_metrics = [m for m in metrics if not isinstance(m, Exception) and m is not None]
            
            if not valid_metrics:
                return None
            
            # Calculate weighted overall score
            total_score = 0
            total_weight = 0
            
            for metric in valid_metrics:
                weight = metric.get('weight', 1.0)
                total_score += metric['score'] * weight
                total_weight += weight
            
            overall_score = total_score / total_weight if total_weight > 0 else 0
            
            # Store pair metrics for future reference
            self.pair_metrics[pair] = {
                'score': overall_score,
                'timestamp': datetime.now(),
                'metrics': valid_metrics,
                'market_regime': market_regime.value
            }
            
            return {
                'score': overall_score,
                'metrics': valid_metrics,
                'best_timeframe': max(timeframe_data.keys(), 
                                    key=lambda tf: valid_metrics[0].get('timeframe_score', {}).get(tf, 0))
            }
            
        except Exception as e:
            self.logger.warning(f"Pair analysis failed for {pair}: {e}")
            return None
    
    async def _calculate_volatility_quality(self, timeframe_data: Dict, pair: str) -> Dict:
        """Calculate volatility quality score"""
        try:
            scores = {}
            
            for tf, df in timeframe_data.items():
                if df is None or len(df) < 20:
                    scores[tf] = 0
                    continue
                
                returns = df['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # Annualized
                
                # Ideal volatility range for trading
                # Too low -> no movement, too high -> too risky
                ideal_min, ideal_max = 0.15, 0.80  # 15% to 80% annualized
                
                if volatility < ideal_min:
                    score = volatility / ideal_min
                elif volatility > ideal_max:
                    score = ideal_max / volatility
                else:
                    # Within ideal range
                    score = 1.0 - (abs(volatility - (ideal_min + ideal_max)/2) / ((ideal_max - ideal_min)/2))
                
                scores[tf] = score
            
            avg_score = np.mean(list(scores.values())) if scores else 0
            
            return {
                'metric': 'volatility_quality',
                'score': avg_score,
                'weight': 0.25,
                'timeframe_scores': scores
            }
            
        except Exception as e:
            self.logger.warning(f"Volatility quality calculation failed for {pair}: {e}")
            return {'metric': 'volatility_quality', 'score': 0.5, 'weight': 0.25}
    
    async def _calculate_trend_quality(self, timeframe_data: Dict, pair: str) -> Dict:
        """Calculate trend quality score"""
        try:
            scores = {}
            
            for tf, df in timeframe_data.items():
                if df is None or len(df) < 30:
                    scores[tf] = 0
                    continue
                
                prices = df['close'].values
                
                # Calculate trend strength using ADX
                from ..data.feature_engineering import AdvancedFeatureEngine
                features = AdvancedFeatureEngine.create_advanced_features(df)
                
                if 'adx' in features.columns:
                    adx = features['adx'].dropna()
                    if len(adx) > 0:
                        current_adx = adx.iloc[-1]
                        # ADX > 25 indicates strong trend
                        trend_strength = min(1.0, current_adx / 50)
                    else:
                        trend_strength = 0.5
                else:
                    # Fallback: use linear regression R-squared
                    from scipy import stats
                    x = np.arange(len(prices))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
                    trend_strength = r_value ** 2
                
                # Trend consistency (recent vs historical)
                if len(prices) >= 20:
                    short_trend = (prices[-1] - prices[-5]) / prices[-5]
                    medium_trend = (prices[-1] - prices[-10]) / prices[-10]
                    long_trend = (prices[-1] - prices[-20]) / prices[-20]
                    
                    trend_alignment = 1.0 - (np.std([short_trend, medium_trend, long_trend]) / 0.1)
                    trend_alignment = max(0, min(1.0, trend_alignment))
                else:
                    trend_alignment = 0.5
                
                combined_score = (trend_strength * 0.6 + trend_alignment * 0.4)
                scores[tf] = combined_score
            
            avg_score = np.mean(list(scores.values())) if scores else 0
            
            return {
                'metric': 'trend_quality',
                'score': avg_score,
                'weight': 0.30,
                'timeframe_scores': scores
            }
            
        except Exception as e:
            self.logger.warning(f"Trend quality calculation failed for {pair}: {e}")
            return {'metric': 'trend_quality', 'score': 0.5, 'weight': 0.30}
    
    async def _calculate_liquidity_quality(self, timeframe_data: Dict, pair: str) -> Dict:
        """Calculate liquidity quality score"""
        try:
            scores = {}
            
            for tf, df in timeframe_data.items():
                if df is None or len(df) < 10:
                    scores[tf] = 0
                    continue
                
                # Use volume as liquidity proxy
                volumes = df['volume'].values
                avg_volume = np.mean(volumes)
                
                # Volume consistency
                volume_std = np.std(volumes) / avg_volume if avg_volume > 0 else 1
                volume_consistency = 1.0 - min(1.0, volume_std)
                
                # Volume trend (increasing is better)
                if len(volumes) >= 10:
                    recent_volume = np.mean(volumes[-5:])
                    historical_volume = np.mean(volumes[-10:-5])
                    volume_trend = recent_volume / historical_volume if historical_volume > 0 else 1
                    volume_trend_score = min(2.0, volume_trend) / 2.0  # Cap at 2x growth
                else:
                    volume_trend_score = 0.5
                
                # Normalize volume to 0-1 scale (market-specific)
                if 'BTC' in pair or 'ETH' in pair:
                    # Crypto - different scale
                    volume_normalized = min(1.0, avg_volume / 1e6)  # Adjust based on typical volumes
                else:
                    # Forex - different scale
                    volume_normalized = min(1.0, avg_volume / 1e8)  # Adjust based on typical volumes
                
                combined_score = (volume_normalized * 0.4 + 
                                volume_consistency * 0.3 + 
                                volume_trend_score * 0.3)
                
                scores[tf] = combined_score
            
            avg_score = np.mean(list(scores.values())) if scores else 0
            
            return {
                'metric': 'liquidity_quality',
                'score': avg_score,
                'weight': 0.20,
                'timeframe_scores': scores
            }
            
        except Exception as e:
            self.logger.warning(f"Liquidity quality calculation failed for {pair}: {e}")
            return {'metric': 'liquidity_quality', 'score': 0.5, 'weight': 0.20}
    
    async def _calculate_regime_alignment(self, timeframe_data: Dict, pair: str, market_regime) -> Dict:
        """Calculate how well pair aligns with current market regime"""
        try:
            scores = {}
            
            for tf, df in timeframe_data.items():
                if df is None or len(df) < 20:
                    scores[tf] = 0
                    continue
                
                # Different regimes favor different pair characteristics
                if market_regime.value == 'TRENDING':
                    # Trending markets favor pairs with strong, clear trends
                    prices = df['close'].values
                    from scipy import stats
                    x = np.arange(len(prices))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
                    trend_strength = r_value ** 2
                    scores[tf] = trend_strength
                    
                elif market_regime.value == 'RANGING':
                    # Ranging markets favor pairs with mean-reversion characteristics
                    returns = df['close'].pct_change().dropna()
                    # Low autocorrelation suggests mean-reversion
                    if len(returns) > 1:
                        autocorr = np.corrcoef(returns[:-1], returns[1:])[0,1] if len(returns) > 1 else 0
                        mean_reversion_score = 1.0 - min(1.0, abs(autocorr))
                    else:
                        mean_reversion_score = 0.5
                    scores[tf] = mean_reversion_score
                    
                elif market_regime.value == 'VOLATILE':
                    # Volatile markets favor pairs with good risk-reward ratios
                    returns = df['close'].pct_change().dropna()
                    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
                    # Convert to 0-1 score
                    sharpe_score = min(1.0, (sharpe_ratio + 2) / 4)  # Map [-2, 2] to [0, 1]
                    scores[tf] = sharpe_score
                    
                else:  # CRASH or RECOVERY
                    # Safe-haven or risk-on pairs based on regime
                    if 'USD' in pair or 'JPY' in pair:
                        # Safe-haven currencies in crash, risk-off in recovery
                        if market_regime.value == 'CRASH':
                            scores[tf] = 0.8
                        else:
                            scores[tf] = 0.4
                    elif 'BTC' in pair or 'ETH' in pair:
                        # Crypto behaves differently
                        if market_regime.value == 'RECOVERY':
                            scores[tf] = 0.8
                        else:
                            scores[tf] = 0.4
                    else:
                        scores[tf] = 0.5
            
            avg_score = np.mean(list(scores.values())) if scores else 0
            
            return {
                'metric': 'regime_alignment',
                'score': avg_score,
                'weight': 0.15,
                'timeframe_scores': scores
            }
            
        except Exception as e:
            self.logger.warning(f"Regime alignment calculation failed for {pair}: {e}")
            return {'metric': 'regime_alignment', 'score': 0.5, 'weight': 0.15}
    
    async def _calculate_technical_setup(self, timeframe_data: Dict, pair: str) -> Dict:
        """Calculate technical setup quality"""
        try:
            scores = {}
            
            for tf, df in timeframe_data.items():
                if df is None or len(df) < 30:
                    scores[tf] = 0
                    continue
                
                from ..data.feature_engineering import AdvancedFeatureEngine
                features = AdvancedFeatureEngine.create_advanced_features(df)
                
                setup_score = 0.5
                indicator_count = 0
                
                # Check multiple technical indicators
                if 'rsi' in features.columns:
                    rsi = features['rsi'].iloc[-1]
                    if not np.isnan(rsi):
                        # RSI not in extreme zones is better
                        rsi_extreme = 1.0 - (abs(rsi - 50) / 50)
                        setup_score += rsi_extreme
                        indicator_count += 1
                
                if 'macd' in features.columns and 'macd_signal' in features.columns:
                    macd = features['macd'].iloc[-1]
                    macd_signal = features['macd_signal'].iloc[-1]
                    if not np.isnan(macd) and not np.isnan(macd_signal):
                        # MACD crossover potential
                        macd_strength = abs(macd - macd_signal) / (abs(macd) + 0.001)
                        setup_score += min(1.0, macd_strength * 5)
                        indicator_count += 1
                
                if 'bb_upper' in features.columns and 'bb_lower' in features.columns:
                    current_price = df['close'].iloc[-1]
                    bb_upper = features['bb_upper'].iloc[-1]
                    bb_lower = features['bb_lower'].iloc[-1]
                    if not np.isnan(bb_upper) and not np.isnan(bb_lower):
                        # Not at Bollinger Band extremes
                        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                        bb_extreme = 1.0 - (abs(bb_position - 0.5) * 2)
                        setup_score += bb_extreme
                        indicator_count += 1
                
                if indicator_count > 0:
                    final_score = setup_score / indicator_count
                else:
                    final_score = 0.5
                
                scores[tf] = final_score
            
            avg_score = np.mean(list(scores.values())) if scores else 0
            
            return {
                'metric': 'technical_setup',
                'score': avg_score,
                'weight': 0.10,
                'timeframe_scores': scores
            }
            
        except Exception as e:
            self.logger.warning(f"Technical setup calculation failed for {pair}: {e}")
            return {'metric': 'technical_setup', 'score': 0.5, 'weight': 0.10}
    
    async def analyze_symbol(self, symbol: str, timeframe: str, market_regime) -> Optional[Dict]:
        """Analyze specific symbol for trading suitability"""
        # This agent focuses on pair selection, not individual signal generation
        return None
    
    async def analyze_market_regime(self):
        """Analyze market regime from pair selection perspective"""
        from .master_orchestrator import MarketRegime
        
        # Analyze which types of pairs are performing best
        # Forex pairs vs Crypto pairs, etc.
        
        return MarketRegime.TRENDING
    
    def get_pair_recommendations(self, count: int = 5) -> List[Tuple[str, float]]:
        """Get current pair recommendations"""
        recent_pairs = []
        current_time = datetime.now()
        
        for pair, metrics in self.pair_metrics.items():
            # Only consider metrics from last 24 hours
            if (current_time - metrics['timestamp']).total_seconds() < 86400:
                recent_pairs.append((pair, metrics['score']))
        
        # Sort by score and return top pairs
        recent_pairs.sort(key=lambda x: x[1], reverse=True)
        return recent_pairs[:count]
