"""
Risk Intelligence Agent for Position Sizing and Risk Management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import scipy.stats as stats

class RiskIntelligenceAgent:
    """AI agent for risk management and position sizing"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.portfolio = {}
        self.risk_per_trade = config.get('risk_per_trade', 0.02)  # 2% per trade
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.10)  # 10% total
        self.correlation_matrix = None
    
    async def analyze_symbol(self, symbol: str, timeframe: str, market_regime) -> Optional[Dict]:
        """Analyze risk for a trading signal"""
        try:
            from ..data.market_data import MarketDataFetcher
            data_fetcher = MarketDataFetcher(self.config)
            df = await data_fetcher.get_symbol_data(symbol, timeframe, limit=100)
            
            if df is None:
                return None
            
            # Calculate various risk metrics
            risk_metrics = await asyncio.gather(
                self._calculate_volatility_risk(df, symbol),
                self._calculate_drawdown_risk(df),
                self._calculate_liquidity_risk(symbol),
                self._calculate_correlation_risk(symbol),
                return_exceptions=True
            )
            
            # Filter valid metrics
            valid_metrics = [m for m in risk_metrics if not isinstance(m, Exception) and m is not None]
            
            if not valid_metrics:
                return None
            
            # Calculate overall risk score (0-1, where 1 is highest risk)
            overall_risk = np.mean([m['risk_score'] for m in valid_metrics])
            
            # Convert risk score to confidence and action
            action, confidence = self._risk_to_action(overall_risk, market_regime)
            
            if action != 'HOLD':
                # Calculate position size
                position_size = self._calculate_position_size(symbol, overall_risk, df)
                
                return {
                    'action': action,
                    'confidence': confidence,
                    'risk_score': overall_risk,
                    'position_size': position_size,
                    'risk_metrics': valid_metrics,
                    'max_drawdown': self._calculate_max_drawdown(df)
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Risk analysis failed for {symbol}: {e}")
            return None
    
    async def _calculate_volatility_risk(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Calculate volatility-based risk"""
        try:
            returns = df['close'].pct_change().dropna()
            
            # Historical volatility (annualized)
            hist_volatility = returns.std() * np.sqrt(252)
            
            # GARCH volatility estimation (simplified)
            garch_vol = self._estimate_garch_volatility(returns)
            
            # Volatility regime detection
            short_term_vol = returns[-20:].std() * np.sqrt(252)
            long_term_vol = returns.std() * np.sqrt(252)
            vol_ratio = short_term_vol / long_term_vol if long_term_vol > 0 else 1
            
            # Risk score based on volatility (0-1 scale)
            vol_risk = min(1.0, hist_volatility / 0.8)  # 80% annual vol as max risk
            
            return {
                'metric': 'volatility',
                'risk_score': vol_risk,
                'historical_volatility': hist_volatility,
                'garch_volatility': garch_vol,
                'volatility_ratio': vol_ratio
            }
            
        except Exception as e:
            self.logger.warning(f"Volatility risk calculation failed: {e}")
            return {'metric': 'volatility', 'risk_score': 0.5}
    
    async def _calculate_drawdown_risk(self, df: pd.DataFrame) -> Dict:
        """Calculate drawdown-based risk"""
        try:
            prices = df['close'].values
            peak = prices[0]
            max_drawdown = 0
            
            for price in prices:
                if price > peak:
                    peak = price
                drawdown = (peak - price) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # Recent drawdown
            recent_peak = np.max(prices[-10:])
            current_price = prices[-1]
            recent_drawdown = (recent_peak - current_price) / recent_peak
            
            # Risk score based on drawdown
            drawdown_risk = min(1.0, max_drawdown / 0.5)  # 50% drawdown as max risk
            
            return {
                'metric': 'drawdown',
                'risk_score': drawdown_risk,
                'max_drawdown': max_drawdown,
                'recent_drawdown': recent_drawdown
            }
            
        except Exception as e:
            self.logger.warning(f"Drawdown risk calculation failed: {e}")
            return {'metric': 'drawdown', 'risk_score': 0.5}
    
    async def _calculate_liquidity_risk(self, symbol: str) -> Dict:
        """Calculate liquidity risk"""
        try:
            # This would typically use volume data and bid-ask spreads
            # For now, use a simplified approach
            from ..data.market_data import MarketDataFetcher
            data_fetcher = MarketDataFetcher(self.config)
            df = await data_fetcher.get_symbol_data(symbol, '1d', limit=30)
            
            if df is None:
                return {'metric': 'liquidity', 'risk_score': 0.5}
            
            # Average volume and volume trend
            avg_volume = df['volume'].mean()
            volume_trend = df['volume'].iloc[-5:].mean() / df['volume'].iloc[-10:-5].mean()
            
            # Liquidity risk (inverse of volume)
            # Normalize volume to 0-1 risk score
            volume_risk = 1.0 - min(1.0, avg_volume / 1e9)  # Adjust scale based on market
            
            # Higher risk if volume is declining
            if volume_trend < 0.8:
                volume_risk = min(1.0, volume_risk * 1.5)
            
            return {
                'metric': 'liquidity',
                'risk_score': volume_risk,
                'average_volume': avg_volume,
                'volume_trend': volume_trend
            }
            
        except Exception as e:
            self.logger.warning(f"Liquidity risk calculation failed: {e}")
            return {'metric': 'liquidity', 'risk_score': 0.5}
    
    async def _calculate_correlation_risk(self, symbol: str) -> Dict:
        """Calculate portfolio correlation risk"""
        try:
            if not self.portfolio:
                return {'metric': 'correlation', 'risk_score': 0.3}
            
            # This would calculate correlation with existing positions
            # For now, return a baseline risk
            return {
                'metric': 'correlation',
                'risk_score': 0.3,
                'portfolio_exposure': len(self.portfolio)
            }
            
        except Exception as e:
            self.logger.warning(f"Correlation risk calculation failed: {e}")
            return {'metric': 'correlation', 'risk_score': 0.3}
    
    def _risk_to_action(self, risk_score: float, market_regime) -> Tuple[str, float]:
        """Convert risk score to trading action"""
        # Adjust risk tolerance based on market regime
        regime_risk_multipliers = {
            'TRENDING': 1.2,    # Higher risk tolerance
            'RANGING': 1.0,     # Normal risk
            'VOLATILE': 0.7,    # Lower risk tolerance
            'CRASH': 0.3,       # Very low risk tolerance
            'RECOVERY': 0.9     # Moderate risk
        }
        
        multiplier = regime_risk_multipliers.get(market_regime.value, 1.0)
        adjusted_risk_threshold = 0.6 * multiplier
        
        if risk_score < adjusted_risk_threshold:
            # Low risk - can trade
            confidence = 1.0 - risk_score
            return 'BUY', confidence
        else:
            # High risk - avoid trading
            return 'HOLD', 0.0
    
    def _calculate_position_size(self, symbol: str, risk_score: float, df: pd.DataFrame) -> float:
        """Calculate optimal position size based on risk"""
        try:
            account_size = self.config.get('account_size', 10000)
            atr = self._calculate_atr(df)
            current_price = df['close'].iloc[-1]
            
            # Adjust risk per trade based on overall risk
            adjusted_risk = self.risk_per_trade * (1 - risk_score)
            
            # Calculate position size using ATR-based stop loss
            risk_amount = account_size * adjusted_risk
            stop_distance = atr * 2  # 2 ATR stop loss
            
            if stop_distance > 0:
                position_size = risk_amount / stop_distance
                # Convert to percentage of account
                position_value = position_size * current_price
                position_pct = position_value / account_size
                
                # Cap position size
                max_position_pct = 0.1  # 10% of account max
                return min(position_pct, max_position_pct)
            
            return 0.02  # Default 2% if calculation fails
            
        except Exception as e:
            self.logger.warning(f"Position size calculation failed: {e}")
            return 0.02
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            true_range = np.maximum(np.maximum(tr1, tr2), tr3)
            atr = true_range.rolling(period).mean().iloc[-1]
            
            return atr if not np.isnan(atr) else df['close'].iloc[-1] * 0.02
            
        except Exception as e:
            self.logger.warning(f"ATR calculation failed: {e}")
            return df['close'].iloc[-1] * 0.02
    
    def _estimate_garch_volatility(self, returns: pd.Series) -> float:
        """Simplified GARCH volatility estimation"""
        # This is a simplified version - in production, use arch package
        try:
            # Use EWMA as simple GARCH approximation
            lambda_param = 0.94
            variances = [returns.var()]
            
            for i in range(1, len(returns)):
                new_var = (lambda_param * variances[-1] + 
                          (1 - lambda_param) * returns.iloc[i-1]**2)
                variances.append(new_var)
            
            return np.sqrt(variances[-1]) * np.sqrt(252)
            
        except Exception as e:
            self.logger.warning(f"GARCH estimation failed: {e}")
            return returns.std() * np.sqrt(252)
    
    def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        try:
            prices = df['close'].values
            peak = prices[0]
            max_dd = 0
            
            for price in prices:
                if price > peak:
                    peak = price
                dd = (peak - price) / peak
                if dd > max_dd:
                    max_dd = dd
            
            return max_dd
            
        except Exception as e:
            self.logger.warning(f"Max drawdown calculation failed: {e}")
            return 0.0
    
    async def analyze_market_regime(self):
        """Analyze market regime from risk perspective"""
        from .master_orchestrator import MarketRegime
        
        # Risk-based regime detection
        # High volatility -> volatile regime
        # High drawdown -> crash regime
        # etc.
        
        return MarketRegime.TRENDING
    
    def update_portfolio(self, new_positions: Dict):
        """Update portfolio with new positions for correlation analysis"""
        self.portfolio.update(new_positions)
