"""
Signal Validation Agent - Validates and filters trading signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import asyncio
from ..agents.master_orchestrator import TradingSignal

class SignalValidationAgent:
    """AI agent for validating and filtering trading signals"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.validation_rules = self._initialize_validation_rules()
        self.signal_history = []
        self.performance_metrics = {}
        
    def _initialize_validation_rules(self) -> Dict:
        """Initialize signal validation rules"""
        return {
            'confidence_threshold': 0.70,
            'minimum_agents': 3,
            'maximum_drawdown': 0.05,
            'risk_reward_ratio': 1.5,
            'volume_requirement': 0.7,
            'volatility_limit': 2.0,
            'time_since_last_signal': 300,  # 5 minutes
            'correlation_limit': 0.8
        }
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate a trading signal using multiple criteria"""
        try:
            validation_results = await asyncio.gather(
                self._validate_confidence(signal),
                self._validate_agent_consensus(signal),
                self._validate_risk_reward(signal),
                self._validate_market_conditions(signal),
                self._validate_volume(signal),
                self._validate_volatility(signal),
                self._validate_recency(signal),
                self._validate_correlation(signal),
                return_exceptions=True
            )
            
            # Count successful validations
            successful_validations = 0
            total_validations = 0
            
            for result in validation_results:
                if not isinstance(result, Exception) and result is not None:
                    total_validations += 1
                    if result.get('passed', False):
                        successful_validations += 1
            
            # Require at least 75% of validations to pass
            validation_rate = successful_validations / total_validations if total_validations > 0 else 0
            
            is_valid = validation_rate >= 0.75
            
            if is_valid:
                self.logger.info(f"Signal {signal.signal_id} validated successfully ({validation_rate:.1%})")
            else:
                self.logger.info(f"Signal {signal.signal_id} failed validation ({validation_rate:.1%})")
            
            # Store validation result
            self._store_validation_result(signal, is_valid, validation_results)
            
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Signal validation failed: {e}")
            return False
    
    async def _validate_confidence(self, signal: TradingSignal) -> Dict:
        """Validate signal confidence"""
        try:
            min_confidence = self.validation_rules['confidence_threshold']
            passed = signal.confidence >= min_confidence
            
            return {
                'validation': 'confidence',
                'passed': passed,
                'actual_value': signal.confidence,
                'required_value': min_confidence,
                'weight': 0.20
            }
            
        except Exception as e:
            self.logger.warning(f"Confidence validation failed: {e}")
            return {'validation': 'confidence', 'passed': False, 'weight': 0.20}
    
    async def _validate_agent_consensus(self, signal: TradingSignal) -> Dict:
        """Validate agent consensus"""
        try:
            min_agents = self.validation_rules['minimum_agents']
            agent_count = len(signal.agent_breakdown)
            
            # Count agents that agree with the signal
            agreeing_agents = 0
            for agent_name, prediction in signal.agent_breakdown.items():
                if prediction.get('action') == signal.action:
                    agreeing_agents += 1
            
            consensus_ratio = agreeing_agents / agent_count if agent_count > 0 else 0
            passed = agent_count >= min_agents and consensus_ratio >= 0.6
            
            return {
                'validation': 'agent_consensus',
                'passed': passed,
                'agent_count': agent_count,
                'agreeing_agents': agreeing_agents,
                'consensus_ratio': consensus_ratio,
                'weight': 0.15
            }
            
        except Exception as e:
            self.logger.warning(f"Agent consensus validation failed: {e}")
            return {'validation': 'agent_consensus', 'passed': False, 'weight': 0.15}
    
    async def _validate_risk_reward(self, signal: TradingSignal) -> Dict:
        """Validate risk-reward ratio"""
        try:
            min_risk_reward = self.validation_rules['risk_reward_ratio']
            
            if len(signal.targets) == 0 or signal.stop_loss == 0:
                return {'validation': 'risk_reward', 'passed': False, 'weight': 0.15}
            
            entry = signal.entry_price
            stop_loss = signal.stop_loss
            
            # Calculate risk (distance to stop loss)
            risk = abs(entry - stop_loss)
            
            if risk == 0:
                return {'validation': 'risk_reward', 'passed': False, 'weight': 0.15}
            
            # Calculate average reward (distance to targets)
            rewards = [abs(target - entry) for target in signal.targets]
            avg_reward = np.mean(rewards)
            
            risk_reward_ratio = avg_reward / risk
            passed = risk_reward_ratio >= min_risk_reward
            
            return {
                'validation': 'risk_reward',
                'passed': passed,
                'risk_reward_ratio': risk_reward_ratio,
                'required_ratio': min_risk_reward,
                'weight': 0.15
            }
            
        except Exception as e:
            self.logger.warning(f"Risk-reward validation failed: {e}")
            return {'validation': 'risk_reward', 'passed': False, 'weight': 0.15}
    
    async def _validate_market_conditions(self, signal: TradingSignal) -> Dict:
        """Validate against current market conditions"""
        try:
            from ..data.market_data import MarketDataFetcher
            data_fetcher = MarketDataFetcher(self.config)
            df = await data_fetcher.get_symbol_data(signal.symbol, signal.timeframe, limit=50)
            
            if df is None:
                return {'validation': 'market_conditions', 'passed': True, 'weight': 0.10}  # Pass if no data
            
            current_price = df['close'].iloc[-1]
            
            # Check if price is near significant support/resistance
            from ..data.feature_engineering import AdvancedFeatureEngine
            features = AdvancedFeatureEngine.create_advanced_features(df)
            
            passed = True
            
            if 'bb_upper' in features.columns and 'bb_lower' in features.columns:
                bb_upper = features['bb_upper'].iloc[-1]
                bb_lower = features['bb_lower'].iloc[-1]
                
                # For BUY signals, avoid if price is near upper Bollinger Band
                if signal.action == 'BUY' and current_price > bb_upper * 0.98:
                    passed = False
                # For SELL signals, avoid if price is near lower Bollinger Band
                elif signal.action == 'SELL' and current_price < bb_lower * 1.02:
                    passed = False
            
            # Check RSI extremes
            if 'rsi' in features.columns:
                rsi = features['rsi'].iloc[-1]
                if not np.isnan(rsi):
                    # For BUY signals, avoid if RSI is overbought
                    if signal.action == 'BUY' and rsi > 70:
                        passed = False
                    # For SELL signals, avoid if RSI is oversold
                    elif signal.action == 'SELL' and rsi < 30:
                        passed = False
            
            return {
                'validation': 'market_conditions',
                'passed': passed,
                'weight': 0.10
            }
            
        except Exception as e:
            self.logger.warning(f"Market conditions validation failed: {e}")
            return {'validation': 'market_conditions', 'passed': True, 'weight': 0.10}  # Pass on error
    
    async def _validate_volume(self, signal: TradingSignal) -> Dict:
        """Validate volume conditions"""
        try:
            from ..data.market_data import MarketDataFetcher
            data_fetcher = MarketDataFetcher(self.config)
            df = await data_fetcher.get_symbol_data(signal.symbol, signal.timeframe, limit=20)
            
            if df is None or 'volume' not in df.columns:
                return {'validation': 'volume', 'passed': True, 'weight': 0.10}  # Pass if no volume data
            
            volumes = df['volume'].values
            min_volume_requirement = self.validation_rules['volume_requirement']
            
            # Check if recent volume is sufficient
            recent_volume = volumes[-5:].mean()
            historical_volume = volumes[:-5].mean() if len(volumes) > 5 else recent_volume
            
            if historical_volume > 0:
                volume_ratio = recent_volume / historical_volume
                passed = volume_ratio >= min_volume_requirement
            else:
                passed = recent_volume > 0  # Any volume is sufficient if no history
            
            return {
                'validation': 'volume',
                'passed': passed,
                'volume_ratio': volume_ratio if historical_volume > 0 else 1.0,
                'weight': 0.10
            }
            
        except Exception as e:
            self.logger.warning(f"Volume validation failed: {e}")
            return {'validation': 'volume', 'passed': True, 'weight': 0.10}  # Pass on error
    
    async def _validate_volatility(self, signal: TradingSignal) -> Dict:
        """Validate volatility conditions"""
        try:
            from ..data.market_data import MarketDataFetcher
            data_fetcher = MarketDataFetcher(self.config)
            df = await data_fetcher.get_symbol_data(signal.symbol, signal.timeframe, limit=50)
            
            if df is None:
                return {'validation': 'volatility', 'passed': True, 'weight': 0.05}  # Pass if no data
            
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            max_volatility = self.validation_rules['volatility_limit']
            passed = volatility <= max_volatility
            
            return {
                'validation': 'volatility',
                'passed': passed,
                'volatility': volatility,
                'max_volatility': max_volatility,
                'weight': 0.05
            }
            
        except Exception as e:
            self.logger.warning(f"Volatility validation failed: {e}")
            return {'validation': 'volatility', 'passed': True, 'weight': 0.05}  # Pass on error
    
    async def _validate_recency(self, signal: TradingSignal) -> Dict:
        """Validate signal recency"""
        try:
            min_time_gap = self.validation_rules['time_since_last_signal']
            
            # Check if we recently sent a signal for this symbol
            recent_signals = [
                s for s in self.signal_history 
                if s.symbol == signal.symbol 
                and (datetime.now() - s.timestamp).total_seconds() < min_time_gap
            ]
            
            passed = len(recent_signals) == 0
            
            return {
                'validation': 'recency',
                'passed': passed,
                'recent_signals_count': len(recent_signals),
                'weight': 0.05
            }
            
        except Exception as e:
            self.logger.warning(f"Recency validation failed: {e}")
            return {'validation': 'recency', 'passed': True, 'weight': 0.05}  # Pass on error
    
    async def _validate_correlation(self, signal: TradingSignal) -> Dict:
        """Validate correlation with existing positions"""
        try:
            max_correlation = self.validation_rules['correlation_limit']
            
            # This would check correlation with existing portfolio positions
            # For now, we'll return passed as we don't have portfolio data
            # In production, this would integrate with portfolio management
            
            return {
                'validation': 'correlation',
                'passed': True,  # Pass for now
                'weight': 0.05
            }
            
        except Exception as e:
            self.logger.warning(f"Correlation validation failed: {e}")
            return {'validation': 'correlation', 'passed': True, 'weight': 0.05}  # Pass on error
    
    def _store_validation_result(self, signal: TradingSignal, is_valid: bool, validation_results: List):
        """Store validation results for performance tracking"""
        validation_record = {
            'signal_id': signal.signal_id,
            'symbol': signal.symbol,
            'action': signal.action,
            'timestamp': datetime.now(),
            'is_valid': is_valid,
            'confidence': signal.confidence,
            'validation_results': validation_results
        }
        
        self.signal_history.append(validation_record)
        
        # Keep only recent history (last 1000 signals)
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
    
    async def analyze_symbol(self, symbol: str, timeframe: str, market_regime) -> Optional[Dict]:
        """Analyze symbol for validation criteria"""
        # This agent focuses on signal validation, not signal generation
        return None
    
    async def analyze_market_regime(self):
        """Analyze market regime from validation perspective"""
        from .master_orchestrator import MarketRegime
        
        # Analyze validation success rates by regime
        if len(self.signal_history) < 10:
            return MarketRegime.TRENDING
        
        recent_signals = [s for s in self.signal_history 
                         if (datetime.now() - s['timestamp']).total_seconds() < 86400]  # Last 24 hours
        
        if not recent_signals:
            return MarketRegime.TRENDING
        
        validation_rate = sum(1 for s in recent_signals if s['is_valid']) / len(recent_signals)
        
        # Low validation rates might indicate volatile or crashing markets
        if validation_rate < 0.3:
            return MarketRegime.CRASH
        elif validation_rate < 0.6:
            return MarketRegime.VOLATILE
        else:
            return MarketRegime.TRENDING
    
    def get_validation_stats(self) -> Dict:
        """Get validation performance statistics"""
        if not self.signal_history:
            return {}
        
        recent_signals = [s for s in self.signal_history 
                         if (datetime.now() - s['timestamp']).total_seconds() < 604800]  # Last week
        
        if not recent_signals:
            return {}
        
        total_signals = len(recent_signals)
        valid_signals = sum(1 for s in recent_signals if s['is_valid'])
        validation_rate = valid_signals / total_signals
        
        # Average confidence
        avg_confidence = np.mean([s['confidence'] for s in recent_signals])
        
        # Validation failure reasons
        failure_reasons = {}
        for signal in recent_signals:
            if not signal['is_valid']:
                for validation in signal['validation_results']:
                    if (not isinstance(validation, Exception) and 
                        validation is not None and 
                        not validation.get('passed', True)):
                        reason = validation.get('validation', 'unknown')
                        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        return {
            'total_signals_week': total_signals,
            'validation_rate': validation_rate,
            'average_confidence': avg_confidence,
            'failure_reasons': failure_reasons
        }
