"""
Master AI Orchestrator - Coordinates all AI agents
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import uuid

class MarketRegime(Enum):
    TRENDING = "trending"
    RANGING = "ranging" 
    VOLATILE = "volatile"
    CRASH = "crash"
    RECOVERY = "recovery"

@dataclass
class TradingSignal:
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    timeframe: str
    entry_price: float
    targets: List[float]
    stop_loss: float
    timestamp: datetime
    signal_id: str
    agent_breakdown: Dict
    market_regime: MarketRegime

class MasterAIOrchestrator:
    """Main orchestrator that manages all AI agents"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.agents = {}
        self.market_regime = MarketRegime.TRENDING
        self.performance_metrics = {}
        self.signal_history = []
        self._initialize_agents()
        self.logger = logging.getLogger(__name__)
        
    def _initialize_agents(self):
        """Initialize all AI trading agents"""
        try:
            from .deep_predictive_agent import DeepPredictiveAgent
            from .sentiment_analyzer import SentimentIntelligenceAgent
            from .pattern_recognizer import PatternRecognitionAgent
            from .risk_orchestrator import RiskIntelligenceAgent
            from .market_psychologist import MarketPsychologyAgent
            from .quantum_analyzer import QuantumAnalysisAgent
            from .timeframe_optimizer import TimeframeOptimizationAgent
            from .pair_selector import PairSelectionAgent
            from .signal_validator import SignalValidationAgent
            
            self.agents = {
                'deep_predictor': DeepPredictiveAgent(self.config),
                'sentiment_analyzer': SentimentIntelligenceAgent(self.config),
                'pattern_recognizer': PatternRecognitionAgent(self.config),
                'risk_orchestrator': RiskIntelligenceAgent(self.config),
                'market_psychologist': MarketPsychologyAgent(self.config),
                'quantum_analyzer': QuantumAnalysisAgent(self.config),
                'timeframe_optimizer': TimeframeOptimizationAgent(self.config),
                'pair_selector': PairSelectionAgent(self.config),
                'signal_validator': SignalValidationAgent(self.config)
            }
            self.logger.info("All AI agents initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {e}")
            raise
        
    async def analyze_market_regime(self) -> MarketRegime:
        """Analyze current market regime using multiple agents"""
        regime_scores = {
            MarketRegime.TRENDING: 0,
            MarketRegime.RANGING: 0,
            MarketRegime.VOLATILE: 0,
            MarketRegime.CRASH: 0,
            MarketRegime.RECOVERY: 0
        }
        
        # Get regime analysis from relevant agents
        regime_agents = ['deep_predictor', 'sentiment_analyzer', 'market_psychologist']
        
        for agent_name in regime_agents:
            if agent_name in self.agents:
                try:
                    regime = await self.agents[agent_name].analyze_market_regime()
                    regime_scores[regime] += 1
                except Exception as e:
                    self.logger.warning(f"Agent {agent_name} regime analysis failed: {e}")
        
        # Determine dominant regime
        self.market_regime = max(regime_scores, key=regime_scores.get)
        return self.market_regime
    
    async def generate_signals(self) -> List[TradingSignal]:
        """Generate trading signals using all agents"""
        signals = []
        
        try:
            # Get current market regime
            await self.analyze_market_regime()
            self.logger.info(f"Current market regime: {self.market_regime.value}")
            
            # Get optimal pairs for current regime
            optimal_pairs = await self.agents['pair_selector'].select_pairs(self.market_regime)
            self.logger.info(f"Selected pairs: {[p[0] for p in optimal_pairs[:5]]}")
            
            # Get optimal timeframe
            optimal_timeframe = await self.agents['timeframe_optimizer'].get_optimal_timeframe(self.market_regime)
            
            # Generate signals for each pair
            for symbol, score in optimal_pairs[:8]:  # Top 8 pairs
                if score > 0.6:  # Minimum opportunity score
                    signal = await self._analyze_symbol_signal(symbol, optimal_timeframe)
                    if signal and signal.confidence > 0.7:
                        signals.append(signal)
            
            # Validate signals
            validated_signals = []
            for signal in signals:
                is_valid = await self.agents['signal_validator'].validate_signal(signal)
                if is_valid:
                    validated_signals.append(signal)
            
            self.signal_history.extend(validated_signals)
            self.logger.info(f"Generated {len(validated_signals)} validated signals")
            
            return validated_signals
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            return []
    
    async def _analyze_symbol_signal(self, symbol: str, timeframe: str) -> Optional[TradingSignal]:
        """Analyze a single symbol using all agents"""
        agent_predictions = {}
        
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'analyze_symbol'):
                try:
                    prediction = await agent.analyze_symbol(symbol, timeframe, self.market_regime)
                    if prediction:
                        agent_predictions[agent_name] = prediction
                except Exception as e:
                    self.logger.warning(f"Agent {agent_name} analysis failed for {symbol}: {e}")
        
        if not agent_predictions:
            return None
            
        # Ensemble the predictions
        ensemble_result = self._ensemble_predictions(agent_predictions)
        
        if ensemble_result['action'] != 'HOLD':
            return TradingSignal(
                symbol=symbol,
                action=ensemble_result['action'],
                confidence=ensemble_result['confidence'],
                timeframe=timeframe,
                entry_price=ensemble_result.get('entry_price', 0),
                targets=ensemble_result.get('targets', []),
                stop_loss=ensemble_result.get('stop_loss', 0),
                timestamp=datetime.now(),
                signal_id=f"signal_{uuid.uuid4().hex[:8]}",
                agent_breakdown=agent_predictions,
                market_regime=self.market_regime
            )
        
        return None
    
    def _ensemble_predictions(self, predictions: Dict) -> Dict:
        """Combine predictions from all AI agents using weighted averaging"""
        action_weights = {
            'BUY': 0,
            'SELL': 0,
            'HOLD': 0
        }
        
        # Agent weights (configurable)
        agent_weights = {
            'deep_predictor': 0.25,
            'sentiment_analyzer': 0.15,
            'pattern_recognizer': 0.20,
            'risk_orchestrator': 0.15,
            'market_psychologist': 0.10,
            'quantum_analyzer': 0.15
        }
        
        total_confidence = 0
        
        for agent_name, prediction in predictions.items():
            weight = agent_weights.get(agent_name, 0.1)
            confidence = prediction.get('confidence', 0)
            action = prediction.get('action', 'HOLD')
            
            action_weights[action] += confidence * weight
            total_confidence += confidence * weight
        
        # Normalize if we have predictions
        if total_confidence > 0:
            for action in action_weights:
                action_weights[action] /= total_confidence
        
        # Determine final action
        final_action = max(action_weights, key=action_weights.get)
        final_confidence = action_weights[final_action]
        
        # Calculate aggregate price levels
        entry_prices = [p.get('entry_price', 0) for p in predictions.values() if p.get('entry_price')]
        targets_list = [p.get('targets', []) for p in predictions.values() if p.get('targets')]
        stop_losses = [p.get('stop_loss', 0) for p in predictions.values() if p.get('stop_loss')]
        
        return {
            'action': final_action,
            'confidence': final_confidence,
            'entry_price': np.mean(entry_prices) if entry_prices else 0,
            'targets': list(np.mean(targets_list, axis=0)) if targets_list else [],
            'stop_loss': np.mean(stop_losses) if stop_losses else 0,
            'timestamp': datetime.now()
        }
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for all agents"""
        return self.performance_metrics
    
    async def update_agent_weights(self, performance_data: Dict):
        """Update agent weights based on performance"""
        # Implement adaptive weight updating based on historical performance
        pass
