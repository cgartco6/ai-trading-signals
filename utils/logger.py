"""
Logging Configuration for AI Trading System
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional

def setup_logging(level: int = logging.INFO, 
                 log_file: Optional[str] = None,
                 max_bytes: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        level: Logging level
        log_file: Path to log file (optional)
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
    
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    if log_file and not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Default log file if not specified
    if log_file is None:
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'trading_system_{timestamp}.log')
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Set specific log levels for noisy libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    
    logger = get_logger(__name__)
    logger.info(f"Logging configured. Level: {logging.getLevelName(level)}")
    logger.info(f"Log file: {log_file}")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get logger with given name
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

class PerformanceLogger:
    """Specialized logger for performance metrics"""
    
    def __init__(self, name: str = 'performance'):
        self.logger = get_logger(name)
        self.metrics = {}
    
    def log_signal_performance(self, signal_id: str, symbol: str, action: str, 
                             confidence: float, outcome: Optional[bool] = None):
        """Log trading signal performance"""
        log_data = {
            'signal_id': signal_id,
            'symbol': symbol,
            'action': action,
            'confidence': confidence,
            'outcome': outcome,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"SIGNAL_PERFORMANCE: {log_data}")
        
        # Store for aggregation
        if symbol not in self.metrics:
            self.metrics[symbol] = []
        self.metrics[symbol].append(log_data)
    
    def log_agent_performance(self, agent_name: str, accuracy: float, 
                            total_signals: int, successful_signals: int):
        """Log agent performance metrics"""
        log_data = {
            'agent': agent_name,
            'accuracy': accuracy,
            'total_signals': total_signals,
            'successful_signals': successful_signals,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"AGENT_PERFORMANCE: {log_data}")
    
    def log_system_metrics(self, cpu_usage: float, memory_usage: float, 
                          active_tasks: int, signal_count: int):
        """Log system performance metrics"""
        log_data = {
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'active_tasks': active_tasks,
            'signal_count': signal_count,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"SYSTEM_METRICS: {log_data}")
    
    def get_performance_summary(self) -> dict:
        """Get performance summary across all signals"""
        summary = {
            'total_signals': 0,
            'successful_signals': 0,
            'accuracy': 0.0,
            'by_symbol': {}
        }
        
        for symbol, signals in self.metrics.items():
            symbol_signals = len(signals)
            successful = sum(1 for s in signals if s.get('outcome') is True)
            accuracy = successful / symbol_signals if symbol_signals > 0 else 0
            
            summary['by_symbol'][symbol] = {
                'total_signals': symbol_signals,
                'successful_signals': successful,
                'accuracy': accuracy
            }
            
            summary['total_signals'] += symbol_signals
            summary['successful_signals'] += successful
        
        if summary['total_signals'] > 0:
            summary['accuracy'] = summary['successful_signals'] / summary['total_signals']
        
        return summary

# Example usage
if __name__ == "__main__":
    # Setup logging
    logger = setup_logging(level=logging.DEBUG)
    
    # Test different log levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Test performance logger
    perf_logger = PerformanceLogger()
    perf_logger.log_signal_performance(
        signal_id="test_123",
        symbol="BTC/USDT",
        action="BUY",
        confidence=0.85,
        outcome=True
    )
    
    # Print performance summary
    summary = perf_logger.get_performance_summary()
    print("Performance Summary:", summary)
