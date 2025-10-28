"""
Configuration Loader with Environment-Specific Settings
"""

import yaml
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import logging

load_dotenv()

class Config:
    """Configuration manager for AI trading system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self._validate_config()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if config_path is None:
            env = os.getenv('ENVIRONMENT', 'development')
            config_path = f'environments/{env}.yaml'
            
            if not os.path.exists(config_path):
                # Fallback to local i7 setup
                config_path = 'environments/local_i7_setup.yaml'
        
        self.logger.info(f"Loading configuration from: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults")
            config = self._get_default_config()
        
        # Override with environment variables
        self._override_with_env_vars(config)
        
        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'system': {
                'max_workers': 4,
                'memory_limit': '12GB',
                'gpu_enabled': False,
                'precision': 'mixed'
            },
            'trading': {
                'symbols_per_batch': 8,
                'update_interval': 300,
                'max_concurrent_pairs': 6,
                'confidence_threshold': 0.75,
                'risk_per_trade': 0.02
            },
            'models': {
                'batch_size': 32,
                'model_parallelism': True,
                'cache_models': True,
                'quantization': True
            },
            'data': {
                'cache_dir': './data/cache',
                'max_cache_size': '50GB',
                'preload_data': True
            },
            'performance': {
                'use_async': True,
                'max_pending_tasks': 10,
                'timeout': 30
            },
            'telegram': {
                'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
                'chat_id': os.getenv('TELEGRAM_CHAT_ID', '')
            }
        }
    
    def _override_with_env_vars(self, config: Dict[str, Any]):
        """Override config with environment variables"""
        # System settings
        if 'SYSTEM_MAX_WORKERS' in os.environ:
            config['system']['max_workers'] = int(os.environ['SYSTEM_MAX_WORKERS'])
        
        if 'SYSTEM_MEMORY_LIMIT' in os.environ:
            config['system']['memory_limit'] = os.environ['SYSTEM_MEMORY_LIMIT']
        
        # Trading settings
        if 'TRADING_UPDATE_INTERVAL' in os.environ:
            config['trading']['update_interval'] = int(os.environ['TRADING_UPDATE_INTERVAL'])
        
        if 'TRADING_CONFIDENCE_THRESHOLD' in os.environ:
            config['trading']['confidence_threshold'] = float(os.environ['TRADING_CONFIDENCE_THRESHOLD'])
        
        # Telegram settings
        if 'TELEGRAM_BOT_TOKEN' in os.environ:
            config['telegram']['bot_token'] = os.environ['TELEGRAM_BOT_TOKEN']
        
        if 'TELEGRAM_CHAT_ID' in os.environ:
            config['telegram']['chat_id'] = os.environ['TELEGRAM_CHAT_ID']
    
    def _validate_config(self):
        """Validate configuration"""
        required_sections = ['system', 'trading', 'models', 'telegram']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate Telegram configuration
        telegram_config = self.config.get('telegram', {})
        if not telegram_config.get('bot_token') or not telegram_config.get('chat_id'):
            self.logger.warning("Telegram bot token or chat ID not configured")
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def update(self, key: str, value: Any):
        """Update configuration value"""
        keys = key.split('.')
        config_ref = self.config
        
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        
        config_ref[keys[-1]] = value
    
    def save(self, config_path: Optional[str] = None):
        """Save configuration to file"""
        if config_path is None:
            env = os.getenv('ENVIRONMENT', 'development')
            config_path = f'environments/{env}.yaml'
        
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            self.logger.info(f"Configuration saved to: {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")

# Example usage
if __name__ == "__main__":
    # Test configuration loading
    config = Config('environments/local_i7_setup.yaml')
    
    print("System max workers:", config.get('system.max_workers'))
    print("Trading confidence threshold:", config.get('trading.confidence_threshold'))
    print("Telegram bot token configured:", bool(config.get('telegram.bot_token')))
