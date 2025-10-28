"""
Configuration loader with environment-specific settings
"""

import yaml
import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration manager for AI trading system"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self._validate_config()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if config_path is None:
            env = os.getenv('ENVIRONMENT', 'development')
            config_path = f'environments/{env}.yaml'
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Override with environment variables
        self._override_with_env_vars(config)
        return config
    
    def _override_with_env_vars(self, config: Dict):
        """Override config with environment variables"""
        if 'TELEGRAM_BOT_TOKEN' in os.environ:
            config['telegram']['bot_token'] = os.environ['TELEGRAM_BOT_TOKEN']
        if 'TELEGRAM_CHAT_ID' in os.environ:
            config['telegram']['chat_id'] = os.environ['TELEGRAM_CHAT_ID']
            
    def _validate_config(self):
        """Validate configuration"""
        required_keys = ['trading', 'models', 'performance']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config section: {key}")
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default
