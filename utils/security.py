"""
Security utilities for the trading system
"""

import hashlib
import hmac
import os
from typing import Optional
import logging

class SecurityManager:
    """Manages security aspects of the trading system"""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or os.getenv('SECRET_KEY')
        if not self.secret_key:
            logging.warning("No secret key provided, using default")
            self.secret_key = "default-secret-key-change-in-production"
            
    def encrypt_signal(self, signal_data: dict) -> str:
        """Encrypt trading signal data"""
        import json
        data_str = json.dumps(signal_data, sort_keys=True)
        signature = hmac.new(
            self.secret_key.encode(),
            data_str.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"{signature}:{data_str}"
    
    def verify_signal(self, encrypted_data: str) -> Optional[dict]:
        """Verify and decrypt trading signal"""
        try:
            signature, data_str = encrypted_data.split(':', 1)
            expected_sig = hmac.new(
                self.secret_key.encode(),
                data_str.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if hmac.compare_digest(signature, expected_sig):
                import json
                return json.loads(data_str)
        except Exception as e:
            logging.error(f"Signal verification failed: {e}")
            return None
