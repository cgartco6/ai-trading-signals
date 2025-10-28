"""
Security Utilities for Trading System
"""

import hashlib
import hmac
import os
import json
from typing import Optional, Dict, Any
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

class SecurityManager:
    """Manages security aspects of the trading system"""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.secret_key = secret_key or os.getenv('SECRET_KEY')
        
        if not self.secret_key:
            self.logger.warning("No secret key provided, using default")
            self.secret_key = "default-secret-key-change-in-production"
        
        # Initialize encryption
        self.fernet = self._initialize_encryption()
    
    def _initialize_encryption(self) -> Fernet:
        """Initialize encryption system"""
        try:
            # Derive key from secret
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'ai_trading_system',
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.secret_key.encode()))
            return Fernet(key)
        except Exception as e:
            self.logger.error(f"Encryption initialization failed: {e}")
            # Fallback to basic key
            return Fernet(base64.urlsafe_b64encode(self.secret_key.encode().ljust(32)[:32]))
    
    def encrypt_signal(self, signal_data: Dict[str, Any]) -> str:
        """Encrypt trading signal data"""
        try:
            # Convert to JSON string
            data_str = json.dumps(signal_data, sort_keys=True)
            
            # Create signature
            signature = hmac.new(
                self.secret_key.encode(),
                data_str.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Encrypt data
            encrypted_data = self.fernet.encrypt(data_str.encode())
            encrypted_b64 = base64.urlsafe_b64encode(encrypted_data).decode()
            
            return f"{signature}:{encrypted_b64}"
            
        except Exception as e:
            self.logger.error(f"Signal encryption failed: {e}")
            raise
    
    def decrypt_signal(self, encrypted_data: str) -> Optional[Dict[str, Any]]:
        """Decrypt and verify trading signal"""
        try:
            if ':' not in encrypted_data:
                self.logger.error("Invalid encrypted data format")
                return None
            
            signature, encrypted_b64 = encrypted_data.split(':', 1)
            
            # Decrypt data
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_b64)
            decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
            data_str = decrypted_bytes.decode()
            
            # Verify signature
            expected_sig = hmac.new(
                self.secret_key.encode(),
                data_str.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_sig):
                self.logger.error("Signal signature verification failed")
                return None
            
            # Parse JSON
            return json.loads(data_str)
            
        except Exception as e:
            self.logger.error(f"Signal decryption failed: {e}")
            return None
    
    def hash_sensitive_data(self, data: str) -> str:
        """Hash sensitive data"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def validate_api_key(self, api_key: str, expected_hash: str) -> bool:
        """Validate API key against expected hash"""
        try:
            actual_hash = self.hash_sensitive_data(api_key)
            return hmac.compare_digest(actual_hash, expected_hash)
        except Exception as e:
            self.logger.error(f"API key validation failed: {e}")
            return False
    
    def create_secure_token(self, data: Dict[str, Any], expires_in: int = 3600) -> str:
        """Create secure token with expiration"""
        try:
            import time
            
            # Add expiration timestamp
            data['expires'] = int(time.time()) + expires_in
            data['created'] = int(time.time())
            
            return self.encrypt_signal(data)
            
        except Exception as e:
            self.logger.error(f"Token creation failed: {e}")
            raise
    
    def validate_secure_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate secure token and check expiration"""
        try:
            import time
            
            data = self.decrypt_signal(token)
            if not data:
                return None
            
            # Check expiration
            if data.get('expires', 0) < time.time():
                self.logger.warning("Token has expired")
                return None
            
            return data
            
        except Exception as e:
            self.logger.error(f"Token validation failed: {e}")
            return None
    
    def sanitize_user_input(self, input_str: str) -> str:
        """Sanitize user input to prevent injection attacks"""
        import html
        
        # Basic HTML escaping
        sanitized = html.escape(input_str)
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '&', "'", '"', ';', '|', '`', '$']
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        return sanitized.strip()
    
    def check_password_strength(self, password: str) -> Dict[str, Any]:
        """Check password strength"""
        result = {
            'is_strong': False,
            'score': 0,
            'feedback': []
        }
        
        # Length check
        if len(password) >= 12:
            result['score'] += 2
        elif len(password) >= 8:
            result['score'] += 1
        else:
            result['feedback'].append("Password should be at least 8 characters long")
        
        # Complexity checks
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(not c.isalnum() for c in password)
        
        if has_upper:
            result['score'] += 1
        else:
            result['feedback'].append("Add uppercase letters")
        
        if has_lower:
            result['score'] += 1
        else:
            result['feedback'].append("Add lowercase letters")
        
        if has_digit:
            result['score'] += 1
        else:
            result['feedback'].append("Add numbers")
        
        if has_special:
            result['score'] += 1
        else:
            result['feedback'].append("Add special characters")
        
        # Common password check (simplified)
        common_passwords = ['password', '123456', 'qwerty', 'admin']
        if password.lower() in common_passwords:
            result['score'] = 0
            result['feedback'].append("Password is too common")
        
        result['is_strong'] = result['score'] >= 5
        
        return result

class RateLimiter:
    """Rate limiting for API calls and signal generation"""
    
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.logger = logging.getLogger(__name__)
    
    def is_allowed(self) -> bool:
        """Check if request is allowed"""
        import time
        
        current_time = time.time()
        
        # Remove old requests
        self.requests = [t for t in self.requests if t > current_time - self.time_window]
        
        # Check if under limit
        if len(self.requests) < self.max_requests:
            self.requests.append(current_time)
            return True
        else:
            self.logger.warning("Rate limit exceeded")
            return False
    
    def get_wait_time(self) -> float:
        """Get time until next allowed request"""
        import time
        
        if not self.requests:
            return 0
        
        current_time = time.time()
        oldest_request = min(self.requests)
        
        if len(self.requests) < self.max_requests:
            return 0
        
        return max(0, oldest_request + self.time_window - current_time)

# Example usage
if __name__ == "__main__":
    # Test security manager
    security = SecurityManager("test-secret-key")
    
    # Test signal encryption
    test_signal = {
        'symbol': 'BTC/USDT',
        'action': 'BUY',
        'confidence': 0.85,
        'timestamp': '2023-01-01T00:00:00Z'
    }
    
    encrypted = security.encrypt_signal(test_signal)
    print(f"Encrypted signal: {encrypted}")
    
    decrypted = security.decrypt_signal(encrypted)
    print(f"Decrypted signal: {decrypted}")
    
    # Test rate limiter
    limiter = RateLimiter(max_requests=5, time_window=60)  # 5 requests per minute
    
    for i in range(10):
        allowed = limiter.is_allowed()
        print(f"Request {i+1}: {'Allowed' if allowed else 'Rate limited'}")
