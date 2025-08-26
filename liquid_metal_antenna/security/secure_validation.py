"""
Generation 2 Security Framework
===============================

Comprehensive security validation, input sanitization, authentication,
authorization, and secure configuration management.

Features:
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- Authentication and authorization
- Secure configuration management
- Audit logging
- Rate limiting
- Encryption utilities
"""

import hashlib
import hmac
import secrets
import time
import re
import json
import base64
import os
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
from functools import wraps

# Optional cryptography import
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class SecurityLevel(Enum):
    """Security levels for different operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationResult(Enum):
    """Validation result status"""
    VALID = "valid"
    INVALID = "invalid"
    SUSPICIOUS = "suspicious"
    BLOCKED = "blocked"


@dataclass
class SecurityContext:
    """Security context for operations"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    permissions: List[str] = field(default_factory=list)


class InputValidator:
    """Comprehensive input validation and sanitization"""
    
    def __init__(self):
        # Dangerous patterns to detect
        self.sql_injection_patterns = [
            r"(\b(union|select|insert|update|delete|drop|create|alter)\b)",
            r"(--|\#|\/\*)",
            r"(\b(or|and)\s+\d+\s*=\s*\d+)",
            r"(\bexec\s*\()",
            r"(\bsp_executesql\b)"
        ]
        
        self.xss_patterns = [
            r"<\s*script[^>]*>",
            r"javascript\s*:",
            r"on\w+\s*=\s*['\"][^'\"]*['\"]",
            r"<\s*iframe[^>]*>",
            r"<\s*object[^>]*>",
            r"<\s*embed[^>]*>"
        ]
        
        self.command_injection_patterns = [
            r"(\||&|;|\$\(|\`)",
            r"(\bcat\b|\bls\b|\brm\b|\bmv\b|\bcp\b)",
            r"(\bwget\b|\bcurl\b|\bnc\b|\btelnet\b)",
            r"(\bpython\b|\bperl\b|\brush\b|\bbash\b)"
        ]
        
        self.path_traversal_patterns = [
            r"\.\./",
            r"\.\.\\\\",
            r"%2e%2e%2f",
            r"%2e%2e\\",
            r"\\.\\./"
        ]
        
        # Allowed patterns
        self.allowed_email_pattern = re.compile(
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        )
        
        self.allowed_filename_pattern = re.compile(
            r"^[a-zA-Z0-9._-]+$"
        )
        
        self.allowed_number_pattern = re.compile(
            r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$"
        )
        
        self.logger = logging.getLogger("security.validation")
    
    def validate_input(
        self, 
        data: Any, 
        data_type: str = "string",
        max_length: Optional[int] = None,
        allow_html: bool = False,
        context: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Comprehensive input validation
        
        Args:
            data: Input data to validate
            data_type: Expected data type (string, number, email, filename, etc.)
            max_length: Maximum allowed length
            allow_html: Whether to allow HTML content
            context: Context of the validation for logging
        
        Returns:
            Validation result with sanitized data
        """
        
        result = {
            'valid': True,
            'sanitized_data': data,
            'original_data': data,
            'issues': [],
            'severity': 'none',
            'context': context,
            'validation_time': time.time()
        }
        
        try:
            # Type-specific validation
            if data_type == "string":
                result = self._validate_string(data, max_length, allow_html, result)
            elif data_type == "number":
                result = self._validate_number(data, result)
            elif data_type == "email":
                result = self._validate_email(data, result)
            elif data_type == "filename":
                result = self._validate_filename(data, result)
            elif data_type == "json":
                result = self._validate_json(data, result)
            elif data_type == "dict":
                result = self._validate_dict(data, result)
            elif data_type == "list":
                result = self._validate_list(data, result)
            else:
                # Generic validation
                result = self._validate_generic(data, result)
            
            # Log validation results
            if result['issues']:
                self.logger.warning(
                    f"Validation issues in {context}: {result['issues']}"
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Validation error in {context}: {e}")
            return {
                'valid': False,
                'sanitized_data': None,
                'original_data': data,
                'issues': [f"Validation error: {e}"],
                'severity': 'high',
                'context': context
            }
    
    def _validate_string(
        self, 
        data: Any, 
        max_length: Optional[int], 
        allow_html: bool, 
        result: Dict
    ) -> Dict:
        """Validate string input"""
        
        if not isinstance(data, str):
            try:
                data = str(data)
                result['issues'].append("Converted non-string to string")
            except Exception:
                result['valid'] = False
                result['issues'].append("Cannot convert to string")
                return result
        
        # Length check
        if max_length and len(data) > max_length:
            result['valid'] = False
            result['issues'].append(f"String too long: {len(data)} > {max_length}")
            result['severity'] = 'medium'
        
        # Check for malicious patterns
        issues = []
        
        # SQL injection check
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                issues.append("Potential SQL injection detected")
                result['severity'] = 'high'
        
        # XSS check (unless HTML is explicitly allowed)
        if not allow_html:
            for pattern in self.xss_patterns:
                if re.search(pattern, data, re.IGNORECASE):
                    issues.append("Potential XSS detected")
                    result['severity'] = 'high'
        
        # Command injection check
        for pattern in self.command_injection_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                issues.append("Potential command injection detected")
                result['severity'] = 'high'
        
        # Path traversal check
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                issues.append("Potential path traversal detected")
                result['severity'] = 'high'
        
        result['issues'].extend(issues)
        
        if issues:
            result['valid'] = False
        
        # Sanitize the string
        sanitized = self._sanitize_string(data, allow_html)
        result['sanitized_data'] = sanitized
        
        return result
    
    def _sanitize_string(self, data: str, allow_html: bool) -> str:
        """Sanitize string input"""
        
        # Basic HTML escaping if HTML not allowed
        if not allow_html:
            data = data.replace('<', '&lt;')
            data = data.replace('>', '&gt;')
            data = data.replace('"', '&quot;')
            data = data.replace("'", '&#x27;')
            data = data.replace('&', '&amp;')
        
        # Remove null bytes
        data = data.replace('\x00', '')
        
        # Remove other control characters except common whitespace
        data = ''.join(char for char in data 
                      if ord(char) >= 32 or char in '\t\n\r')
        
        return data
    
    def _validate_number(self, data: Any, result: Dict) -> Dict:
        """Validate numeric input"""
        
        if isinstance(data, (int, float)):
            result['sanitized_data'] = data
            return result
        
        if isinstance(data, str):
            if self.allowed_number_pattern.match(data):
                try:
                    # Try to convert to appropriate numeric type
                    if '.' in data or 'e' in data.lower():
                        result['sanitized_data'] = float(data)
                    else:
                        result['sanitized_data'] = int(data)
                except ValueError:
                    result['valid'] = False
                    result['issues'].append("Cannot convert to number")
            else:
                result['valid'] = False
                result['issues'].append("Invalid number format")
        else:
            result['valid'] = False
            result['issues'].append("Not a valid number type")
        
        return result
    
    def _validate_email(self, data: Any, result: Dict) -> Dict:
        """Validate email address"""
        
        if not isinstance(data, str):
            result['valid'] = False
            result['issues'].append("Email must be a string")
            return result
        
        # Basic email validation
        if not self.allowed_email_pattern.match(data):
            result['valid'] = False
            result['issues'].append("Invalid email format")
        
        # Additional security checks
        if len(data) > 254:  # RFC 5321 limit
            result['valid'] = False
            result['issues'].append("Email address too long")
        
        result['sanitized_data'] = data.lower().strip()
        return result
    
    def _validate_filename(self, data: Any, result: Dict) -> Dict:
        """Validate filename"""
        
        if not isinstance(data, str):
            result['valid'] = False
            result['issues'].append("Filename must be a string")
            return result
        
        # Check for valid filename pattern
        if not self.allowed_filename_pattern.match(data):
            result['valid'] = False
            result['issues'].append("Invalid filename characters")
        
        # Check for reserved names (Windows)
        reserved_names = [
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        ]
        
        if data.upper() in reserved_names:
            result['valid'] = False
            result['issues'].append("Reserved filename")
        
        result['sanitized_data'] = data.strip()
        return result
    
    def _validate_json(self, data: Any, result: Dict) -> Dict:
        """Validate JSON input"""
        
        if isinstance(data, str):
            try:
                parsed_data = json.loads(data)
                result['sanitized_data'] = parsed_data
            except json.JSONDecodeError as e:
                result['valid'] = False
                result['issues'].append(f"Invalid JSON: {e}")
        elif isinstance(data, (dict, list)):
            result['sanitized_data'] = data
        else:
            result['valid'] = False
            result['issues'].append("Invalid JSON type")
        
        return result
    
    def _validate_dict(self, data: Any, result: Dict) -> Dict:
        """Validate dictionary input"""
        
        if not isinstance(data, dict):
            result['valid'] = False
            result['issues'].append("Expected dictionary")
            return result
        
        # Recursively validate dictionary contents
        sanitized_dict = {}
        
        for key, value in data.items():
            # Validate key
            key_result = self.validate_input(
                key, 
                data_type="string", 
                max_length=100,
                context=f"{result['context']}.key"
            )
            
            if not key_result['valid']:
                result['valid'] = False
                result['issues'].append(f"Invalid key: {key_result['issues']}")
                continue
            
            # Validate value (generic validation)
            value_result = self._validate_generic(value, {'issues': []})
            
            if not value_result.get('valid', True):
                result['issues'].extend(value_result['issues'])
            
            sanitized_dict[key_result['sanitized_data']] = value_result.get('sanitized_data', value)
        
        result['sanitized_data'] = sanitized_dict
        return result
    
    def _validate_list(self, data: Any, result: Dict) -> Dict:
        """Validate list input"""
        
        if not isinstance(data, list):
            result['valid'] = False
            result['issues'].append("Expected list")
            return result
        
        # Validate list length
        if len(data) > 1000:  # Reasonable limit
            result['valid'] = False
            result['issues'].append("List too long")
            return result
        
        # Validate list contents
        sanitized_list = []
        
        for i, item in enumerate(data):
            item_result = self._validate_generic(item, {'issues': []})
            
            if not item_result.get('valid', True):
                result['issues'].extend([f"Item {i}: {issue}" for issue in item_result['issues']])
            
            sanitized_list.append(item_result.get('sanitized_data', item))
        
        result['sanitized_data'] = sanitized_list
        return result
    
    def _validate_generic(self, data: Any, result: Dict) -> Dict:
        """Generic validation for any data type"""
        
        # Basic type and size checks
        if isinstance(data, str):
            if len(data) > 10000:  # Reasonable string limit
                result['valid'] = False
                result['issues'].append("String too long")
            else:
                # Basic string sanitization
                result['sanitized_data'] = self._sanitize_string(data, allow_html=False)
        
        elif isinstance(data, (int, float)):
            # Check for reasonable numeric ranges
            if isinstance(data, float) and (abs(data) > 1e10 or abs(data) < 1e-10):
                result['issues'].append("Number outside reasonable range")
            result['sanitized_data'] = data
        
        elif isinstance(data, (dict, list)):
            # Already handled by specific validators
            result['sanitized_data'] = data
        
        else:
            # For other types, convert to string and validate
            try:
                str_representation = str(data)
                if len(str_representation) > 1000:
                    result['issues'].append("Data too large when converted to string")
                result['sanitized_data'] = data
            except Exception:
                result['valid'] = False
                result['issues'].append("Cannot process data type")
        
        return result


class RateLimiter:
    """Rate limiting to prevent abuse"""
    
    def __init__(self, default_limit: int = 100, window_seconds: int = 3600):
        self.default_limit = default_limit
        self.window_seconds = window_seconds
        self.request_counts = {}
        self.custom_limits = {}
        self.logger = logging.getLogger("security.ratelimit")
    
    def is_allowed(
        self, 
        identifier: str, 
        limit: Optional[int] = None,
        window: Optional[int] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed under rate limit
        
        Args:
            identifier: Unique identifier for the client (IP, user_id, etc.)
            limit: Custom limit for this check
            window: Custom window for this check
        
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        
        current_time = time.time()
        limit = limit or self.custom_limits.get(identifier, self.default_limit)
        window = window or self.window_seconds
        
        # Clean old entries
        self._cleanup_old_entries(current_time, window)
        
        # Get current count for identifier
        if identifier not in self.request_counts:
            self.request_counts[identifier] = []
        
        # Remove requests outside the window
        window_start = current_time - window
        self.request_counts[identifier] = [
            timestamp for timestamp in self.request_counts[identifier]
            if timestamp > window_start
        ]
        
        # Check if limit exceeded
        current_count = len(self.request_counts[identifier])
        is_allowed = current_count < limit
        
        if is_allowed:
            # Record this request
            self.request_counts[identifier].append(current_time)
        else:
            # Log rate limit violation
            self.logger.warning(
                f"Rate limit exceeded for {identifier}: {current_count} >= {limit} "
                f"in {window} seconds"
            )
        
        rate_limit_info = {
            'limit': limit,
            'current_count': current_count,
            'window_seconds': window,
            'reset_time': current_time + window - (current_time % window)
        }
        
        return is_allowed, rate_limit_info
    
    def set_custom_limit(self, identifier: str, limit: int):
        """Set custom rate limit for specific identifier"""
        self.custom_limits[identifier] = limit
        self.logger.info(f"Custom rate limit set for {identifier}: {limit}")
    
    def _cleanup_old_entries(self, current_time: float, window: int):
        """Clean up old entries to prevent memory growth"""
        
        cutoff_time = current_time - window
        
        for identifier in list(self.request_counts.keys()):
            # Remove old timestamps
            self.request_counts[identifier] = [
                timestamp for timestamp in self.request_counts[identifier]
                if timestamp > cutoff_time
            ]
            
            # Remove empty entries
            if not self.request_counts[identifier]:
                del self.request_counts[identifier]


class SecureTokenManager:
    """Secure token generation and validation"""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.logger = logging.getLogger("security.tokens")
    
    def generate_token(
        self, 
        payload: Dict[str, Any], 
        expires_in: int = 3600
    ) -> str:
        """
        Generate secure token with payload and expiration
        
        Args:
            payload: Data to include in token
            expires_in: Token expiration in seconds
        
        Returns:
            Secure token string
        """
        
        # Create token data
        token_data = {
            'payload': payload,
            'expires_at': time.time() + expires_in,
            'issued_at': time.time(),
            'token_id': secrets.token_hex(16)
        }
        
        # Serialize and encode
        serialized = json.dumps(token_data, separators=(',', ':'))
        encoded = base64.b64encode(serialized.encode()).decode()
        
        # Create signature
        signature = hmac.new(
            self.secret_key.encode(),
            encoded.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Combine encoded data and signature
        token = f"{encoded}.{signature}"
        
        self.logger.info(f"Token generated for payload keys: {list(payload.keys())}")
        
        return token
    
    def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate token and extract payload
        
        Args:
            token: Token to validate
        
        Returns:
            Validation result with payload if valid
        """
        
        result = {
            'valid': False,
            'payload': None,
            'error': None,
            'expires_at': None
        }
        
        try:
            # Split token and signature
            if '.' not in token:
                result['error'] = "Invalid token format"
                return result
            
            encoded_data, signature = token.rsplit('.', 1)
            
            # Verify signature
            expected_signature = hmac.new(
                self.secret_key.encode(),
                encoded_data.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                result['error'] = "Invalid token signature"
                return result
            
            # Decode and deserialize
            serialized = base64.b64decode(encoded_data.encode()).decode()
            token_data = json.loads(serialized)
            
            # Check expiration
            current_time = time.time()
            if token_data['expires_at'] < current_time:
                result['error'] = "Token expired"
                return result
            
            # Token is valid
            result['valid'] = True
            result['payload'] = token_data['payload']
            result['expires_at'] = token_data['expires_at']
            result['issued_at'] = token_data.get('issued_at')
            result['token_id'] = token_data.get('token_id')
            
            return result
            
        except Exception as e:
            result['error'] = f"Token validation error: {e}"
            return result


class EncryptionManager:
    """Encryption utilities for sensitive data"""
    
    def __init__(self, key: Optional[bytes] = None):
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("Cryptography library not available")
        
        self.key = key or os.urandom(32)  # 256-bit key
        self.logger = logging.getLogger("security.encryption")
    
    def encrypt(self, data: Union[str, bytes]) -> Dict[str, str]:
        """
        Encrypt data using AES-GCM
        
        Args:
            data: Data to encrypt (string or bytes)
        
        Returns:
            Dictionary with encrypted data and metadata
        """
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Generate random IV
        iv = os.urandom(12)  # 96-bit IV for GCM
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.GCM(iv),
            backend=default_backend()
        )
        
        encryptor = cipher.encryptor()
        
        # Encrypt data
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        # Return encrypted data with metadata
        return {
            'ciphertext': base64.b64encode(ciphertext).decode(),
            'iv': base64.b64encode(iv).decode(),
            'tag': base64.b64encode(encryptor.tag).decode(),
            'algorithm': 'AES-256-GCM'
        }
    
    def decrypt(self, encrypted_data: Dict[str, str]) -> bytes:
        """
        Decrypt data encrypted with encrypt()
        
        Args:
            encrypted_data: Dictionary from encrypt() method
        
        Returns:
            Decrypted data as bytes
        """
        
        # Extract components
        ciphertext = base64.b64decode(encrypted_data['ciphertext'])
        iv = base64.b64decode(encrypted_data['iv'])
        tag = base64.b64decode(encrypted_data['tag'])
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.GCM(iv, tag),
            backend=default_backend()
        )
        
        decryptor = cipher.decryptor()
        
        # Decrypt data
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext


class AuditLogger:
    """Secure audit logging for security events"""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file or "security_audit.log"
        self.logger = logging.getLogger("security.audit")
        
        # Setup file handler for audit logs
        handler = logging.FileHandler(self.log_file)
        handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        context: SecurityContext,
        details: Dict[str, Any],
        success: bool = True
    ):
        """Log security event with full context"""
        
        audit_entry = {
            'event_type': event_type,
            'severity': severity,
            'timestamp': time.time(),
            'success': success,
            'user_id': context.user_id,
            'session_id': context.session_id,
            'ip_address': context.ip_address,
            'user_agent': context.user_agent,
            'security_level': context.security_level.value,
            'permissions': context.permissions,
            'details': details
        }
        
        # Log as JSON for easy parsing
        self.logger.info(json.dumps(audit_entry))
    
    def log_authentication_attempt(
        self,
        username: str,
        success: bool,
        ip_address: str,
        failure_reason: Optional[str] = None
    ):
        """Log authentication attempt"""
        
        context = SecurityContext(
            user_id=username if success else None,
            ip_address=ip_address
        )
        
        details = {'username': username}
        if failure_reason:
            details['failure_reason'] = failure_reason
        
        self.log_security_event(
            'authentication',
            'high' if not success else 'medium',
            context,
            details,
            success
        )
    
    def log_authorization_check(
        self,
        user_id: str,
        resource: str,
        action: str,
        success: bool,
        context: SecurityContext
    ):
        """Log authorization check"""
        
        details = {
            'resource': resource,
            'action': action,
            'user_permissions': context.permissions
        }
        
        self.log_security_event(
            'authorization',
            'medium',
            context,
            details,
            success
        )


# Main security framework class
class SecurityFramework:
    """Main security framework integrating all security components"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.input_validator = InputValidator()
        self.rate_limiter = RateLimiter(
            default_limit=self.config.get('rate_limit', 100),
            window_seconds=self.config.get('rate_window', 3600)
        )
        self.token_manager = SecureTokenManager(
            secret_key=self.config.get('secret_key')
        )
        self.audit_logger = AuditLogger(
            log_file=self.config.get('audit_log_file')
        )
        
        if CRYPTO_AVAILABLE:
            self.encryption_manager = EncryptionManager()
        else:
            self.encryption_manager = None
        
        self.logger = logging.getLogger("security.framework")
        self.logger.info("Security framework initialized")
    
    def secure_operation(
        self,
        operation_name: str,
        context: SecurityContext,
        required_permissions: Optional[List[str]] = None,
        rate_limit: Optional[int] = None
    ):
        """
        Decorator for securing operations
        """
        
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                
                # Rate limiting check
                allowed, rate_info = self.rate_limiter.is_allowed(
                    context.user_id or context.ip_address or "anonymous",
                    limit=rate_limit
                )
                
                if not allowed:
                    self.audit_logger.log_security_event(
                        'rate_limit_exceeded',
                        'medium',
                        context,
                        {'operation': operation_name, 'rate_info': rate_info},
                        success=False
                    )
                    raise PermissionError("Rate limit exceeded")
                
                # Permission check
                if required_permissions:
                    if not all(perm in context.permissions for perm in required_permissions):
                        self.audit_logger.log_authorization_check(
                            context.user_id or "anonymous",
                            operation_name,
                            "execute",
                            False,
                            context
                        )
                        raise PermissionError("Insufficient permissions")
                
                # Input validation
                for key, value in kwargs.items():
                    validation = self.input_validator.validate_input(
                        value,
                        context=f"{operation_name}.{key}"
                    )
                    
                    if not validation['valid']:
                        self.audit_logger.log_security_event(
                            'input_validation_failed',
                            'high',
                            context,
                            {
                                'operation': operation_name,
                                'parameter': key,
                                'issues': validation['issues']
                            },
                            success=False
                        )
                        raise ValueError(f"Input validation failed: {validation['issues']}")
                    
                    # Use sanitized data
                    kwargs[key] = validation['sanitized_data']
                
                try:
                    # Execute the operation
                    result = func(*args, **kwargs)
                    
                    # Log successful operation
                    self.audit_logger.log_security_event(
                        'operation_executed',
                        'low',
                        context,
                        {'operation': operation_name},
                        success=True
                    )
                    
                    return result
                    
                except Exception as e:
                    # Log operation failure
                    self.audit_logger.log_security_event(
                        'operation_failed',
                        'medium',
                        context,
                        {
                            'operation': operation_name,
                            'error': str(e)
                        },
                        success=False
                    )
                    raise
            
            return wrapper
        return decorator


# Example usage
if __name__ == "__main__":
    
    # Initialize security framework
    security = SecurityFramework({
        'rate_limit': 50,
        'rate_window': 3600,
        'secret_key': 'test_secret_key_do_not_use_in_production'
    })
    
    # Create security context
    context = SecurityContext(
        user_id="test_user",
        ip_address="192.168.1.100",
        permissions=["read", "write", "optimize"]
    )
    
    # Example secure operation
    @security.secure_operation(
        operation_name="antenna_optimization",
        context=context,
        required_permissions=["optimize"],
        rate_limit=10
    )
    def optimize_antenna(frequency: str, gain_target: str):
        """Secure antenna optimization operation"""
        
        print(f"Optimizing antenna for frequency: {frequency}, target gain: {gain_target}")
        return {"status": "success", "gain": "15.2 dBi"}
    
    # Test the secure operation
    try:
        result = optimize_antenna(frequency="2.4GHz", gain_target="15dBi")
        print(f"✅ Operation successful: {result}")
    except Exception as e:
        print(f"❌ Security check failed: {e}")
    
    # Test input validation
    validator = InputValidator()
    
    test_inputs = [
        ("normal_string", "This is a normal string"),
        ("sql_injection", "'; DROP TABLE users; --"),
        ("xss_attempt", "<script>alert('xss')</script>"),
        ("valid_email", "user@example.com"),
        ("invalid_email", "not-an-email"),
        ("valid_number", "42.5"),
        ("invalid_number", "not a number")
    ]
    
    print("\nTesting input validation:")
    for test_name, test_input in test_inputs:
        data_type = "email" if "email" in test_name else "string"
        if "number" in test_name:
            data_type = "number"
        
        result = validator.validate_input(test_input, data_type=data_type)
        status = "✅" if result['valid'] else "❌"
        print(f"{status} {test_name}: {result['valid']} - Issues: {result['issues']}")
    
    print("\n✅ Generation 2 security framework demonstration complete")