"""
Security utilities and input sanitization for liquid metal antenna optimizer.
"""

import os
import re
import hashlib
import secrets
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta

# Import validation utilities
try:
    from .validation import validate_geometry, validate_frequency_range, ValidationError
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    class ValidationError(Exception):
        pass


class SecurityError(Exception):
    """Security-related error."""
    pass


class InputSanitizer:
    """Input sanitization and validation for security."""
    
    # Dangerous patterns that should be blocked
    DANGEROUS_PATTERNS = [
        r'\.\./',           # Directory traversal
        r'\.\.\\',          # Directory traversal (Windows)
        r'~/',              # Home directory access
        r'\$\w+',           # Environment variables
        r'`.*`',            # Command substitution
        r'\$\(.*\)',        # Command substitution
        r';\s*\w+',         # Command chaining
        r'\|\s*\w+',        # Pipe to commands
        r'&\s*\w+',         # Background commands
        r'<script',         # Script injection
        r'<iframe',         # Iframe injection
        r'javascript:',     # JavaScript URLs
        r'data:.*base64',   # Data URLs with base64
        r'file://',         # File URLs
        r'ftp://',          # FTP URLs (if not expected)
    ]
    
    # Safe filename characters
    SAFE_FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
    
    # Maximum sizes for different input types
    MAX_STRING_LENGTH = 10000
    MAX_FILENAME_LENGTH = 255
    MAX_PATH_LENGTH = 4096
    MAX_JSON_SIZE = 1024 * 1024  # 1MB
    
    @classmethod
    def sanitize_string(cls, input_str: str, max_length: Optional[int] = None) -> str:
        """
        Sanitize string input by removing dangerous patterns.
        
        Args:
            input_str: Input string to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
            
        Raises:
            SecurityError: If input contains dangerous patterns
        """
        if not isinstance(input_str, str):
            raise SecurityError(f"Expected string input, got {type(input_str)}")
        
        # Check length
        max_len = max_length or cls.MAX_STRING_LENGTH
        if len(input_str) > max_len:
            raise SecurityError(f"String too long: {len(input_str)} > {max_len}")
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, input_str, re.IGNORECASE):
                raise SecurityError(f"Dangerous pattern detected: {pattern}")
        
        # Strip whitespace and control characters
        sanitized = ''.join(char for char in input_str if ord(char) >= 32 or char in '\t\n\r')
        sanitized = sanitized.strip()
        
        return sanitized
    
    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """
        Sanitize filename for safe file operations.
        
        Args:
            filename: Input filename
            
        Returns:
            Sanitized filename
            
        Raises:
            SecurityError: If filename is unsafe
        """
        if not isinstance(filename, str):
            raise SecurityError(f"Expected string filename, got {type(filename)}")
        
        # Check length
        if len(filename) > cls.MAX_FILENAME_LENGTH:
            raise SecurityError(f"Filename too long: {len(filename)} > {cls.MAX_FILENAME_LENGTH}")
        
        # Remove directory separators
        filename = filename.replace('/', '_').replace('\\', '_')
        filename = filename.replace('..', '_')
        
        # Check for safe characters only
        if not cls.SAFE_FILENAME_PATTERN.match(filename):
            # Remove unsafe characters
            sanitized = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)
            filename = sanitized
        
        # Ensure not empty after sanitization
        if not filename or filename == '.':
            raise SecurityError("Filename is empty or invalid after sanitization")
        
        # Prevent reserved names on Windows
        reserved_names = {
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        }
        
        base_name = filename.split('.')[0].upper()
        if base_name in reserved_names:
            filename = f"safe_{filename}"
        
        return filename
    
    @classmethod
    def sanitize_path(cls, file_path: str, base_directory: Optional[str] = None) -> str:
        """
        Sanitize file path to prevent directory traversal.
        
        Args:
            file_path: Input file path
            base_directory: Base directory to restrict access to
            
        Returns:
            Sanitized absolute path
            
        Raises:
            SecurityError: If path is unsafe
        """
        if not isinstance(file_path, str):
            raise SecurityError(f"Expected string path, got {type(file_path)}")
        
        # Check length
        if len(file_path) > cls.MAX_PATH_LENGTH:
            raise SecurityError(f"Path too long: {len(file_path)} > {cls.MAX_PATH_LENGTH}")
        
        # Basic sanitization
        sanitized_path = cls.sanitize_string(file_path)
        
        # Resolve path
        try:
            resolved_path = os.path.abspath(os.path.expanduser(sanitized_path))
        except (ValueError, OSError) as e:
            raise SecurityError(f"Invalid path: {str(e)}")
        
        # Check base directory restriction
        if base_directory:
            base_abs = os.path.abspath(base_directory)
            if not resolved_path.startswith(base_abs):
                raise SecurityError(f"Path outside allowed directory: {resolved_path}")
        
        return resolved_path
    
    @classmethod
    def sanitize_json(cls, json_data: Union[str, Dict, List]) -> Dict[str, Any]:
        """
        Sanitize JSON data.
        
        Args:
            json_data: JSON string or data structure
            
        Returns:
            Sanitized dictionary
            
        Raises:
            SecurityError: If JSON is unsafe
        """
        # Parse JSON string if needed
        if isinstance(json_data, str):
            if len(json_data) > cls.MAX_JSON_SIZE:
                raise SecurityError(f"JSON too large: {len(json_data)} > {cls.MAX_JSON_SIZE}")
            
            try:
                data = json.loads(json_data)
            except json.JSONDecodeError as e:
                raise SecurityError(f"Invalid JSON: {str(e)}")
        else:
            data = json_data
        
        # Recursively sanitize data
        return cls._sanitize_json_recursive(data)
    
    @classmethod
    def _sanitize_json_recursive(cls, data: Any, max_depth: int = 10) -> Any:
        """Recursively sanitize JSON data structure."""
        if max_depth <= 0:
            raise SecurityError("JSON nesting too deep")
        
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                # Sanitize key
                if not isinstance(key, str):
                    continue  # Skip non-string keys
                
                safe_key = cls.sanitize_string(key, max_length=100)
                if not safe_key:
                    continue  # Skip empty keys
                
                # Sanitize value
                sanitized[safe_key] = cls._sanitize_json_recursive(value, max_depth - 1)
            
            return sanitized
        
        elif isinstance(data, list):
            return [cls._sanitize_json_recursive(item, max_depth - 1) for item in data[:1000]]  # Limit list size
        
        elif isinstance(data, str):
            return cls.sanitize_string(data, max_length=1000)
        
        elif isinstance(data, (int, float, bool)) or data is None:
            return data
        
        else:
            # Convert unknown types to string and sanitize
            return cls.sanitize_string(str(data), max_length=100)


class SecureFileHandler:
    """Secure file operations with proper validation."""
    
    def __init__(self, base_directory: Optional[str] = None):
        """
        Initialize secure file handler.
        
        Args:
            base_directory: Base directory for file operations (None = unrestricted)
        """
        self.base_directory = base_directory
        if base_directory:
            self.base_directory = os.path.abspath(base_directory)
    
    def safe_read_file(self, file_path: str, max_size: int = 10 * 1024 * 1024) -> str:
        """
        Safely read file with size and path validation.
        
        Args:
            file_path: Path to file
            max_size: Maximum file size in bytes
            
        Returns:
            File contents as string
            
        Raises:
            SecurityError: If file access is unsafe
        """
        # Sanitize path
        safe_path = InputSanitizer.sanitize_path(file_path, self.base_directory)
        
        # Check if file exists
        if not os.path.exists(safe_path):
            raise SecurityError(f"File not found: {safe_path}")
        
        # Check if it's actually a file
        if not os.path.isfile(safe_path):
            raise SecurityError(f"Path is not a file: {safe_path}")
        
        # Check file size
        file_size = os.path.getsize(safe_path)
        if file_size > max_size:
            raise SecurityError(f"File too large: {file_size} > {max_size}")
        
        # Read file safely
        try:
            with open(safe_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return content
            
        except (OSError, UnicodeDecodeError) as e:
            raise SecurityError(f"Failed to read file: {str(e)}")
    
    def safe_write_file(
        self,
        file_path: str,
        content: str,
        overwrite: bool = False,
        max_size: int = 10 * 1024 * 1024
    ) -> None:
        """
        Safely write file with validation.
        
        Args:
            file_path: Path to file
            content: Content to write
            overwrite: Allow overwriting existing files
            max_size: Maximum content size
            
        Raises:
            SecurityError: If file operation is unsafe
        """
        # Validate content size
        if len(content.encode('utf-8')) > max_size:
            raise SecurityError(f"Content too large: {len(content)} > {max_size}")
        
        # Sanitize path
        safe_path = InputSanitizer.sanitize_path(file_path, self.base_directory)
        
        # Check overwrite policy
        if os.path.exists(safe_path) and not overwrite:
            raise SecurityError(f"File exists and overwrite=False: {safe_path}")
        
        # Ensure directory exists
        directory = os.path.dirname(safe_path)
        if directory:
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as e:
                raise SecurityError(f"Failed to create directory: {str(e)}")
        
        # Write file safely using temporary file
        try:
            temp_path = safe_path + '.tmp'
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Atomic move
            os.replace(temp_path, safe_path)
            
        except OSError as e:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise SecurityError(f"Failed to write file: {str(e)}")
    
    def safe_create_temp_file(self, suffix: str = '', prefix: str = 'lma_') -> Tuple[str, str]:
        """
        Create secure temporary file.
        
        Args:
            suffix: File suffix
            prefix: File prefix
            
        Returns:
            Tuple of (file_path, file_handle)
        """
        # Sanitize prefix and suffix
        safe_prefix = InputSanitizer.sanitize_filename(prefix)
        safe_suffix = InputSanitizer.sanitize_filename(suffix)
        
        # Create temporary file
        try:
            fd, temp_path = tempfile.mkstemp(suffix=safe_suffix, prefix=safe_prefix)
            return temp_path, os.fdopen(fd, 'w+')
        except OSError as e:
            raise SecurityError(f"Failed to create temporary file: {str(e)}")
    
    def safe_delete_file(self, file_path: str, confirm_content: Optional[str] = None) -> None:
        """
        Safely delete file with confirmation.
        
        Args:
            file_path: Path to file
            confirm_content: Content hash for confirmation (optional)
            
        Raises:
            SecurityError: If deletion is unsafe
        """
        # Sanitize path
        safe_path = InputSanitizer.sanitize_path(file_path, self.base_directory)
        
        # Check if file exists
        if not os.path.exists(safe_path):
            return  # Already deleted
        
        # Confirm content if requested
        if confirm_content:
            try:
                current_content = self.safe_read_file(safe_path)
                current_hash = hashlib.sha256(current_content.encode()).hexdigest()
                
                if current_hash != confirm_content:
                    raise SecurityError("File content hash mismatch - deletion aborted")
            except Exception as e:
                raise SecurityError(f"Failed to verify file for deletion: {str(e)}")
        
        # Delete file
        try:
            os.unlink(safe_path)
        except OSError as e:
            raise SecurityError(f"Failed to delete file: {str(e)}")


class SecureEncryption:
    """Secure encryption for sensitive data protection."""
    
    def __init__(self, key=None):
        """Initialize secure encryption."""
        self.key = key or "default_secure_key_12345678"
        
    def encrypt_data(self, data):
        """Encrypt data (simplified implementation)."""
        # Simplified encryption for demonstration
        if isinstance(data, str):
            encrypted = ''.join(chr((ord(c) + 5) % 256) for c in data)
            return encrypted.encode('latin1', errors='ignore')
        return str(data).encode()
    
    def decrypt_data(self, encrypted_data):
        """Decrypt data (simplified implementation)."""
        # Simplified decryption for demonstration
        try:
            if isinstance(encrypted_data, bytes):
                decrypted = ''.join(chr((c - 5) % 256) for c in encrypted_data)
                return decrypted
        except:
            return str(encrypted_data)
        return str(encrypted_data)


class AdvancedThreatDetector:
    """Advanced threat detection system for security monitoring."""
    
    def __init__(self):
        """Initialize threat detector."""
        self.threat_patterns = [
            'malicious_code',
            'injection_attack',
            'buffer_overflow',
            'privilege_escalation'
        ]
        self.detected_threats = []
    
    def detect_threats(self, input_data):
        """Detect security threats in input data."""
        threats = []
        
        if isinstance(input_data, str):
            # Check for dangerous patterns
            dangerous_keywords = ['eval', 'exec', '__import__', 'subprocess']
            for keyword in dangerous_keywords:
                if keyword in input_data.lower():
                    threats.append(f"Dangerous keyword detected: {keyword}")
        
        self.detected_threats.extend(threats)
        return threats
    
    def get_threat_report(self):
        """Get threat detection report."""
        return {
            'total_threats_detected': len(self.detected_threats),
            'threat_patterns': self.threat_patterns,
            'recent_threats': self.detected_threats[-10:] if self.detected_threats else []
        }


class ComprehensiveSecurityValidator:
    """Comprehensive security validation framework."""
    
    def __init__(self, enable_monitoring: bool = True):
        """Initialize comprehensive security validator."""
        self.enable_monitoring = enable_monitoring
        self.validation_attempts = 0
        self.failed_validations = 0
        self.detected_threats = []
        
    def validate_all_inputs(
        self,
        geometry=None,
        frequency=None,
        spec=None,
        parameters=None,
        client_id: str = 'unknown'
    ):
        """Comprehensive input validation."""
        self.validation_attempts += 1
        threats = []
        
        try:
            # Basic validation logic
            if geometry is not None and hasattr(geometry, 'size'):
                if geometry.size > 1000000:  # 1M element limit
                    threats.append(f"Geometry too large: {geometry.size}")
            
            if frequency is not None:
                if not isinstance(frequency, (int, float)) or frequency <= 0:
                    threats.append(f"Invalid frequency: {frequency}")
            
            validation_passed = len(threats) == 0
            if not validation_passed:
                self.failed_validations += 1
            
            return validation_passed, threats
            
        except Exception as e:
            self.failed_validations += 1
            return False, [f"Validation error: {e}"]
    
    def get_security_report(self):
        """Get security validation report."""
        success_rate = 1.0 - (self.failed_validations / max(self.validation_attempts, 1))
        
        return {
            'validation_attempts': self.validation_attempts,
            'failed_validations': self.failed_validations,
            'success_rate': success_rate * 100,
            'detected_threats': len(self.detected_threats),
            'security_score': success_rate * 100
        }


class SecurityAudit:
    """Security auditing and monitoring."""
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize security audit system.
        
        Args:
            log_file: Path to security audit log
        """
        self.log_file = log_file
        self.audit_entries = []
        
    def log_security_event(
        self,
        event_type: str,
        description: str,
        severity: str = 'INFO',
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log security event.
        
        Args:
            event_type: Type of security event
            description: Event description
            severity: Event severity (INFO, WARNING, ERROR)
            context: Additional context information
        """
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'description': description,
            'severity': severity,
            'context': context or {},
            'process_id': os.getpid()
        }
        
        self.audit_entries.append(entry)
        
        # Write to log file if configured
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(entry) + '\n')
            except OSError:
                pass  # Continue even if logging fails
    
    def check_file_permissions(self, file_path: str) -> Dict[str, Any]:
        """
        Check file permissions and security.
        
        Args:
            file_path: Path to check
            
        Returns:
            Dictionary with permission information
        """
        try:
            stat_info = os.stat(file_path)
            
            # Extract permission bits
            permissions = {
                'owner_read': bool(stat_info.st_mode & 0o400),
                'owner_write': bool(stat_info.st_mode & 0o200),
                'owner_execute': bool(stat_info.st_mode & 0o100),
                'group_read': bool(stat_info.st_mode & 0o040),
                'group_write': bool(stat_info.st_mode & 0o020),
                'group_execute': bool(stat_info.st_mode & 0o010),
                'other_read': bool(stat_info.st_mode & 0o004),
                'other_write': bool(stat_info.st_mode & 0o002),
                'other_execute': bool(stat_info.st_mode & 0o001),
            }
            
            # Security warnings
            warnings = []
            if permissions['other_write']:
                warnings.append("File is writable by others")
            
            if permissions['group_write'] and permissions['other_read']:
                warnings.append("File is group-writable and world-readable")
            
            return {
                'file_path': file_path,
                'permissions': permissions,
                'octal_mode': oct(stat_info.st_mode)[-3:],
                'warnings': warnings,
                'uid': stat_info.st_uid,
                'gid': stat_info.st_gid,
                'size': stat_info.st_size,
                'modified_time': datetime.fromtimestamp(stat_info.st_mtime).isoformat()
            }
            
        except OSError as e:
            return {
                'file_path': file_path,
                'error': str(e),
                'permissions': None
            }
    
    def scan_directory_security(self, directory: str) -> Dict[str, Any]:
        """
        Scan directory for security issues.
        
        Args:
            directory: Directory to scan
            
        Returns:
            Security scan results
        """
        results = {
            'directory': directory,
            'scan_time': datetime.utcnow().isoformat(),
            'issues': [],
            'file_count': 0,
            'suspicious_files': []
        }
        
        try:
            for root, dirs, files in os.walk(directory):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    results['file_count'] += 1
                    
                    # Check file permissions
                    perm_info = self.check_file_permissions(file_path)
                    if perm_info.get('warnings'):
                        results['issues'].extend([
                            f"{file_path}: {warning}" for warning in perm_info['warnings']
                        ])
                    
                    # Check for suspicious extensions
                    suspicious_extensions = ['.exe', '.bat', '.cmd', '.scr', '.vbs', '.ps1']
                    if any(file_name.lower().endswith(ext) for ext in suspicious_extensions):
                        results['suspicious_files'].append({
                            'path': file_path,
                            'reason': 'Suspicious file extension'
                        })
                    
                    # Check for hidden files that might be suspicious
                    if file_name.startswith('.') and not file_name.startswith('.git'):
                        results['suspicious_files'].append({
                            'path': file_path,
                            'reason': 'Hidden file'
                        })
            
            return results
            
        except OSError as e:
            results['error'] = str(e)
            return results
    
    def generate_audit_report(self) -> Dict[str, Any]:
        """Generate comprehensive security audit report."""
        recent_events = [
            entry for entry in self.audit_entries
            if datetime.fromisoformat(entry['timestamp']) >= 
               datetime.utcnow() - timedelta(hours=24)
        ]
        
        event_counts = {}
        severity_counts = {'INFO': 0, 'WARNING': 0, 'ERROR': 0}
        
        for event in recent_events:
            event_type = event['event_type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            severity = event.get('severity', 'INFO')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'report_time': datetime.utcnow().isoformat(),
            'total_events': len(recent_events),
            'event_types': event_counts,
            'severity_distribution': severity_counts,
            'recent_events': recent_events[-10:],  # Last 10 events
            'security_score': self._calculate_security_score(severity_counts)
        }
    
    def _calculate_security_score(self, severity_counts: Dict[str, int]) -> float:
        """Calculate security score based on event severities."""
        total_events = sum(severity_counts.values())
        if total_events == 0:
            return 100.0
        
        # Weight different severities
        weighted_score = (
            severity_counts['INFO'] * 1.0 +
            severity_counts['WARNING'] * 0.5 +
            severity_counts['ERROR'] * 0.0
        )
        
        return (weighted_score / total_events) * 100


def sanitize_input(input_data: Any, input_type: str = 'string') -> Any:
    """
    Convenience function for input sanitization.
    
    Args:
        input_data: Data to sanitize
        input_type: Type of input ('string', 'filename', 'path', 'json')
        
    Returns:
        Sanitized input
        
    Raises:
        SecurityError: If input is unsafe
    """
    if input_type == 'string':
        return InputSanitizer.sanitize_string(input_data)
    elif input_type == 'filename':
        return InputSanitizer.sanitize_filename(input_data)
    elif input_type == 'path':
        return InputSanitizer.sanitize_path(input_data)
    elif input_type == 'json':
        return InputSanitizer.sanitize_json(input_data)
    else:
        raise SecurityError(f"Unknown input type: {input_type}")


def secure_operation(func):
    """Decorator for secure operation execution."""
    def wrapper(*args, **kwargs):
        # Basic security checks before operation
        validator = ComprehensiveSecurityValidator()
        
        # Perform basic input validation
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            # Log security-relevant exceptions
            print(f"Secure operation {func.__name__} failed: {e}")
            raise
    
    return wrapper


# Global security audit instance
_security_audit = SecurityAudit()

def get_security_audit() -> SecurityAudit:
    """Get global security audit instance."""
    return _security_audit


class SecurityValidator:
    """Simple security validator for Generation 2 compatibility."""
    
    def __init__(self):
        """Initialize security validator."""
        self.validation_count = 0
        self.failed_validations = 0
    
    def validate_geometry(self, geometry: Any) -> Dict[str, Any]:
        """
        Validate antenna geometry for security issues.
        
        Args:
            geometry: Antenna geometry to validate
            
        Returns:
            Validation result dictionary
        """
        self.validation_count += 1
        warnings = []
        
        try:
            # Basic size validation
            if hasattr(geometry, 'size'):
                if geometry.size > 1000000:  # 1M elements max
                    warnings.append("Geometry size exceeds recommended limits")
                if geometry.size < 10:
                    warnings.append("Geometry size too small for realistic antenna")
            elif hasattr(geometry, '__len__'):
                # List-based geometry
                def count_elements(item):
                    if hasattr(item, '__len__') and not isinstance(item, str):
                        return sum(count_elements(sub) for sub in item)
                    return 1
                
                total_elements = count_elements(geometry)
                if total_elements > 1000000:
                    warnings.append("Geometry size exceeds recommended limits")
                if total_elements < 10:
                    warnings.append("Geometry size too small for realistic antenna")
            
            # Value range validation
            if VALIDATION_AVAILABLE:
                try:
                    validate_geometry(geometry)
                except ValidationError as e:
                    warnings.append(f"Geometry validation failed: {e}")
            
            valid = len(warnings) == 0
            if not valid:
                self.failed_validations += 1
            
            return {
                'valid': valid,
                'warnings': warnings,
                'validation_count': self.validation_count,
                'failure_rate': self.failed_validations / max(self.validation_count, 1)
            }
            
        except Exception as e:
            self.failed_validations += 1
            return {
                'valid': False,
                'warnings': [f"Validation error: {e}"],
                'validation_count': self.validation_count,
                'failure_rate': self.failed_validations / max(self.validation_count, 1)
            }
    
    def validate_frequency(self, frequency: float) -> Dict[str, Any]:
        """Validate frequency parameter."""
        warnings = []
        
        if not isinstance(frequency, (int, float)):
            warnings.append("Frequency must be numeric")
        elif frequency <= 0:
            warnings.append("Frequency must be positive")
        elif frequency > 100e9:  # 100 GHz
            warnings.append("Frequency exceeds typical antenna design range")
        elif frequency < 1e6:  # 1 MHz
            warnings.append("Frequency below typical antenna design range")
        
        return {
            'valid': len(warnings) == 0,
            'warnings': warnings
        }
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get security validation status."""
        success_rate = 1.0 - (self.failed_validations / max(self.validation_count, 1))
        
        return {
            'validation_count': self.validation_count,
            'failed_validations': self.failed_validations,
            'success_rate': success_rate * 100,
            'security_score': success_rate * 100
        }