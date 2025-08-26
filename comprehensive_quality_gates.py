#!/usr/bin/env python3
"""
Comprehensive Quality Gates Framework
====================================

Mandatory quality validation for all SDLC generations ensuring:
- Code quality and standards compliance
- Security vulnerability scanning
- Performance benchmarks
- Test coverage validation
- Documentation completeness
- Deployment readiness
"""

import os
import sys
import time
import subprocess
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class QualityGateStatus(Enum):
    """Quality gate status levels"""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


class Severity(Enum):
    """Issue severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class QualityResult:
    """Quality gate result"""
    name: str
    status: QualityGateStatus
    score: float  # 0.0 - 1.0
    message: str
    details: List[str]
    severity: Severity = Severity.MEDIUM
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'status': self.status.value,
            'score': self.score,
            'message': self.message,
            'details': self.details,
            'severity': self.severity.value,
            'execution_time': self.execution_time
        }


class ComprehensiveQualityGates:
    """Comprehensive quality gates framework"""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.results: List[QualityResult] = []
        self.total_score = 0.0
        self.passing_gates = 0
        self.total_gates = 0
        
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates"""
        
        print("🛡️  RUNNING COMPREHENSIVE QUALITY GATES")
        print("=" * 60)
        
        start_time = time.time()
        
        # Core Quality Gates
        gates = [
            ("Code Structure", self._validate_code_structure),
            ("Import Safety", self._validate_imports),
            ("Security Scan", self._security_scan),
            ("Performance Test", self._performance_test),
            ("Test Coverage", self._test_coverage),
            ("Documentation", self._documentation_check),
            ("Dependency Security", self._dependency_security),
            ("Code Quality", self._code_quality_check),
            ("Configuration Validation", self._config_validation),
            ("Deployment Readiness", self._deployment_readiness)
        ]
        
        for gate_name, gate_func in gates:
            print(f"\\n🔍 {gate_name}...")
            result = self._run_gate(gate_name, gate_func)
            self.results.append(result)
            
            status_icon = "✅" if result.status == QualityGateStatus.PASS else \
                         "⚠️" if result.status == QualityGateStatus.WARN else \
                         "❌" if result.status == QualityGateStatus.FAIL else "⏭️"
            
            print(f"{status_icon} {gate_name}: {result.status.value.upper()} "
                  f"(Score: {result.score:.1%}, Time: {result.execution_time:.2f}s)")
            
            if result.details and result.status != QualityGateStatus.PASS:
                for detail in result.details[:3]:  # Show first 3 details
                    print(f"   • {detail}")
                if len(result.details) > 3:
                    print(f"   • ... and {len(result.details)-3} more issues")
        
        total_time = time.time() - start_time
        
        # Calculate overall score
        self._calculate_scores()
        
        # Generate summary
        summary = self._generate_summary(total_time)
        
        print("\\n" + "=" * 60)
        print("📊 QUALITY GATES SUMMARY")
        print("=" * 60)
        
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:.1%}")
            else:
                print(f"{key}: {value}")
        
        return summary
    
    def _run_gate(self, name: str, gate_func) -> QualityResult:
        """Run individual quality gate"""
        
        start_time = time.time()
        
        try:
            result = gate_func()
            result.execution_time = time.time() - start_time
            return result
        except Exception as e:
            return QualityResult(
                name=name,
                status=QualityGateStatus.FAIL,
                score=0.0,
                message=f"Gate execution failed: {e}",
                details=[str(e)],
                severity=Severity.HIGH,
                execution_time=time.time() - start_time
            )
    
    def _validate_code_structure(self) -> QualityResult:
        """Validate code structure and organization"""
        
        issues = []
        score = 1.0
        
        # Check for required directories
        required_dirs = [
            'liquid_metal_antenna',
            'tests',
            'examples'
        ]
        
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                issues.append(f"Missing required directory: {dir_name}")
                score -= 0.2
        
        # Check for __init__.py files
        package_dirs = []
        for root, dirs, files in os.walk(self.project_root / 'liquid_metal_antenna'):
            if '__init__.py' not in files and any(f.endswith('.py') for f in files):
                issues.append(f"Missing __init__.py in: {root}")
                score -= 0.1
        
        # Check file organization
        python_files = list((self.project_root / 'liquid_metal_antenna').rglob('*.py'))
        if len(python_files) == 0:
            issues.append("No Python files found in main package")
            score = 0.0
        
        # Check for circular imports (simplified)
        for py_file in python_files[:20]:  # Check first 20 files
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                if 'from . import' in content and 'import .' in content:
                    # Potential circular import pattern
                    lines = content.split('\\n')
                    import_lines = [l for l in lines if 'import' in l and ('from .' in l or 'import .' in l)]
                    if len(import_lines) > 10:  # Many relative imports might indicate issues
                        issues.append(f"Potential complex import structure in {py_file.name}")
                        score -= 0.05
            except Exception:
                continue
        
        score = max(0.0, score)
        
        status = QualityGateStatus.PASS if score >= 0.8 else \
                QualityGateStatus.WARN if score >= 0.6 else \
                QualityGateStatus.FAIL
        
        return QualityResult(
            name="Code Structure",
            status=status,
            score=score,
            message=f"Code structure validation: {len(issues)} issues found",
            details=issues[:10]  # Limit details
        )
    
    def _validate_imports(self) -> QualityResult:
        """Validate import safety and detect malicious imports"""
        
        dangerous_imports = [
            'os.system', 'subprocess.call', 'eval', 'exec', 'compile',
            '__import__', 'input', 'raw_input', 'open'
        ]
        
        suspicious_patterns = [
            r'urllib\.request\.urlopen',
            r'requests\.get.*http',
            r'socket\.socket',
            r'pickle\.loads',
            r'yaml\.load\(',
            r'__.*__\(',
        ]
        
        issues = []
        score = 1.0
        
        python_files = list(self.project_root.rglob('*.py'))
        
        for py_file in python_files:
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                
                # Check for dangerous imports
                for dangerous in dangerous_imports:
                    if dangerous in content:
                        # Check if it's in a comment or string
                        lines = content.split('\\n')
                        for i, line in enumerate(lines):
                            if dangerous in line and not line.strip().startswith('#'):
                                if 'test' not in str(py_file).lower():  # Allow in tests
                                    issues.append(f"Potentially dangerous import '{dangerous}' in {py_file.name}:{i+1}")
                                    score -= 0.1
                
                # Check for suspicious patterns
                for pattern in suspicious_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        line_num = content[:match.start()].count('\\n') + 1
                        if 'test' not in str(py_file).lower() and 'example' not in str(py_file).lower():
                            issues.append(f"Suspicious pattern '{pattern}' in {py_file.name}:{line_num}")
                            score -= 0.05
                
            except Exception as e:
                issues.append(f"Could not read {py_file.name}: {e}")
                score -= 0.02
        
        score = max(0.0, score)
        
        # Determine status
        if len(issues) == 0:
            status = QualityGateStatus.PASS
        elif len(issues) <= 5 and score >= 0.7:
            status = QualityGateStatus.WARN
        else:
            status = QualityGateStatus.FAIL
        
        return QualityResult(
            name="Import Safety",
            status=status,
            score=score,
            message=f"Import safety check: {len(issues)} potential issues found",
            details=issues[:15],
            severity=Severity.HIGH if status == QualityGateStatus.FAIL else Severity.MEDIUM
        )
    
    def _security_scan(self) -> QualityResult:
        """Basic security vulnerability scan"""
        
        issues = []
        score = 1.0
        
        # Check for hardcoded secrets
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            r'["\'][A-Za-z0-9]{32,}["\']',  # Long strings that might be keys
        ]
        
        python_files = list(self.project_root.rglob('*.py'))
        
        for py_file in python_files:
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                
                for pattern in secret_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\\n') + 1
                        matched_text = match.group()
                        
                        # Skip obvious test/example values
                        if not any(test_val in matched_text.lower() for test_val in 
                                 ['test', 'example', 'demo', 'placeholder', 'your_', 'xxx']):
                            issues.append(f"Potential hardcoded secret in {py_file.name}:{line_num}")
                            score -= 0.15
                
                # Check for SQL injection vulnerabilities
                sql_patterns = [
                    r'execute\s*\(\s*["\'].*%.*["\']',
                    r'query\s*\(\s*["\'].*\+.*["\']',
                ]
                
                for pattern in sql_patterns:
                    if re.search(pattern, content):
                        issues.append(f"Potential SQL injection vulnerability in {py_file.name}")
                        score -= 0.2
                
            except Exception:
                continue
        
        # Check configuration files for secrets
        config_files = list(self.project_root.rglob('*.json')) + \
                      list(self.project_root.rglob('*.yaml')) + \
                      list(self.project_root.rglob('*.yml')) + \
                      list(self.project_root.rglob('*.env'))
        
        for config_file in config_files:
            try:
                content = config_file.read_text(encoding='utf-8', errors='ignore')
                
                # Look for suspicious values
                if re.search(r'["\'][A-Za-z0-9+/]{40,}["\']', content):
                    issues.append(f"Potential secret in config file: {config_file.name}")
                    score -= 0.1
                    
            except Exception:
                continue
        
        score = max(0.0, score)
        
        status = QualityGateStatus.PASS if score >= 0.9 else \
                QualityGateStatus.WARN if score >= 0.7 else \
                QualityGateStatus.FAIL
        
        return QualityResult(
            name="Security Scan",
            status=status,
            score=score,
            message=f"Security scan: {len(issues)} potential vulnerabilities",
            details=issues[:10],
            severity=Severity.CRITICAL if status == QualityGateStatus.FAIL else Severity.HIGH
        )
    
    def _performance_test(self) -> QualityResult:
        """Run basic performance tests"""
        
        issues = []
        score = 1.0
        
        # Check if any performance test files exist
        test_files = list(self.project_root.rglob('*performance*.py')) + \
                    list(self.project_root.rglob('*benchmark*.py')) + \
                    list(self.project_root.rglob('*profile*.py'))
        
        if not test_files:
            issues.append("No performance test files found")
            score -= 0.3
        
        # Try to import main package to check for obvious performance issues
        try:
            import_start = time.time()
            
            # Add project to path
            sys.path.insert(0, str(self.project_root))
            
            # Try importing main package
            try:
                import liquid_metal_antenna
                import_time = time.time() - import_start
                
                if import_time > 2.0:
                    issues.append(f"Slow package import: {import_time:.2f}s")
                    score -= 0.2
                elif import_time > 5.0:
                    issues.append(f"Very slow package import: {import_time:.2f}s")
                    score -= 0.4
                
            except ImportError as e:
                issues.append(f"Package import failed: {e}")
                score -= 0.5
                
        except Exception as e:
            issues.append(f"Performance test setup failed: {e}")
            score -= 0.1
        
        # Check for obvious performance anti-patterns in code
        python_files = list(self.project_root.rglob('*.py'))[:20]  # Check first 20 files
        
        for py_file in python_files:
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                
                # Look for performance anti-patterns
                antipatterns = [
                    (r'for.*in.*range\(len\(', "Use enumerate() instead of range(len())"),
                    (r'\.append\(.*\)\s*$', "Consider list comprehensions for better performance"),
                    (r'time\.sleep\([^)]*[5-9]\d*', "Long sleep found (>= 5 seconds)"),
                ]
                
                for pattern, message in antipatterns:
                    if re.search(pattern, content):
                        issues.append(f"{message} in {py_file.name}")
                        score -= 0.05
                        
            except Exception:
                continue
        
        score = max(0.0, score)
        
        status = QualityGateStatus.PASS if score >= 0.8 else \
                QualityGateStatus.WARN if score >= 0.6 else \
                QualityGateStatus.FAIL
        
        return QualityResult(
            name="Performance Test",
            status=status,
            score=score,
            message=f"Performance validation: {len(issues)} issues found",
            details=issues
        )
    
    def _test_coverage(self) -> QualityResult:
        """Analyze test coverage"""
        
        issues = []
        score = 0.0
        
        # Count test files
        test_files = list(self.project_root.rglob('test_*.py')) + \
                    list(self.project_root.rglob('*_test.py')) + \
                    list((self.project_root / 'tests').rglob('*.py') if (self.project_root / 'tests').exists() else [])
        
        # Count source files
        source_files = list((self.project_root / 'liquid_metal_antenna').rglob('*.py'))
        
        if not test_files:
            issues.append("No test files found")
            score = 0.0
        elif not source_files:
            issues.append("No source files found")
            score = 0.0
        else:
            # Rough coverage estimation based on file count ratio
            test_to_source_ratio = len(test_files) / len(source_files)
            
            if test_to_source_ratio >= 0.5:  # At least 1 test file per 2 source files
                score = 0.9
            elif test_to_source_ratio >= 0.3:
                score = 0.7
                issues.append(f"Test coverage might be low: {len(test_files)} test files for {len(source_files)} source files")
            else:
                score = 0.4
                issues.append(f"Low test coverage: {len(test_files)} test files for {len(source_files)} source files")
            
            # Check if tests actually import and test the source code
            tested_modules = set()
            for test_file in test_files[:10]:  # Check first 10 test files
                try:
                    content = test_file.read_text(encoding='utf-8', errors='ignore')
                    
                    # Look for imports of the main package
                    import_matches = re.finditer(r'from liquid_metal_antenna[\\w.]*\\s+import|import liquid_metal_antenna', content)
                    for match in import_matches:
                        tested_modules.add(match.group())
                        
                    # Look for test functions
                    test_functions = re.finditer(r'def test_\\w+', content)
                    test_count = len(list(test_functions))
                    
                    if test_count == 0:
                        issues.append(f"No test functions found in {test_file.name}")
                        score -= 0.05
                        
                except Exception:
                    continue
            
            if len(tested_modules) == 0 and len(test_files) > 0:
                issues.append("Test files don't seem to import the main package")
                score -= 0.3
        
        score = max(0.0, min(1.0, score))
        
        status = QualityGateStatus.PASS if score >= 0.85 else \
                QualityGateStatus.WARN if score >= 0.6 else \
                QualityGateStatus.FAIL
        
        return QualityResult(
            name="Test Coverage",
            status=status,
            score=score,
            message=f"Test coverage analysis: {len(test_files)} test files, {len(source_files)} source files",
            details=issues
        )
    
    def _documentation_check(self) -> QualityResult:
        """Check documentation completeness"""
        
        issues = []
        score = 1.0
        
        # Check for README
        readme_files = list(self.project_root.glob('README*'))
        if not readme_files:
            issues.append("No README file found")
            score -= 0.3
        else:
            # Check README quality
            readme_content = readme_files[0].read_text(encoding='utf-8', errors='ignore')
            if len(readme_content) < 500:
                issues.append("README file seems too short")
                score -= 0.1
        
        # Check for documentation files
        doc_extensions = ['*.md', '*.rst', '*.txt']
        doc_files = []
        for ext in doc_extensions:
            doc_files.extend(list(self.project_root.rglob(ext)))
        
        if len(doc_files) < 3:
            issues.append("Limited documentation files found")
            score -= 0.2
        
        # Check for docstrings in Python files
        python_files = list((self.project_root / 'liquid_metal_antenna').rglob('*.py'))[:10]
        
        documented_functions = 0
        total_functions = 0
        
        for py_file in python_files:
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                
                # Find function definitions
                function_matches = list(re.finditer(r'^\\s*def\\s+(\\w+)', content, re.MULTILINE))
                class_matches = list(re.finditer(r'^\\s*class\\s+(\\w+)', content, re.MULTILINE))
                
                total_functions += len(function_matches) + len(class_matches)
                
                # Check for docstrings after function definitions
                for match in function_matches + class_matches:
                    func_start = match.end()
                    # Look for docstring within next 200 characters
                    next_content = content[func_start:func_start + 200]
                    if '"""' in next_content or "'''" in next_content:
                        documented_functions += 1
                        
            except Exception:
                continue
        
        if total_functions > 0:
            doc_ratio = documented_functions / total_functions
            if doc_ratio < 0.3:
                issues.append(f"Low docstring coverage: {documented_functions}/{total_functions} ({doc_ratio:.1%})")
                score -= 0.3
            elif doc_ratio < 0.6:
                issues.append(f"Moderate docstring coverage: {documented_functions}/{total_functions} ({doc_ratio:.1%})")
                score -= 0.1
        
        score = max(0.0, score)
        
        status = QualityGateStatus.PASS if score >= 0.8 else \
                QualityGateStatus.WARN if score >= 0.6 else \
                QualityGateStatus.FAIL
        
        return QualityResult(
            name="Documentation",
            status=status,
            score=score,
            message=f"Documentation check: {len(doc_files)} doc files, {documented_functions}/{total_functions} documented functions",
            details=issues
        )
    
    def _dependency_security(self) -> QualityResult:
        """Check dependency security"""
        
        issues = []
        score = 1.0
        
        # Check requirements files
        req_files = list(self.project_root.glob('requirements*.txt')) + \
                   list(self.project_root.glob('pyproject.toml')) + \
                   list(self.project_root.glob('setup.py'))
        
        if not req_files:
            issues.append("No dependency files found")
            score -= 0.2
        
        # Known vulnerable packages (simplified check)
        known_vulnerable = {
            'pillow': '< 8.3.2',
            'urllib3': '< 1.26.5',
            'requests': '< 2.25.0',
            'pyyaml': '< 5.4.0'
        }
        
        for req_file in req_files:
            try:
                content = req_file.read_text(encoding='utf-8', errors='ignore')
                
                for vulnerable_pkg, vulnerable_version in known_vulnerable.items():
                    if vulnerable_pkg in content.lower():
                        issues.append(f"Potentially vulnerable dependency: {vulnerable_pkg}")
                        score -= 0.1
                        
            except Exception:
                continue
        
        score = max(0.0, score)
        
        status = QualityGateStatus.PASS if score >= 0.9 else \
                QualityGateStatus.WARN if score >= 0.7 else \
                QualityGateStatus.FAIL
        
        return QualityResult(
            name="Dependency Security",
            status=status,
            score=score,
            message=f"Dependency security check: {len(issues)} potential issues",
            details=issues
        )
    
    def _code_quality_check(self) -> QualityResult:
        """Basic code quality checks"""
        
        issues = []
        score = 1.0
        
        python_files = list(self.project_root.rglob('*.py'))[:20]  # Check first 20 files
        
        for py_file in python_files:
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                lines = content.split('\\n')
                
                # Check for very long lines
                for i, line in enumerate(lines):
                    if len(line) > 120:
                        issues.append(f"Long line in {py_file.name}:{i+1} ({len(line)} chars)")
                        score -= 0.01
                
                # Check for too many nested levels
                max_indent = 0
                for line in lines:
                    if line.strip():
                        indent = len(line) - len(line.lstrip())
                        max_indent = max(max_indent, indent)
                
                if max_indent > 32:  # More than 8 levels of 4-space indentation
                    issues.append(f"Deep nesting in {py_file.name} ({max_indent//4} levels)")
                    score -= 0.05
                
                # Check for TODO/FIXME comments
                todo_count = content.lower().count('todo') + content.lower().count('fixme')
                if todo_count > 10:
                    issues.append(f"Many TODO/FIXME comments in {py_file.name} ({todo_count})")
                    score -= 0.02
                
            except Exception:
                continue
        
        score = max(0.0, score)
        
        status = QualityGateStatus.PASS if score >= 0.8 else \
                QualityGateStatus.WARN if score >= 0.6 else \
                QualityGateStatus.FAIL
        
        return QualityResult(
            name="Code Quality",
            status=status,
            score=score,
            message=f"Code quality check: {len(issues)} issues found",
            details=issues[:10]  # Limit details
        )
    
    def _config_validation(self) -> QualityResult:
        """Validate configuration files"""
        
        issues = []
        score = 1.0
        
        # Check JSON files
        json_files = list(self.project_root.rglob('*.json'))
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                issues.append(f"Invalid JSON in {json_file.name}: {e}")
                score -= 0.2
            except Exception:
                continue
        
        # Check Python config files
        config_files = list(self.project_root.glob('*.cfg')) + \
                      list(self.project_root.glob('*.ini')) + \
                      list(self.project_root.glob('setup.py')) + \
                      list(self.project_root.glob('pyproject.toml'))
        
        for config_file in config_files:
            if not config_file.exists():
                continue
                
            try:
                content = config_file.read_text(encoding='utf-8', errors='ignore')
                
                # Basic syntax check for Python files
                if config_file.suffix == '.py':
                    try:
                        compile(content, str(config_file), 'exec')
                    except SyntaxError as e:
                        issues.append(f"Syntax error in {config_file.name}: {e}")
                        score -= 0.3
                        
            except Exception:
                continue
        
        score = max(0.0, score)
        
        status = QualityGateStatus.PASS if score >= 0.9 else \
                QualityGateStatus.WARN if score >= 0.7 else \
                QualityGateStatus.FAIL
        
        return QualityResult(
            name="Configuration Validation",
            status=status,
            score=score,
            message=f"Configuration validation: {len(issues)} issues found",
            details=issues
        )
    
    def _deployment_readiness(self) -> QualityResult:
        """Check deployment readiness"""
        
        issues = []
        score = 0.0
        
        # Check for essential files
        essential_files = [
            ('README.md', 0.2),
            ('requirements.txt', 0.15),
            ('setup.py', 0.1),
            ('pyproject.toml', 0.1),
            ('LICENSE', 0.1),
        ]
        
        for filename, weight in essential_files:
            file_variants = [filename, filename.lower(), filename.upper()]
            found = any((self.project_root / variant).exists() for variant in file_variants)
            if found:
                score += weight
            else:
                issues.append(f"Missing {filename}")
        
        # Check for Docker support
        docker_files = ['Dockerfile', 'docker-compose.yml', '.dockerignore']
        docker_score = 0
        for docker_file in docker_files:
            if (self.project_root / docker_file).exists():
                docker_score += 1
        
        if docker_score > 0:
            score += 0.15
        else:
            issues.append("No Docker deployment files found")
        
        # Check for CI/CD configuration
        ci_dirs = ['.github', '.gitlab-ci.yml', '.travis.yml', 'Jenkinsfile']
        ci_found = any((self.project_root / ci_item).exists() for ci_item in ci_dirs)
        
        if ci_found:
            score += 0.1
        else:
            issues.append("No CI/CD configuration found")
        
        # Check for package structure
        if (self.project_root / 'liquid_metal_antenna' / '__init__.py').exists():
            score += 0.1
        else:
            issues.append("Package structure incomplete")
        
        score = min(1.0, score)
        
        status = QualityGateStatus.PASS if score >= 0.8 else \
                QualityGateStatus.WARN if score >= 0.6 else \
                QualityGateStatus.FAIL
        
        return QualityResult(
            name="Deployment Readiness",
            status=status,
            score=score,
            message=f"Deployment readiness: {score:.1%} ready",
            details=issues
        )
    
    def _calculate_scores(self):
        """Calculate overall scores"""
        
        self.total_gates = len(self.results)
        self.passing_gates = sum(1 for r in self.results if r.status == QualityGateStatus.PASS)
        
        # Calculate weighted total score
        total_weight = 0
        weighted_score = 0
        
        # Different weights for different gates
        weights = {
            'Security Scan': 0.25,
            'Import Safety': 0.20,
            'Test Coverage': 0.15,
            'Code Structure': 0.10,
            'Deployment Readiness': 0.10,
            'Code Quality': 0.08,
            'Documentation': 0.05,
            'Performance Test': 0.03,
            'Dependency Security': 0.02,
            'Configuration Validation': 0.02
        }
        
        for result in self.results:
            weight = weights.get(result.name, 0.01)
            weighted_score += result.score * weight
            total_weight += weight
        
        self.total_score = weighted_score / total_weight if total_weight > 0 else 0
    
    def _generate_summary(self, execution_time: float) -> Dict[str, Any]:
        """Generate quality gates summary"""
        
        critical_failures = [r for r in self.results if r.status == QualityGateStatus.FAIL and r.severity == Severity.CRITICAL]
        high_failures = [r for r in self.results if r.status == QualityGateStatus.FAIL and r.severity == Severity.HIGH]
        warnings = [r for r in self.results if r.status == QualityGateStatus.WARN]
        
        overall_status = "FAIL" if critical_failures or len(high_failures) > 2 else \
                        "WARN" if high_failures or len(warnings) > 3 else \
                        "PASS"
        
        return {
            'Overall Status': overall_status,
            'Total Score': self.total_score,
            'Passing Gates': f"{self.passing_gates}/{self.total_gates}",
            'Critical Failures': len(critical_failures),
            'High Severity Failures': len(high_failures),
            'Warnings': len(warnings),
            'Execution Time': f"{execution_time:.2f}s",
            'Ready for Production': overall_status == "PASS" and self.total_score >= 0.85
        }
    
    def save_report(self, filename: str = "quality_gates_report.json"):
        """Save detailed quality report"""
        
        report = {
            'timestamp': time.time(),
            'overall_score': self.total_score,
            'passing_gates': self.passing_gates,
            'total_gates': self.total_gates,
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\\n📄 Quality report saved to: {filename}")


def main():
    """Run comprehensive quality gates"""
    
    project_root = Path.cwd()
    quality_gates = ComprehensiveQualityGates(project_root)
    
    print("🛡️  TERRAGON LABS - COMPREHENSIVE QUALITY GATES")
    print("🚀 Ensuring Production-Ready Code Quality")
    print()
    
    summary = quality_gates.run_all_gates()
    
    # Save detailed report
    quality_gates.save_report()
    
    # Final verdict
    print("\\n" + "=" * 60)
    if summary['Overall Status'] == 'PASS':
        print("✅ ALL QUALITY GATES PASSED")
        print("🚀 Code is ready for production deployment!")
        return 0
    elif summary['Overall Status'] == 'WARN':
        print("⚠️  QUALITY GATES PASSED WITH WARNINGS")
        print("🔧 Consider addressing warnings before deployment")
        return 1
    else:
        print("❌ QUALITY GATES FAILED")
        print("🛑 Critical issues must be resolved before deployment")
        return 2


if __name__ == "__main__":
    sys.exit(main())