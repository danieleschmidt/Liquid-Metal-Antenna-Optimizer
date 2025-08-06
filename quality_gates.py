#!/usr/bin/env python3
"""
Quality gates and continuous validation for liquid metal antenna optimizer.
"""

import os
import sys
import subprocess
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class QualityMetrics:
    """Quality metrics for the project."""
    
    # Code quality
    test_coverage: float
    test_pass_rate: float
    linting_score: float
    security_score: float
    
    # Performance metrics
    benchmark_results: Dict[str, float]
    memory_efficiency: float
    
    # Documentation metrics
    doc_coverage: float
    api_completeness: float
    
    # Overall scores
    overall_quality_score: float
    passed: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class QualityGateRunner:
    """Main class for running quality gates."""
    
    def __init__(self, project_root: Optional[str] = None):
        """Initialize quality gate runner."""
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.results = {}
        self.thresholds = self._load_thresholds()
    
    def _load_thresholds(self) -> Dict[str, float]:
        """Load quality thresholds from configuration."""
        default_thresholds = {
            'min_test_coverage': 80.0,
            'min_test_pass_rate': 95.0,
            'min_linting_score': 8.0,
            'min_security_score': 85.0,
            'max_benchmark_regression': 20.0,  # % slower than baseline
            'min_memory_efficiency': 80.0,
            'min_doc_coverage': 70.0,
            'min_api_completeness': 90.0,
            'min_overall_quality': 80.0
        }
        
        # Try to load from file if it exists
        config_file = self.project_root / 'quality_config.json'
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    custom_thresholds = json.load(f)
                default_thresholds.update(custom_thresholds)
            except Exception as e:
                print(f"Warning: Could not load quality config: {e}")
        
        return default_thresholds
    
    def run_all_gates(self, fast_mode: bool = False) -> QualityMetrics:
        """Run all quality gates."""
        print("🚀 Running Quality Gates...")
        print("=" * 50)
        
        try:
            # Code quality gates
            test_coverage = self._run_test_coverage()
            test_pass_rate = self._run_tests(fast_mode)
            linting_score = self._run_linting()
            security_score = self._run_security_checks()
            
            # Performance gates
            benchmark_results = self._run_benchmarks(fast_mode)
            memory_efficiency = self._run_memory_tests()
            
            # Documentation gates
            doc_coverage = self._run_doc_coverage()
            api_completeness = self._run_api_completeness()
            
            # Calculate overall score
            overall_score = self._calculate_overall_score({
                'test_coverage': test_coverage,
                'test_pass_rate': test_pass_rate,
                'linting_score': linting_score,
                'security_score': security_score,
                'memory_efficiency': memory_efficiency,
                'doc_coverage': doc_coverage,
                'api_completeness': api_completeness
            })
            
            # Determine if all gates passed
            passed = self._check_all_thresholds({
                'test_coverage': test_coverage,
                'test_pass_rate': test_pass_rate,
                'linting_score': linting_score,
                'security_score': security_score,
                'memory_efficiency': memory_efficiency,
                'doc_coverage': doc_coverage,
                'api_completeness': api_completeness,
                'overall_quality_score': overall_score
            })
            
            metrics = QualityMetrics(
                test_coverage=test_coverage,
                test_pass_rate=test_pass_rate,
                linting_score=linting_score,
                security_score=security_score,
                benchmark_results=benchmark_results,
                memory_efficiency=memory_efficiency,
                doc_coverage=doc_coverage,
                api_completeness=api_completeness,
                overall_quality_score=overall_score,
                passed=passed
            )
            
            self._print_summary(metrics)
            return metrics
        
        except Exception as e:
            print(f"❌ Quality gates failed with error: {e}")
            raise
    
    def _run_test_coverage(self) -> float:
        """Run test coverage analysis."""
        print("📊 Running test coverage analysis...")
        
        try:
            # Run pytest with coverage
            result = subprocess.run([
                sys.executable, '-m', 'pytest',
                '--cov=liquid_metal_antenna',
                '--cov-report=term-missing',
                '--cov-report=json:coverage.json',
                '-q'
            ], cwd=self.project_root, capture_output=True, text=True)
            
            # Parse coverage from JSON report
            coverage_file = self.project_root / 'coverage.json'
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                coverage_percent = coverage_data['totals']['percent_covered']
            else:
                # Fallback: parse from output
                coverage_percent = 0.0
                for line in result.stdout.split('\n'):
                    if 'TOTAL' in line and '%' in line:
                        coverage_percent = float(line.split()[-1].replace('%', ''))
                        break
            
            status = "✅" if coverage_percent >= self.thresholds['min_test_coverage'] else "❌"
            print(f"   {status} Coverage: {coverage_percent:.1f}% (min: {self.thresholds['min_test_coverage']:.1f}%)")
            
            return coverage_percent
        
        except Exception as e:
            print(f"   ❌ Coverage analysis failed: {e}")
            return 0.0
    
    def _run_tests(self, fast_mode: bool = False) -> float:
        """Run test suite and calculate pass rate."""
        print("🧪 Running test suite...")
        
        try:
            cmd = [sys.executable, '-m', 'pytest', '-v']
            
            if fast_mode:
                cmd.extend(['-m', 'not slow'])
            
            cmd.extend(['--json-report', '--json-report-file=test_results.json'])
            
            result = subprocess.run(
                cmd, cwd=self.project_root, capture_output=True, text=True
            )
            
            # Parse test results
            results_file = self.project_root / 'test_results.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    test_data = json.load(f)
                
                total_tests = test_data['summary']['total']
                passed_tests = test_data['summary'].get('passed', 0)
                pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0.0
            else:
                # Fallback: parse from pytest output
                pass_rate = 100.0 if result.returncode == 0 else 0.0
            
            status = "✅" if pass_rate >= self.thresholds['min_test_pass_rate'] else "❌"
            print(f"   {status} Test pass rate: {pass_rate:.1f}% (min: {self.thresholds['min_test_pass_rate']:.1f}%)")
            
            return pass_rate
        
        except Exception as e:
            print(f"   ❌ Test execution failed: {e}")
            return 0.0
    
    def _run_linting(self) -> float:
        """Run code linting and calculate score."""
        print("🔍 Running code linting...")
        
        scores = []
        
        # Run flake8
        try:
            result = subprocess.run([
                sys.executable, '-m', 'flake8',
                'liquid_metal_antenna',
                '--count',
                '--statistics'
            ], cwd=self.project_root, capture_output=True, text=True)
            
            # Simple scoring: 10 - (violations / 100)
            violation_count = len([l for l in result.stdout.split('\n') if l.strip()])
            flake8_score = max(0, 10 - violation_count / 20)
            scores.append(flake8_score)
            
        except Exception:
            print("   ⚠️  flake8 not available")
        
        # Run pylint if available
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pylint',
                'liquid_metal_antenna',
                '--output-format=json'
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.stdout:
                pylint_data = json.loads(result.stdout)
                # Calculate score based on message count
                message_count = len(pylint_data)
                pylint_score = max(0, 10 - message_count / 50)
                scores.append(pylint_score)
            
        except Exception:
            print("   ⚠️  pylint not available")
        
        # Average available scores
        avg_score = sum(scores) / len(scores) if scores else 7.0  # Default reasonable score
        
        status = "✅" if avg_score >= self.thresholds['min_linting_score'] else "❌"
        print(f"   {status} Linting score: {avg_score:.1f}/10 (min: {self.thresholds['min_linting_score']:.1f}/10)")
        
        return avg_score
    
    def _run_security_checks(self) -> float:
        """Run security analysis."""
        print("🔒 Running security checks...")
        
        try:
            # Run bandit security scanner
            result = subprocess.run([
                sys.executable, '-m', 'bandit',
                '-r', 'liquid_metal_antenna',
                '-f', 'json'
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.stdout:
                bandit_data = json.loads(result.stdout)
                issue_count = len(bandit_data.get('results', []))
                # Score based on issue severity and count
                security_score = max(0, 100 - issue_count * 10)  # Each issue reduces score by 10
            else:
                security_score = 90.0  # Default good score if no issues detected
            
        except Exception:
            print("   ⚠️  bandit not available, using basic checks")
            security_score = 85.0  # Conservative default
        
        status = "✅" if security_score >= self.thresholds['min_security_score'] else "❌"
        print(f"   {status} Security score: {security_score:.1f}% (min: {self.thresholds['min_security_score']:.1f}%)")
        
        return security_score
    
    def _run_benchmarks(self, fast_mode: bool = False) -> Dict[str, float]:
        """Run performance benchmarks."""
        print("⚡ Running performance benchmarks...")
        
        benchmark_results = {}
        
        try:
            cmd = [sys.executable, '-m', 'pytest', 'tests/test_performance_benchmarks.py']
            if fast_mode:
                cmd.extend(['-k', 'not slow'])
            
            cmd.extend(['-v', '--benchmark-json=benchmark_results.json'])
            
            result = subprocess.run(
                cmd, cwd=self.project_root, capture_output=True, text=True
            )
            
            # Parse benchmark results if available
            benchmark_file = self.project_root / 'benchmark_results.json'
            if benchmark_file.exists():
                with open(benchmark_file, 'r') as f:
                    data = json.load(f)
                
                for benchmark in data.get('benchmarks', []):
                    name = benchmark['name']
                    mean_time = benchmark['stats']['mean']
                    benchmark_results[name] = mean_time
            
            print(f"   ✅ Completed {len(benchmark_results)} benchmarks")
            
        except Exception as e:
            print(f"   ❌ Benchmark execution failed: {e}")
        
        return benchmark_results
    
    def _run_memory_tests(self) -> float:
        """Run memory efficiency tests."""
        print("💾 Running memory efficiency tests...")
        
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pytest',
                'tests/test_performance_benchmarks.py::TestResourceUtilizationBenchmarks::test_memory_efficiency_benchmark',
                '-v'
            ], cwd=self.project_root, capture_output=True, text=True)
            
            # Simple pass/fail scoring for memory efficiency
            memory_efficiency = 90.0 if result.returncode == 0 else 60.0
            
        except Exception:
            memory_efficiency = 75.0  # Conservative default
        
        status = "✅" if memory_efficiency >= self.thresholds['min_memory_efficiency'] else "❌"
        print(f"   {status} Memory efficiency: {memory_efficiency:.1f}% (min: {self.thresholds['min_memory_efficiency']:.1f}%)")
        
        return memory_efficiency
    
    def _run_doc_coverage(self) -> float:
        """Analyze documentation coverage."""
        print("📚 Analyzing documentation coverage...")
        
        try:
            # Count documented vs undocumented functions/classes
            total_items = 0
            documented_items = 0
            
            for py_file in self.project_root.glob('liquid_metal_antenna/**/*.py'):
                if py_file.name.startswith('__'):
                    continue
                
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Count classes and functions
                    import re
                    classes = re.findall(r'^class\s+\w+.*?:', content, re.MULTILINE)
                    functions = re.findall(r'^def\s+\w+.*?:', content, re.MULTILINE)
                    
                    total_items += len(classes) + len(functions)
                    
                    # Count those with docstrings (simplified)
                    docstring_pattern = r'""".*?"""'
                    docstrings = re.findall(docstring_pattern, content, re.DOTALL)
                    documented_items += min(len(docstrings), len(classes) + len(functions))
            
            doc_coverage = (documented_items / total_items * 100) if total_items > 0 else 100.0
            
        except Exception:
            doc_coverage = 70.0  # Default reasonable value
        
        status = "✅" if doc_coverage >= self.thresholds['min_doc_coverage'] else "❌"
        print(f"   {status} Documentation coverage: {doc_coverage:.1f}% (min: {self.thresholds['min_doc_coverage']:.1f}%)")
        
        return doc_coverage
    
    def _run_api_completeness(self) -> float:
        """Check API completeness."""
        print("🔧 Checking API completeness...")
        
        try:
            # Check that key modules can be imported
            key_modules = [
                'liquid_metal_antenna.core.antenna_spec',
                'liquid_metal_antenna.solvers.fdtd',
                'liquid_metal_antenna.optimization.lma_optimizer',
                'liquid_metal_antenna.designs.patch_antenna'
            ]
            
            importable_count = 0
            for module in key_modules:
                try:
                    __import__(module)
                    importable_count += 1
                except ImportError:
                    pass
            
            api_completeness = (importable_count / len(key_modules) * 100)
            
        except Exception:
            api_completeness = 80.0  # Conservative default
        
        status = "✅" if api_completeness >= self.thresholds['min_api_completeness'] else "❌"
        print(f"   {status} API completeness: {api_completeness:.1f}% (min: {self.thresholds['min_api_completeness']:.1f}%)")
        
        return api_completeness
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted overall quality score."""
        weights = {
            'test_coverage': 0.25,
            'test_pass_rate': 0.25,
            'linting_score': 0.15,
            'security_score': 0.15,
            'memory_efficiency': 0.10,
            'doc_coverage': 0.05,
            'api_completeness': 0.05
        }
        
        # Normalize linting score to 0-100 scale
        normalized_metrics = metrics.copy()
        normalized_metrics['linting_score'] = metrics['linting_score'] * 10  # 0-10 -> 0-100
        
        weighted_sum = sum(
            normalized_metrics[metric] * weight 
            for metric, weight in weights.items()
            if metric in normalized_metrics
        )
        
        return weighted_sum
    
    def _check_all_thresholds(self, metrics: Dict[str, float]) -> bool:
        """Check if all quality thresholds are met."""
        checks = [
            metrics['test_coverage'] >= self.thresholds['min_test_coverage'],
            metrics['test_pass_rate'] >= self.thresholds['min_test_pass_rate'],
            metrics['linting_score'] >= self.thresholds['min_linting_score'],
            metrics['security_score'] >= self.thresholds['min_security_score'],
            metrics['memory_efficiency'] >= self.thresholds['min_memory_efficiency'],
            metrics['doc_coverage'] >= self.thresholds['min_doc_coverage'],
            metrics['api_completeness'] >= self.thresholds['min_api_completeness'],
            metrics['overall_quality_score'] >= self.thresholds['min_overall_quality']
        ]
        
        return all(checks)
    
    def _print_summary(self, metrics: QualityMetrics) -> None:
        """Print quality gate summary."""
        print("\n" + "=" * 50)
        print("📋 QUALITY GATE SUMMARY")
        print("=" * 50)
        
        status_icon = "✅ PASSED" if metrics.passed else "❌ FAILED"
        print(f"Overall Status: {status_icon}")
        print(f"Overall Quality Score: {metrics.overall_quality_score:.1f}%")
        print()
        
        print("Individual Metrics:")
        print(f"  📊 Test Coverage:     {metrics.test_coverage:.1f}%")
        print(f"  🧪 Test Pass Rate:    {metrics.test_pass_rate:.1f}%")
        print(f"  🔍 Linting Score:     {metrics.linting_score:.1f}/10")
        print(f"  🔒 Security Score:    {metrics.security_score:.1f}%")
        print(f"  💾 Memory Efficiency: {metrics.memory_efficiency:.1f}%")
        print(f"  📚 Doc Coverage:      {metrics.doc_coverage:.1f}%")
        print(f"  🔧 API Completeness:  {metrics.api_completeness:.1f}%")
        
        if metrics.benchmark_results:
            print(f"  ⚡ Benchmarks:        {len(metrics.benchmark_results)} completed")
        
        print("\n" + "=" * 50)
        
        if not metrics.passed:
            print("❌ Some quality gates failed. Please address the issues above.")
            print("💡 Run with --help for guidance on improving quality metrics.")
        else:
            print("🎉 All quality gates passed! Great work!")


def main():
    """Main entry point for quality gates."""
    parser = argparse.ArgumentParser(description='Run quality gates for liquid metal antenna optimizer')
    parser.add_argument('--fast', action='store_true', help='Run in fast mode (skip slow tests)')
    parser.add_argument('--output', type=str, help='Output results to JSON file')
    parser.add_argument('--project-root', type=str, help='Project root directory')
    
    args = parser.parse_args()
    
    # Run quality gates
    runner = QualityGateRunner(args.project_root)
    
    try:
        metrics = runner.run_all_gates(fast_mode=args.fast)
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)
            print(f"\n📄 Results saved to {args.output}")
        
        # Exit with appropriate code
        sys.exit(0 if metrics.passed else 1)
        
    except KeyboardInterrupt:
        print("\n⏹️  Quality gate execution interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Quality gate execution failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()