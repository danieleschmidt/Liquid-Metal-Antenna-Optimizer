# Testing Infrastructure

## Overview

The liquid metal antenna optimizer includes comprehensive testing infrastructure with multiple layers of validation:

## Test Structure

### Core Test Files
- `test_core_antenna_spec.py` - Tests for antenna specification classes
- `test_solvers_fdtd.py` - FDTD solver functionality tests  
- `test_optimization_lma.py` - Optimization algorithm tests
- `test_security_validation.py` - Security and validation tests
- `test_performance_benchmarks.py` - Performance benchmarks and quality gates
- `test_integration.py` - End-to-end integration tests
- `conftest.py` - Shared test fixtures and configuration

### Test Categories

#### Unit Tests
- Individual component functionality
- Input validation and error handling
- Mathematical correctness verification

#### Integration Tests  
- Component interaction testing
- Data flow validation
- Configuration propagation

#### Performance Benchmarks
- Execution time measurements
- Memory usage tracking
- Scalability validation
- Regression detection

#### Security Tests
- Input sanitization verification
- File operation safety
- Vulnerability scanning
- Attack vector validation

## Running Tests

### Basic Test Execution
```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m benchmark     # Performance benchmarks
pytest -m security      # Security tests

# Skip slow tests
pytest -m "not slow"

# Run with coverage
pytest --cov=liquid_metal_antenna
```

### Quality Gates
```bash
# Run complete quality validation
python quality_gates.py

# Fast mode (skip slow tests)
python quality_gates.py --fast

# Save results to file
python quality_gates.py --output quality_results.json
```

## Quality Metrics

### Coverage Targets
- **Test Coverage**: ≥80%
- **Branch Coverage**: ≥75%
- **Function Coverage**: ≥90%

### Performance Benchmarks
- **FDTD Setup**: <0.5s
- **Optimization Iteration**: <0.1s/iteration
- **Cache Hit Time**: <1ms
- **Memory Efficiency**: ≥80%

### Security Standards
- **Input Validation**: 100% of user inputs sanitized
- **File Operations**: All file access through secure handlers
- **Vulnerability Scanning**: Zero high-severity issues

### Code Quality
- **Linting Score**: ≥8.0/10
- **Documentation Coverage**: ≥70%
- **API Completeness**: ≥90%

## Test Configuration

### pytest.ini
- Configures test discovery patterns
- Sets up markers for test categorization
- Defines coverage reporting
- Configures parallel execution

### Fixtures and Utilities
- Geometry generators for consistent test data
- Mock solvers for isolated testing
- Performance monitors for benchmarking
- Security validators for safety checks

## Continuous Integration

The testing infrastructure supports:
- Automated test execution on code changes
- Performance regression detection
- Security vulnerability scanning
- Quality gate enforcement
- Coverage reporting and tracking

## Development Workflow

1. **Write Tests First**: Follow TDD principles
2. **Run Quality Gates**: Ensure all gates pass before commit
3. **Monitor Performance**: Track benchmark trends
4. **Security Review**: Validate all security tests pass
5. **Integration Validation**: Verify end-to-end functionality

## Troubleshooting

### Common Issues
- **Import Errors**: Ensure all dependencies installed
- **GPU Tests Failing**: Use `-m "not gpu"` if no GPU available
- **Slow Test Timeout**: Increase timeout in pytest.ini
- **Memory Issues**: Use smaller test geometries

### Debug Mode
```bash
# Verbose output with debug information
pytest -v -s --tb=long

# Run single test with debugging
pytest -v tests/test_core_antenna_spec.py::TestAntennaSpec::test_basic_antenna_spec_creation
```

## Test Data Management

- **Reproducible**: Fixed random seeds for consistent results  
- **Scalable**: Multiple geometry sizes for scaling tests
- **Realistic**: Based on actual antenna design constraints
- **Comprehensive**: Edge cases and error conditions covered

## Quality Assurance

The testing infrastructure ensures:
- **Correctness**: All algorithms produce expected results
- **Robustness**: System handles edge cases gracefully  
- **Performance**: Meets speed and memory requirements
- **Security**: Input validation prevents vulnerabilities
- **Maintainability**: Tests are clear and well-documented

## Extending Tests

When adding new functionality:
1. Add unit tests for individual functions
2. Include integration tests for component interactions
3. Add performance benchmarks for critical paths
4. Include security tests for user-facing interfaces
5. Update quality gate thresholds as needed

This comprehensive testing approach ensures the liquid metal antenna optimizer maintains high quality, performance, and security standards throughout development and deployment.