#!/usr/bin/env python3
"""
Simple import test without dependencies.
"""

import sys
import os

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_package_structure():
    """Test that package structure is correct."""
    
    # Check if main package directory exists
    package_dir = "liquid_metal_antenna"
    assert os.path.exists(package_dir), f"Package directory {package_dir} not found"
    
    # Check core modules
    core_files = [
        "liquid_metal_antenna/__init__.py",
        "liquid_metal_antenna/core/__init__.py", 
        "liquid_metal_antenna/core/antenna_spec.py",
        "liquid_metal_antenna/core/optimizer.py",
        "liquid_metal_antenna/solvers/__init__.py",
        "liquid_metal_antenna/solvers/base.py",
        "liquid_metal_antenna/solvers/fdtd.py",
        "liquid_metal_antenna/designs/__init__.py",
        "liquid_metal_antenna/designs/patch.py",
        "liquid_metal_antenna/designs/monopole.py",
        "liquid_metal_antenna/designs/array.py",
        "liquid_metal_antenna/designs/metamaterial.py",
        "liquid_metal_antenna/liquid_metal/__init__.py",
        "liquid_metal_antenna/liquid_metal/materials.py",
        "liquid_metal_antenna/liquid_metal/flow.py",
    ]
    
    for file_path in core_files:
        assert os.path.exists(file_path), f"Core file {file_path} not found"
        
        # Check file is not empty
        with open(file_path, 'r') as f:
            content = f.read().strip()
            assert len(content) > 0, f"File {file_path} is empty"
    
    print("✓ Package structure is correct")

def test_example_files():
    """Test that example files exist."""
    example_files = [
        "examples/basic_usage.py",
    ]
    
    for file_path in example_files:
        assert os.path.exists(file_path), f"Example file {file_path} not found"
    
    print("✓ Example files exist")

def test_config_files():
    """Test configuration files."""
    config_files = [
        "pyproject.toml",
        "liquid_metal_antenna/_version.py",
    ]
    
    for file_path in config_files:
        assert os.path.exists(file_path), f"Config file {file_path} not found"
    
    print("✓ Configuration files exist")

def test_file_contents():
    """Test basic file contents."""
    
    # Check pyproject.toml has correct name
    with open("pyproject.toml", 'r') as f:
        content = f.read()
        assert "liquid-metal-antenna-opt" in content, "Package name not found in pyproject.toml"
    
    # Check version file
    with open("liquid_metal_antenna/_version.py", 'r') as f:
        content = f.read()
        assert "__version__" in content, "Version not defined"
    
    # Check main __init__.py has imports
    with open("liquid_metal_antenna/__init__.py", 'r') as f:
        content = f.read()
        assert "AntennaSpec" in content, "AntennaSpec not exported"
        assert "LMAOptimizer" in content, "LMAOptimizer not exported"
    
    print("✓ File contents are correct")

def test_class_definitions():
    """Test that key classes are defined in files."""
    
    # Check antenna_spec.py
    with open("liquid_metal_antenna/core/antenna_spec.py", 'r') as f:
        content = f.read()
        assert "class AntennaSpec" in content, "AntennaSpec class not found"
        assert "class SubstrateMaterial" in content, "SubstrateMaterial enum not found"
    
    # Check optimizer.py
    with open("liquid_metal_antenna/core/optimizer.py", 'r') as f:
        content = f.read()
        assert "class LMAOptimizer" in content, "LMAOptimizer class not found"
    
    # Check FDTD solver
    with open("liquid_metal_antenna/solvers/fdtd.py", 'r') as f:
        content = f.read()
        assert "class DifferentiableFDTD" in content, "DifferentiableFDTD class not found"
    
    # Check patch antenna
    with open("liquid_metal_antenna/designs/patch.py", 'r') as f:
        content = f.read()
        assert "class ReconfigurablePatch" in content, "ReconfigurablePatch class not found"
    
    print("✓ Key classes are defined")

if __name__ == "__main__":
    print("Testing Liquid Metal Antenna Optimizer Package Structure...")
    print("=" * 60)
    
    try:
        test_package_structure()
        test_example_files() 
        test_config_files()
        test_file_contents()
        test_class_definitions()
        
        print("\n" + "=" * 60)
        print("✓ ALL STRUCTURE TESTS PASSED!")
        print("Package is ready for Generation 2 development")
        
    except AssertionError as e:
        print(f"✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        sys.exit(1)