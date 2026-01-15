#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick Test Script for MTDLPy System

This script performs a quick test to verify that the MTDLPy system is properly
installed and all required modules can be imported correctly.

Author: MTDLPy Team
Creation Time: 2024
"""

import sys
import os

# Fix Windows console encoding issue
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def test_imports():
    """Test if all required modules can be imported"""
    print("=" * 60)
    print("MTDLPy Quick Test")
    print("=" * 60)
    print("\n[1/5] Testing Python version...")
    print(f"Python version: {sys.version}")
    
    if sys.version_info < (3, 7):
        print("ERROR: Python 3.7 or higher is required!")
        return False
    print("✓ Python version OK")
    
    print("\n[2/5] Testing core scientific libraries...")
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import NumPy: {e}")
        return False
    
    try:
        import scipy
        print(f"✓ SciPy {scipy.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SciPy: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Matplotlib: {e}")
        return False
    
    print("\n[3/5] Testing deep learning framework...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported successfully")
        if torch.cuda.is_available():
            print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  CUDA not available (CPU mode only)")
    except ImportError as e:
        print(f"✗ Failed to import PyTorch: {e}")
        return False
    
    print("\n[4/5] Testing GUI framework...")
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import Qt
        print("✓ PyQt5 imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PyQt5: {e}")
        print("  Note: GUI functionality will not be available")
    
    print("\n[5/5] Testing MTDLPy configuration modules...")
    try:
        from ParamConfig import DataDim, ModelDim, Epochs, TrainSize
        print("✓ ParamConfig module imported successfully")
        print(f"  DataDim: {DataDim}")
        print(f"  ModelDim: {ModelDim}")
        print(f"  Epochs: {Epochs}")
        print(f"  TrainSize: {TrainSize}")
    except ImportError as e:
        print(f"✗ Failed to import ParamConfig: {e}")
        return False
    
    try:
        from PathConfig import models_dir, results_dir
        print("✓ PathConfig module imported successfully")
    except ImportError as e:
        print(f"⚠ Failed to import PathConfig: {e}")
        print("  Note: This may be due to missing optional dependencies")
        # Don't fail the test for PathConfig, as it may have optional dependencies
    
    try:
        import LibConfig
        print("✓ LibConfig module imported successfully")
    except ImportError as e:
        print(f"⚠ Failed to import LibConfig: {e}")
        print("  Note: This may be due to missing optional dependencies")
        # Don't fail the test for LibConfig, as it imports many modules
    
    print("\n" + "=" * 60)
    print("All basic tests passed! ✓")
    print("=" * 60)
    print("\nThe MTDLPy system is ready to use.")
    print("You can now run the GUI by executing: python MTDLPy_GUI.py")
    print("=" * 60)
    
    return True

def test_model_modules():
    """Test if model modules can be imported"""
    print("\n[Optional] Testing model modules...")
    try:
        from func.UnetModel import UnetModel
        print("✓ UnetModel imported successfully")
    except ImportError as e:
        print(f"⚠ UnetModel import failed: {e}")
    
    try:
        from func.dinknet import DinkNet50
        print("✓ DinkNet50 imported successfully")
    except ImportError as e:
        print(f"⚠ DinkNet50 import failed: {e}")
    
    return True

if __name__ == "__main__":
    print("\nStarting MTDLPy Quick Test...\n")
    
    success = test_imports()
    
    if success:
        test_model_modules()
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("Test failed! Please check the error messages above.")
        print("Make sure all dependencies are installed correctly.")
        print("=" * 60)
        sys.exit(1)
