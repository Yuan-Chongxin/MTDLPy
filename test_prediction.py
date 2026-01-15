#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Test Prediction Script for MTDLPy

This script provides a simple way to test model prediction functionality.
It loads a trained model and performs prediction on test data.

Usage:
    python test_prediction.py [model_path] [file_number] [mt_mode]

Arguments:
    model_path: Path to the trained model file (.pkl) (optional, will search in models/ if not provided)
    file_number: File number (e.g., 2) to load from data/[mode]/resistivity/ and data/[mode]/phase/ (optional)
    mt_mode: MT mode - 'TE', 'TM', or 'Both' (optional, default: 'TM')

Examples:
    # Use default model and file number 2 in TM mode
    python test_prediction.py
    
    # Specify model and file number 2 in TM mode
    python test_prediction.py models/DinkNet_best.pkl 2
    
    # Specify model, file number 2, and TE mode
    python test_prediction.py models/DinkNet_best.pkl 2 TE

Note:
    The script will automatically load:
    - Resistivity data from: data/[mode]/resistivity/[file_number].txt
    - Phase data from: data/[mode]/phase/[file_number].txt

Author: MTDLPy Team
Creation Time: 2024
"""

import sys
import os
import torch
import numpy as np
import datetime
from scipy.ndimage import zoom

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main function for test prediction"""
    print("=" * 60)
    print("MTDLPy Test Prediction Script")
    print("=" * 60)
    
    try:
        # Import required modules
        print("\n[1/4] Importing required modules...")
        from ParamConfig import Nclasses, DataDim, ModelDim, TM_Resistivity_Dir, TM_Phase_Dir, TE_Resistivity_Dir, TE_Phase_Dir
        from PathConfig import models_dir, results_dir
        import LibConfig
        from func.dinknet import DinkNet50
        from func.UnetModel import UnetModel
        # Import load_test_data from MT_test.py
        from MT_test import load_test_data
        print("✓ All modules imported successfully")
        
        # Get command line arguments
        model_path = None
        file_number = None
        mt_mode = 'TM'  # Default mode
        
        if len(sys.argv) > 1:
            # First argument could be model path or file number
            arg1 = sys.argv[1]
            if arg1.endswith('.pkl') or os.path.exists(arg1):
                model_path = arg1
                if len(sys.argv) > 2:
                    file_number = sys.argv[2]
            else:
                # First argument is file number
                file_number = arg1
        if len(sys.argv) > 2 and model_path:
            # Second argument is file number if first was model path
            file_number = sys.argv[2]
        if len(sys.argv) > 3:
            mt_mode = sys.argv[3]
        elif len(sys.argv) > 2 and not model_path:
            # If first arg was file number, second might be mode
            if sys.argv[2] in ['TE', 'TM', 'Both']:
                mt_mode = sys.argv[2]
        
        # If no model path provided, try to find the latest model
        if not model_path:
            print("\n[2/4] Searching for model files...")
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
                if model_files:
                    # Get the most recent model file
                    model_path = max([os.path.join(models_dir, f) for f in model_files], 
                                   key=os.path.getmtime)
                    print(f"✓ Found model: {os.path.basename(model_path)}")
                else:
                    print("✗ No model files found in models directory")
                    print(f"  Please provide a model path or place a .pkl file in: {models_dir}")
                    return False
            else:
                print(f"✗ Models directory not found: {models_dir}")
                print("  Please provide a model path as the first argument")
                return False
        else:
            if not os.path.exists(model_path):
                print(f"✗ Model file not found: {model_path}")
                return False
            print(f"✓ Using model: {model_path}")
        
        # Load model
        print("\n[3/4] Loading model...")
        
        # Check GPU availability and select device (prioritize GPU)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"  Using device: {device}")
        
        # Determine model type and number of channels from filename
        model_filename = os.path.basename(model_path)
        num_channels = 2  # Default
        
        if 'ModeBoth' in model_filename or 'Both' in model_filename:
            num_channels = 4
            if mt_mode != 'Both':
                mt_mode = 'Both'
                print(f"  Model filename indicates Both mode, setting mt_mode to 'Both'")
        elif 'ModeTE' in model_filename or 'TE' in model_filename:
            num_channels = 2
            if mt_mode != 'TE':
                mt_mode = 'TE'
                print(f"  Model filename indicates TE mode, setting mt_mode to 'TE'")
        elif 'ModeTM' in model_filename or 'TM' in model_filename:
            num_channels = 2
            if mt_mode != 'TM':
                mt_mode = 'TM'
                print(f"  Model filename indicates TM mode, setting mt_mode to 'TM'")
        
        # Determine model architecture
        if 'Unet' in model_filename or 'unet' in model_filename.lower():
            net = UnetModel(n_classes=Nclasses, in_channels=num_channels)
            print(f"  Using UnetModel with {num_channels} input channels")
        else:
            net = DinkNet50(num_classes=Nclasses, num_channels=num_channels)
            print(f"  Using DinkNet50 with {num_channels} input channels")
        
        # Load model weights
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.to(device)
        net.eval()
        print("✓ Model loaded successfully")
        
        # Load test data
        print(f"\n[4/4] Loading test data (MT mode: {mt_mode})...")
        
        # Determine test data file path
        test_data_file = None
        if file_number:
            # Build path based on mode and file number
            try:
                if mt_mode == 'TM':
                    resistivity_dir = TM_Resistivity_Dir if TM_Resistivity_Dir else 'data/TM/resistivity'
                elif mt_mode == 'TE':
                    resistivity_dir = TE_Resistivity_Dir if TE_Resistivity_Dir else 'data/TE/resistivity'
                else:
                    # Both mode - use TE as default
                    resistivity_dir = TE_Resistivity_Dir if TE_Resistivity_Dir else 'data/TE/resistivity'
            except NameError:
                # Fallback to default paths
                if mt_mode == 'TM':
                    resistivity_dir = 'data/TM/resistivity'
                elif mt_mode == 'TE':
                    resistivity_dir = 'data/TE/resistivity'
                else:
                    resistivity_dir = 'data/TE/resistivity'
            
            test_data_file = os.path.join(resistivity_dir, f"{file_number}.txt")
            
            if not os.path.exists(test_data_file):
                print(f"✗ Test data file not found: {test_data_file}")
                print(f"  Please check if the file exists or provide a valid file number")
                return False
            print(f"  Using test data file: {test_data_file}")
        else:
            print("  No file number provided, will try to use default data loading...")
            # Try to find a default file (e.g., 0.txt or 2.txt)
            try:
                if mt_mode == 'TM':
                    resistivity_dir = TM_Resistivity_Dir if TM_Resistivity_Dir else 'data/TM/resistivity'
                elif mt_mode == 'TE':
                    resistivity_dir = TE_Resistivity_Dir if TE_Resistivity_Dir else 'data/TE/resistivity'
                else:
                    resistivity_dir = TE_Resistivity_Dir if TE_Resistivity_Dir else 'data/TE/resistivity'
            except NameError:
                # Fallback to default paths
                if mt_mode == 'TM':
                    resistivity_dir = 'data/TM/resistivity'
                elif mt_mode == 'TE':
                    resistivity_dir = 'data/TE/resistivity'
                else:
                    resistivity_dir = 'data/TE/resistivity'
            
            # Try common file numbers
            for num in ['2', '0', '1']:
                test_file = os.path.join(resistivity_dir, f"{num}.txt")
                if os.path.exists(test_file):
                    test_data_file = test_file
                    file_number = num
                    print(f"  Found default test data file: {test_data_file}")
                    break
        
        if not test_data_file:
            print("✗ Could not find test data file")
            print(f"  Please provide a file number (e.g., 2) or ensure data files exist in: {resistivity_dir}")
            return False
        
        # Use load_test_data from MT_test.py
        test_set, data_dsp_dim, label_dsp_dim, file_info = load_test_data(
            test_data_file=test_data_file,
            mt_mode=mt_mode
        )
        
        print(f"  Test data shape: {test_set.shape}")
        print(f"  Data dimensions: {data_dsp_dim}")
        print(f"  Label dimensions: {label_dsp_dim}")
        print("✓ Test data loaded successfully")
        
        # Perform prediction
        print("\n" + "=" * 60)
        print("Starting Prediction...")
        print("=" * 60)
        
        # Convert to tensor
        test_tensor = torch.FloatTensor(test_set).to(device)
        batch_size = test_tensor.shape[0]
        in_channels = test_tensor.shape[1]
        flattened_size = test_tensor.shape[2]  # data_dsp_dim[0] * data_dsp_dim[1]
        target_total = label_dsp_dim[0] * label_dsp_dim[1]
        
        print(f"  Input shape: {test_tensor.shape}")
        print(f"  Flattened size: {flattened_size}, Target total: {target_total}")
        
        # Reshape for model input (same logic as MT_test.py)
        if flattened_size == target_total:
            # Total elements match, can directly view
            print(f"  Reshaping using view (total elements match)")
            test_tensor = test_tensor.view(batch_size, in_channels, label_dsp_dim[0], label_dsp_dim[1])
        else:
            # Total elements differ, need interpolation
            print(f"  Warning: Total elements differ, using interpolation")
            # First reshape to data_dsp_dim
            # Calculate data_dsp_dim from flattened_size (assuming square)
            data_dsp_size = int(np.sqrt(flattened_size))
            test_tensor_2d = test_tensor.view(batch_size, in_channels, data_dsp_size, data_dsp_size)
            # Convert to numpy for interpolation
            test_array = test_tensor_2d.cpu().numpy()
            reshaped_data = np.zeros((batch_size, in_channels, label_dsp_dim[0], label_dsp_dim[1]), dtype=np.float32)
            for b in range(batch_size):
                for c in range(in_channels):
                    scale_factors = (label_dsp_dim[0] / data_dsp_size, label_dsp_dim[1] / data_dsp_size)
                    reshaped_data[b, c] = zoom(test_array[b, c], scale_factors, order=1)
            test_tensor = torch.from_numpy(reshaped_data).float().to(device)
        
        print(f"  Reshaped tensor shape: {test_tensor.shape}")
        
        # Predict
        with torch.no_grad():
            if 'Unet' in net.__class__.__name__:
                outputs = net(test_tensor, label_dsp_dim)
            else:
                outputs = net(test_tensor)
            
            # Process output
            if len(outputs.shape) == 4:
                if outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)
                elif outputs.shape[1] == Nclasses and Nclasses == 1:
                    outputs = outputs.squeeze(1)
            
            # Convert to numpy
            predictions = outputs.cpu().numpy()
            
            # Ensure predictions is 2D or 3D
            if len(predictions.shape) == 2:
                predictions = predictions[np.newaxis, :, :]
            elif len(predictions.shape) == 1:
                predictions = predictions.reshape(label_dsp_dim[0], label_dsp_dim[1])
                predictions = predictions[np.newaxis, :, :]
        
        print(f"✓ Prediction completed")
        print(f"  Prediction shape: {predictions.shape}")
        print(f"  Prediction value range: min={np.min(predictions):.6f}, max={np.max(predictions):.6f}")
        
        # Save results
        print("\nSaving prediction results...")
        os.makedirs(results_dir, exist_ok=True)
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"test_pred_{timestamp}.txt"
        result_path = os.path.join(results_dir, result_filename)
        
        # Save as text file
        np.savetxt(result_path, predictions.flatten(), fmt='%.6f')
        print(f"✓ Results saved to: {result_path}")
        
        print("\n" + "=" * 60)
        print("Test Prediction Completed Successfully! ✓")
        print("=" * 60)
        print(f"\nResults saved to: {result_path}")
        print(f"Prediction shape: {predictions.shape}")
        print("=" * 60)
        
        return True
        
    except ImportError as e:
        print(f"\n✗ Import Error: {e}")
        print("  Please make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        return False
    except FileNotFoundError as e:
        print(f"\n✗ File Not Found: {e}")
        print("  Please check the file paths and try again")
        return False
    except Exception as e:
        print(f"\n✗ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
