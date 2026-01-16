# MTDLPy - Magnetotelluric Deep Learning System

## Why Use This Code?

MTDLPy (Magnetotelluric Deep Learning) is a deep learning inversion system for magnetotelluric (MT) data processing. This code is designed to address the following research and practical needs:

1. **Efficient MT Data Processing**: Provides an end-to-end solution for MT data inversion, from raw data import to geological interpretation, significantly reducing the time and expertise required for traditional inversion methods.

2. **User-Friendly Interface**: Offers a graphical user interface that eliminates the need for programming knowledge, making advanced MT inversion techniques accessible to geophysicists and researchers.

3. **Reproducible Research**: All code is open-source and well-documented, enabling researchers to reproduce results and build upon this work.

## Project Introduction

MTDLPy (Magnetotelluric Deep Learning) is a deep learning inversion system for magnetotelluric (MT) data processing. The system combines physical constraints with deep learning technology to provide efficient and accurate solutions for MT data processing and geological interpretation.

## System Architecture

The MTDLPy system adopts a three-tier architecture design:

1. **Bottom Layer**: PyTorch-based computation core, physical constraint modules, and data processing components
2. **Middle Layer**: Python API interface framework connecting bottom-layer computation functions with top-layer GUI interaction
3. **Top Layer**: PyQt5-based graphical user interface providing full-process visualization operations

## Features

- **Physics-Data Dual Drive**: Embedding electromagnetic physics constraints into deep learning models to ensure inversion results comply with physical laws
- **End-to-End Workflow**: Full-process visualization operations from data import to result export
- **Zero Programming Threshold**: Complete the full process from data preparation to geological interpretation without writing code
- **Professional Model Support**: Deep learning models optimized for magnetotelluric data characteristics
- **Real-time Monitoring**: Real-time monitoring of training process and loss curve display
- **Multi-mode Support**: Support for TE mode, TM mode, and simultaneous processing of TE and TM modes

## Pre-trained Model Download

The pre-trained DinkNet model (optimized for MT data inversion) can be downloaded via the following link:
- File name: DinkNet_best.pkl
- Baidu Netdisk link: https://pan.baidu.com/s/1sCoV5IiYiNM4chC9h9ufIA 
- Extraction code: MTDL


## Installation Guide

### Method 1: Manual Installation (Recommended)

#### 1. Create a Python Virtual Environment

```bash
# Create virtual environment
python -m venv pdmtdl_env

# Activate virtual environment (Windows)
pdmtdl_env\Scripts\activate

# Activate virtual environment (Linux/Mac)
# source pdmtdl_env/bin/activate
```

#### 2. Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install necessary dependency libraries (using mirror source for acceleration)
# For CPU version (default):
pip install PyQt5 numpy matplotlib scipy torch torchvision opencv-python scikit-image pandas ipython -i https://pypi.tuna.tsinghua.edu.cn/simple

# For GPU version (recommended for faster training):
pip install PyQt5 numpy matplotlib scipy opencv-python scikit-image pandas ipython -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch>=1.10.0+cu117 torchvision>=0.11.0+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
```

## How to Use This Code

### Quick Test

Before using the full system, we recommend running the quick test to verify that all dependencies are properly installed:

```bash
python quick_test.py
```

This test will check:
- Python version compatibility
- Required scientific libraries (NumPy, SciPy, Matplotlib)
- Deep learning framework (PyTorch)
- GUI framework (PyQt5)
- MTDLPy configuration modules

If all tests pass, the system is ready to use.

### Start the System

After installation is complete, you can start the system through the following methods:

1. **GUI Mode (Recommended)**: After activating the virtual environment, execute in the command line:
   ```bash
   python MTDLPy_GUI.py
   ```

2. **Command-Line Mode**: Directly use command-line tools for training and prediction

### Usage Flow

1. **Data Import**: Select MT mode and import data files in the "Data Import" tab
2. **Model Configuration**: Select model and set physical constraint parameters in the "Param Config" tab
3. **Model Training**: Start training and monitor progress in the "Model Training" tab
4. **Model Prediction**: Load trained model for prediction and export results in the "Model Prediction" tab

### Command-Line Usage

For advanced users, the system can also be used via command line:

**Training:**
```bash
python MT_train.py
```

**Testing/Prediction:**
```bash
python MT_test.py
```

## File Structure

```
dl/
├── MTDLPy_GUI.py       # Main GUI program
├── quick_test.py       # Quick test script
├── requirements.txt    # Dependencies list
├── LICENSE             # Open-source license
├── README.md           # Project description document
├── ParamConfig.py      # Parameter configuration file
├── PathConfig.py       # Path configuration file
├── LibConfig.py        # Library configuration file
├── ml_trainer.py       # Machine learning training module
├── MT_train.py         # Main training program
├── MT_test.py          # Testing program
├── func/               # Function modules
│   ├── UnetModel.py    # Unet model definition
│   ├── dinknet.py      # DinkNet model definition
│   ├── DataLoad_Train.py # Training data loading module
│   ├── DataLoad_Test.py  # Testing data loading module
│   └── utils.py        # Utility functions
├── nn/                 # Neural network modules
│   ├── resnetv1b.py    # ResNet backbone network
│   ├── vgg.py          # VGG backbone network
│   ├── fcn.py          # FCN network
│   └── jpu.py          # JPU module
├── cei/                # CeiT model module
│   ├── ceit.py         # CeiT model definition
│   ├── module.py       # Module definition
│   └── train.py        # Training script
├── data/               # Data directory
│   ├── TE/             # TE mode data
│   │   ├── phase/      # TE phase data
│   │   └── resistivity/ # TE apparent resistivity data
│   ├── TM/             # TM mode data
│   │   ├── phase/      # TM phase data
│   │   └── resistivity/ # TM apparent resistivity data
│   └── models/         # Model data
├── models/             # Trained model storage directory
└── results/            # Results output directory
```

## IDE Configuration (Optional)

If you need to run the project in PyCharm, VSCode, or other IDEs, set the interpreter to Python in the virtual environment:

```
<Project Path>\pdmtdl_env\Scripts\python.exe
```

## Troubleshooting

### Virtual Environment Creation Failure

If virtual environment creation fails, try the following methods:

1. Make sure you have Python 3.7 or higher installed
2. Try creating a virtual environment manually:
   ```bash
   python -m venv pdmtdl_env
   ```
3. If Python is installed in a specific location, use the full path:
   ```
   D:\Python37\python.exe -m venv pdmtdl_env
   ```

### PyQt5 Import Error

If you encounter "ImportError: Failed to import any qt binding" error, please:

1. Make sure you are running the program in the correct virtual environment
2. Reinstall PyQt5:
   ```
   pdmtdl_env\Scripts\activate
   pip install PyQt5 -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

- Virtual environment Python path: `pdmtdl_env\Scripts\python.exe`

### PyCharm Configuration Steps

1. Open PyCharm, click "File" -> "Settings" -> "Project: dl" -> "Python Interpreter"
2. Click the gear icon in the upper right corner and select "Add"
3. Select "Existing environment", click the "..." button
4. Navigate to `pdmtdl_env\Scripts\python.exe`, click "OK"
5. Click "Apply" -> "OK" to complete configuration

### VSCode Configuration Steps

1. Open VSCode, click the Python version number in the lower left corner
2. In the pop-up interpreter selection menu, select "Enter interpreter path..."
3. Click "Find..."
4. Navigate to `pdmtdl_env\Scripts\python.exe`, click "Select Interpreter"

## Frequently Asked Questions

### Q: What should I do if permission errors occur during installation?
A: Installing with a virtual environment can avoid most permission issues. It is recommended to manually create a virtual environment:
   ```bash
   python -m venv pdmtdl_env
   ```

### Q: What should I do if out-of-memory errors occur during training?
A: Reduce the batch size or use a smaller model, and you can also close other memory-intensive programs.

### Q: Why isn't GPU training starting even though I have a GPU?
A: 1. Ensure you have installed the GPU version of PyTorch
   2. Check if CUDA is properly installed by running `nvidia-smi`
   3. Verify PyTorch can access GPU by running `python -c "import torch; print(torch.cuda.is_available())"`
   4. Make sure you have the correct CUDA version installed (compatible with your PyTorch version)

### Q: How to check if training is using GPU?
A: During training, you can see GPU utilization in the output logs. You can also monitor GPU usage with `nvidia-smi` command.

### Q: How to adjust the weight of physical constraints?
A: In the "Param Config" tab, use the slider to adjust the weight of physical loss in the total loss (range 0~1).

### Q: What formats of data files can be imported?
A: The system supports .dat, .mat, and .npy format data files.

### Q: How to export inversion results?
A: In the "Model Prediction" tab, click the "Export Prediction Results" button to export inversion results as .dat files, and click the "Export Profile Diagram" button to export result images as .png, .jpg, or .svg formats.

## System Requirements

- Windows 7/10/11 operating system
- Python 3.7 or higher
- CPU version: Minimum 8GB RAM, recommended 16GB or more
- GPU version (recommended for faster training):
  - NVIDIA GPU with CUDA 11.0+ support
  - Minimum 6GB VRAM (10GB or more recommended for large datasets)
  - Latest NVIDIA drivers installed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{mtdlpy2024,
  title={Magnetotelluric Deep Learning Inversion System},
  author={MTDLPy Team},
  journal={Computers and Geosciences},
  year={2024}
}
```

## Disclaimer

This software is for academic research use only. The authors are not responsible for any results generated by using this software.

---
© 2024 MTDLPy Team. All rights reserved.
