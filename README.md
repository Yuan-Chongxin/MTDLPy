# MTDLPy - 大地电磁深度学习反演系统

MTDLPy 是一个面向大地电磁（MT）数据的深度学习反演工具，提供 **GUI 图形界面 + 命令行脚本** 两种工作方式，支持：

- TE 模式
- TM 模式
- TE+TM 联合（Both）模式

当前版本以 `MTDLPy_GUI.py`、`MT_train.py`、`MT_test.py` 为核心流程。

---

## 1. 主要功能

- **数据导入与可视化**：在 GUI 中导入 TE/TM 视电阻率、相位、标签（电阻率模型）数据目录，并支持右键可视化。
- **参数配置**：选择模型（`UnetModel`、`DinkNet`、`UnetPlusPlus`），配置 Batch Size、Learning Rate、Train/Validation 比例、Epoch 等。
- **模型训练**：GUI 启动训练时调用 `MT_train.py`，支持训练进度、日志、损失曲线显示，自动保存最佳模型与最终模型。
- **模型预测**：GUI 调用 `MT_test.py` 进行预测，支持根据模型文件自动识别输入通道（2 通道或 4 通道）与模式兼容性检查。
- **结果导出**：预测结果可导出为 `*.dat`（pyGIMLI 兼容）、`*.grd`（Surfer Grid）、`*.txt`。
- **快速环境检测**：`quick_test.py` 可快速检查 Python、PyTorch、PyQt5 及配置模块是否可用。

---

## 2. 项目结构（当前常用）

```text
dl/
├── MTDLPy_GUI.py          # GUI 主程序（数据导入/参数配置/训练/预测）
├── MT_train.py            # 训练入口脚本
├── MT_test.py             # 预测入口脚本
├── quick_test.py          # 环境快速检测
├── test_prediction.py     # 简化预测测试脚本
├── ml_trainer.py          # GUI 训练线程与训练日志解析
├── ParamConfig.py         # 训练/数据/路径参数（核心配置）
├── PathConfig.py          # 路径与模型命名相关配置
├── LibConfig.py           # 公共依赖导入
├── requirements.txt       # 依赖列表
├── models/                # 模型输出目录（自动创建）
├── results/               # 结果输出目录（自动创建）
└── func/
    ├── DataLoad_Train.py  # 训练数据加载
    ├── DataLoad_Test.py   # 测试数据加载（旧流程）
    ├── dinknet.py
    ├── UnetModel.py
    └── unetplusplus.py
```

---

## 3. 环境要求

- Python >= 3.7
- Windows（当前 GUI 与路径默认配置以 Windows 为主）
- 建议使用虚拟环境

---

## 4. 安装依赖

推荐：

```bash
pip install -r requirements.txt
```

`requirements.txt` 主要包括：

- PyQt5
- numpy / scipy / pandas / scikit-learn
- matplotlib
- opencv-python / scikit-image
- torch / torchvision

---

## 5. 快速自检

在项目根目录运行：

```bash
python quick_test.py
```

通过后再启动 GUI 或命令行流程。

---

## 6. GUI 使用流程（推荐）

启动：

```bash
python MTDLPy_GUI.py
```

GUI 主要有 4 个标签页：

1. **Data Import**
   - 选择模式：TE / TM / Both
   - 导入输入数据目录（视电阻率、相位）与标签目录（电阻率模型）
   - 要求各目录文件数匹配（按模式规则检查）
2. **Param Config**
   - 模型：`UnetModel` / `DinkNet` / `UnetPlusPlus`
   - 参数：`BatchSize`、`LearningRate`、`TrainSize`、`ValSize`、`Epochs`、`Optimizer`
3. **Model Training**
   - 开始/暂停/继续/停止训练
   - 实时查看训练日志、进度与损失曲线
4. **Model Prediction**
   - 导入预测输入文件和模型文件（`.pkl/.pt/.pth`）
   - 调用 `MT_test.py` 生成结果
   - 导出反演结果与剖面图

---

## 7. 命令行使用

### 7.1 训练

```bash
python MT_train.py
```

训练参数主要来自 `ParamConfig.py`（如 `Epochs`、`BatchSize`、`LearnRate`、`MT_Mode` 等）。

### 7.2 预测

```bash
python MT_test.py <test_data_file> <model_file> <MT_mode>
```

示例：

```bash
python MT_test.py H:/sdzl/2.txt models/DinkNet_best.pkl TM
```

也可使用简化脚本：

```bash
python test_prediction.py [model_path] [file_number] [mt_mode]
```

---

## 8. 数据与命名约定（当前代码实现）

训练与预测主流程默认使用 `*.txt` 数据。常见约定：

- 输入数据目录中按编号文件：`1.txt`、`2.txt`、`3.txt`...
- 标签目录中模型文件名：`zz1.txt`、`zz2.txt`、`zz3.txt`...
- TE/TM/Both 模式下，相关目录文件数需一致（GUI 会校验）

训练前处理要点：

- 视电阻率会做 `log10` 转换；
- 相位保持原值（不做 0~1 归一化）；
- 标签（电阻率模型）也做 `log10` 转换；
- 使用 `ParamConfig.py` 的网格参数（如 `DataDim`、`ModelDim`、`RawGridShape`）进行重采样/对齐。

---

## 9. 关键配置说明（`ParamConfig.py`）

常用项：

- `MT_Mode`：`TE` / `TM` / `Both`
- `Inchannels`：2（TE/TM）或 4（Both）
- `DataDim`、`ModelDim`
- `RawGridShape`、`RawGridTransposeBeforeResize`
- `PredictionOutputSpatialFix`
- `TrainSize`、`ValSize`
- `Epochs`、`BatchSize`、`LearnRate`
- `ModelName`
- `TE_Resistivity_Dir`、`TE_Phase_Dir`、`TM_Resistivity_Dir`、`TM_Phase_Dir`、`Resistivity_Model_Dir`

> 注意：当前仓库中的部分默认路径是本地绝对路径（如 `H:/...`、`M:/...`）。在你的机器上使用前，请先改成实际可访问目录。

---

## 10. 输出结果

- 模型文件：默认输出到 `models/`
  - 最佳验证集模型（`...best_val_epochX.pkl`）
  - 最终模型（`...final.pkl`）
  - 便捷副本（如 `DinkNet_best.pkl`）
- 预测结果：默认输出到 `results/`
  - 预测结果文本（包含时间戳）
  - 配置记录 JSON

---

## 11. 常见问题

- **Q：训练报找不到数据目录？**  
  A：先检查 `ParamConfig.py` 中数据目录路径，并确认目录下有对应编号的 `*.txt` 文件。

- **Q：预测时模式不匹配？**  
  A：2 通道模型只能用于 TE/TM；4 通道模型用于 Both。建议通过模型文件名中的 `ModeTE/ModeTM/ModeBoth` 区分。

- **Q：结果方向看起来有旋转/转置问题？**  
  A：调整 `PredictionOutputSpatialFix`（`none` / `transpose` / `rot90_cw` / `rot90_ccw`）后重新预测。

---

## 12. License

本项目采用 [MIT License](LICENSE)。
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
