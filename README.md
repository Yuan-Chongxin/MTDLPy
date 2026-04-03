# MTDLPy - Magnetotelluric Deep Learning Inversion System

MTDLPy is a deep learning inversion toolkit for magnetotelluric (MT) data.  
It supports both **GUI workflow** and **command-line scripts**, with three MT modes:

- TE mode
- TM mode
- Both mode (TE + TM)

The current core workflow is based on `MTDLPy_GUI.py`, `MT_train.py`, and `MT_test.py`.

---

## 1. Features

- **Data import and visualization** in GUI (TE/TM apparent resistivity, phase, and label model data).
- **Model/parameter configuration** for `UnetModel`, `DinkNet`, and `UnetPlusPlus`.
- **Training pipeline** via `MT_train.py` with progress logs and loss plotting.
- **Prediction pipeline** via `MT_test.py`, including channel/mode compatibility checks.
- **Result export** to `*.dat` (pyGIMLI-compatible), `*.grd` (Surfer Grid), and `*.txt`.
- **Quick environment validation** using `quick_test.py`.

---

## 2. Project Structure (Commonly Used)

```text
dl/
├── MTDLPy_GUI.py          # Main GUI application
├── MT_train.py            # Training entry script
├── MT_test.py             # Prediction entry script
├── quick_test.py          # Environment quick check
├── test_prediction.py     # Simple prediction test script
├── ml_trainer.py          # GUI training thread and log parsing
├── ParamConfig.py         # Core configuration (data/training/path)
├── PathConfig.py          # Path and model naming settings
├── LibConfig.py           # Shared imports/config loading
├── requirements.txt       # Python dependencies
├── models/                # Model outputs (auto-created)
├── results/               # Prediction outputs (auto-created)
└── func/
    ├── DataLoad_Train.py
    ├── DataLoad_Test.py
    ├── dinknet.py
    ├── UnetModel.py
    └── unetplusplus.py
```

---

## 3. Requirements

- Python >= 3.7
- Windows is recommended (current GUI/path defaults are Windows-oriented)
- Virtual environment is recommended

---

## 4. Installation

```bash
pip install -r requirements.txt
```

Main dependencies include:

- PyQt5
- numpy / scipy / pandas / scikit-learn
- matplotlib
- opencv-python / scikit-image
- torch / torchvision
- ipython

---

## 5. Quick Check

Run in project root:

```bash
python quick_test.py
```

If all checks pass, you can start GUI or CLI workflow.

---

## 6. GUI Workflow (Recommended)

Start GUI:

```bash
python MTDLPy_GUI.py
```

Main tabs:

1. **Data Import**
   - Select mode: TE / TM / Both
   - Import input data folders (apparent resistivity + phase) and label folder
   - File counts are validated by mode
2. **Param Config**
   - Choose model: `UnetModel` / `DinkNet` / `UnetPlusPlus`
   - Configure `BatchSize`, `LearningRate`, `TrainSize`, `ValSize`, `Epochs`, `Optimizer`
3. **Model Training**
   - Start / pause / resume / stop training
   - Monitor logs, progress bar, and loss curves
4. **Model Prediction**
   - Load test data and trained model (`.pkl/.pt/.pth`)
   - Run prediction through `MT_test.py`
   - Export inversion result and profile figure

---

## 7. Command-Line Usage

### 7.1 Training

```bash
python MT_train.py
```

Training parameters are mainly read from `ParamConfig.py` (`Epochs`, `BatchSize`, `LearnRate`, `MT_Mode`, etc.).

### 7.2 Prediction

```bash
python MT_test.py <test_data_file> <model_file> <MT_mode>
```

Example:

```bash
python MT_test.py H:/sdzl/2.txt models/DinkNet_best.pkl TM
```

You can also use the simplified script:

```bash
python test_prediction.py [model_path] [file_number] [mt_mode]
```

---

## 8. Data and Naming Conventions (Current Implementation)

Main training/prediction flow uses `*.txt` files by default:

- Input files are usually numbered: `1.txt`, `2.txt`, `3.txt`, ...
- Label files are usually named: `zz1.txt`, `zz2.txt`, `zz3.txt`, ...
- In TE/TM/Both mode, related folder file counts must match

Preprocessing behavior:

- Apparent resistivity uses `log10` transform
- Phase keeps raw scale (no 0~1 normalization)
- Label resistivity model uses `log10` transform
- Resampling/alignment uses `ParamConfig.py` settings (`DataDim`, `ModelDim`, `RawGridShape`, etc.)

---

## 9. Key Configuration (`ParamConfig.py`)

Common fields:

- `MT_Mode`: `TE` / `TM` / `Both`
- `Inchannels`: 2 for TE/TM, 4 for Both
- `DataDim`, `ModelDim`
- `RawGridShape`, `RawGridTransposeBeforeResize`
- `PredictionOutputSpatialFix`
- `TrainSize`, `ValSize`
- `Epochs`, `BatchSize`, `LearnRate`
- `ModelName`
- `TE_Resistivity_Dir`, `TE_Phase_Dir`, `TM_Resistivity_Dir`, `TM_Phase_Dir`, `Resistivity_Model_Dir`

> Note: Some default paths in the repo are absolute local paths (e.g., `H:/...`, `M:/...`).  
> Update them to valid paths on your own machine before running.

---

## 10. Outputs

- **Model files** (default: `models/`)
  - best validation model: `...best_val_epochX.pkl`
  - final model: `...final.pkl`
  - convenience copy (e.g., `DinkNet_best.pkl`)
- **Prediction files** (default: `results/`)
  - result text with timestamp
  - JSON config snapshot

---

## 11. FAQ

- **Q: Training says data directory not found.**
  - Check data paths in `ParamConfig.py` and ensure expected `*.txt` files exist.

- **Q: Mode mismatch during prediction.**
  - 2-channel models support TE/TM only; 4-channel models are for Both mode.
  - Use model filename tags such as `ModeTE`, `ModeTM`, `ModeBoth`.

- **Q: Output orientation looks rotated/transposed.**
  - Try changing `PredictionOutputSpatialFix` (`none`, `transpose`, `rot90_cw`, `rot90_ccw`) and rerun prediction.

---

## 12. License

This project is licensed under the [MIT License](LICENSE).
