"""
tet
Author: ycx
Creation Time: 2024
"""
################################################
########        IMPORT LIBARIES         ########
################################################

from ParamConfig import *
try:
    from ParamConfig import (TE_Phase_Dir, TM_Phase_Dir, TE_Resistivity_Dir, TM_Resistivity_Dir)
    print(f"Successfully imported directories from ParamConfig")
    print(f"TE_Phase_Dir: {TE_Phase_Dir}")
    print(f"TM_Phase_Dir: {TM_Phase_Dir}")
    print(f"TE_Resistivity_Dir: {TE_Resistivity_Dir}")
    print(f"TM_Resistivity_Dir: {TM_Resistivity_Dir}")
except ImportError:
    print("Warning: ParamConfig module not found specific directories, using globals if available")
    TE_Phase_Dir = None
    TM_Phase_Dir = None
    TE_Resistivity_Dir = None
    TM_Resistivity_Dir = None
from PathConfig import *
from LibConfig import *
import torch
import torch.nn as nn
import os
import numpy as np
import sys
import datetime
import json
import time

# 已移除Swin Transformer相关代码


################################################
########         LOAD    NETWORK        ########
################################################

def load_model(model_type='DinkNet', model_path=None):
    """
    Load network; infer model type, MT mode, and input channel count from filename when possible.

    Returns (net, model_file).
    """
    # Here indicating the GPU you want to use. if you don't have GPU, just leave it.
    cuda_available = torch.cuda.is_available()
    device         = torch.device("cuda" if cuda_available else "cpu")
    
    # 如果提供了模型路径，则优先使用
    if model_path and os.path.exists(model_path):
        model_file = model_path
        print(f"Use the specified model path: {model_file}")
        
        # 从模型文件名中提取MT模式信息
        model_filename = os.path.basename(model_file)
        mt_mode_from_file = None
        num_channels = 2  # 默认2通道
        
        if 'ModeTE' in model_filename:
            mt_mode_from_file = 'TE'
            num_channels = 2
        elif 'ModeTM' in model_filename:
            mt_mode_from_file = 'TM'
            num_channels = 2
        elif 'ModeBoth' in model_filename:
            mt_mode_from_file = 'Both'
            num_channels = 4
        
        print(f"Detected MT mode from filename: {mt_mode_from_file}, using {num_channels} input channels")
        
        # 根据模型文件名推断模型类型
        if 'DinkNet' in model_file or 'dinknet' in model_file.lower():
            # 如果文件名中有明确的通道数标识，优先使用
            if '2ch' in model_file or 'twoch' in model_file or 'doublech' in model_file:
                num_channels = 2
                print(f"Overriding channel count to 2 based on filename identifier")
            elif '4ch' in model_file or 'fourch' in model_file:
                num_channels = 4
                print(f"Overriding channel count to 4 based on filename identifier")
            
            net = DinkNet50(num_classes=Nclasses, num_channels=num_channels)
            print(f"Use DinkNet50 model, input channels: {num_channels}")
            
        elif 'UnetPlusPlus' in model_file or 'unetplusplus' in model_file.lower() or 'UNet++' in model_file:
            net = UNetPlusPlus(num_classes=Nclasses, num_channels=num_channels)
            print(f"Use UNetPlusPlus model, input channels: {num_channels}")
            
        elif 'Unet' in model_file or 'UNet' in model_file or 'unet' in model_file.lower():
            # Unet模型使用从文件名推断的通道数
            net = UnetModel(n_classes=Nclasses, in_channels=num_channels)
            print(f"Use UnetModel, input channels: {num_channels}")
        else:
            # 默认使用DinkNet，使用推断的通道数
            net = DinkNet50(num_classes=Nclasses, num_channels=num_channels)
            print(f"Use default DinkNet50 model, input channels: {num_channels}")
    else:
        # 根据不同的模型类型选择相应的模型文件
        if model_type == 'DinkNet' or model_type is None:
            # 使用models目录中存在的DinkNet模型文件
            # 检查目录中是否存在DinkNet模型文件
            model_files = [f for f in os.listdir(models_dir) if 'DinkNet' in f and f.endswith('.pkl')]
            if model_files:
                model_file = max([os.path.join(models_dir, f) for f in model_files], key=os.path.getmtime)
            else:
                raise FileNotFoundError(f"DinkNet model file not found in the {models_dir} directory")
            
            # 从模型文件名中提取MT模式信息
            model_filename = os.path.basename(model_file)
            mt_mode_from_file = None
            if 'ModeTE' in model_filename:
                mt_mode_from_file = 'TE'
                num_channels = 2
            elif 'ModeTM' in model_filename:
                mt_mode_from_file = 'TM'
                num_channels = 2
            elif 'ModeBoth' in model_filename:
                mt_mode_from_file = 'Both'
                num_channels = 4
            else:
                # 如果没有MT模式信息，尝试从其他标识推断
                if '2ch' in model_file or 'twoch' in model_file or 'doublech' in model_file:
                    num_channels = 2
                elif '4ch' in model_file or 'fourch' in model_file:
                    num_channels = 4
                else:
                    num_channels = 2  # 默认2通道
            
            print(f"Detected MT mode from filename: {mt_mode_from_file}, using {num_channels} input channels")
            net = DinkNet50(num_classes=Nclasses, num_channels=num_channels)
            print(f"Use DinkNet50 model, input channels: {num_channels}")
            
        elif model_type == 'UNet' or model_type == 'UnetModel':
            # 使用U-Net模型
            model_files = [f for f in os.listdir(models_dir) if ('Unet' in f or 'UNet' in f) and f.endswith('.pkl')]
            if model_files:
                model_file = max([os.path.join(models_dir, f) for f in model_files], key=os.path.getmtime)
            else:
                # 如果没有找到U-Net模型，尝试使用一个默认的文件名格式
                model_file = os.path.join(models_dir, "UnetModel_TrainSize0.01_Epoch30_BatchSize8_LR00001.pkl")
            
            # 从模型文件名中提取MT模式信息
            model_filename = os.path.basename(model_file)
            if 'ModeBoth' in model_filename:
                num_channels = 4
            else:  # TE或TM模式
                num_channels = 2
            
            print(f"Detected MT mode from filename, using {num_channels} input channels for UnetModel")
            net = UnetModel(n_classes=Nclasses, in_channels=num_channels)
            print(f"Use UnetModel, input channels: {num_channels}")
            
        elif model_type == 'UnetPlusPlus':
            model_files = [f for f in os.listdir(models_dir) if ('UnetPlusPlus' in f or 'unetplusplus' in f.lower()) and f.endswith('.pkl')]
            if model_files:
                model_file = max([os.path.join(models_dir, f) for f in model_files], key=os.path.getmtime)
            else:
                model_file = os.path.join(models_dir, "UnetPlusPlus_TrainSize0.01_Epoch30_BatchSize8_LR00001.pkl")
            model_filename = os.path.basename(model_file)
            num_channels = 4 if 'ModeBoth' in model_filename else 2
            print(f"Detected MT mode from filename, using {num_channels} input channels for UnetPlusPlus")
            net = UNetPlusPlus(num_classes=Nclasses, num_channels=num_channels)
            print(f"Use UNetPlusPlus model, input channels: {num_channels}")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    print(f"Load model file: {model_file}")
    
    # 加载模型权重
    try:
        net.load_state_dict(torch.load(model_file, map_location=device))
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_file}")
    except RuntimeError as e:
        error_msg = str(e)
        print(f"Model load error: {error_msg}")
        
        # 如果出现size mismatch错误，可能是输入通道数不匹配
        if 'size mismatch' in error_msg and ('firstconv.weight' in error_msg or 'weight' in error_msg):
            print(f"Model loading failed due to channel mismatch, trying different channel configurations: {error_msg}")
            
            # 尝试不同的通道数配置
            if isinstance(net, torch.nn.Module):
                model_class_name = net.__class__.__name__
                
                # 从文件名再次提取MT模式
                model_filename = os.path.basename(model_file)
                if 'ModeBoth' in model_filename:
                    try_channels = [4, 2]  # 先尝试4通道，再尝试2通道
                else:
                    try_channels = [2, 4, 1]  # 先尝试2通道，再尝试4通道，最后尝试1通道
                
                for try_ch in try_channels:
                    try:
                        print(f"Trying to create {model_class_name} model with {try_ch} input channels")
                        if model_class_name == 'DinkNet50':
                            net = DinkNet50(num_classes=Nclasses, num_channels=try_ch)
                        elif model_class_name == 'UnetModel':
                            net = UnetModel(n_classes=Nclasses, in_channels=try_ch)
                        elif model_class_name == 'UNetPlusPlus':
                            net = UNetPlusPlus(num_classes=Nclasses, num_channels=try_ch)
                        else:
                            break
                        
                        # 再次尝试加载模型权重
                        net.load_state_dict(torch.load(model_file, map_location=device))
                        print(f"Successfully loaded model with {try_ch} input channels")
                        break
                    except RuntimeError as e2:
                        if try_ch == try_channels[-1]:  # 最后一次尝试
                            raise RuntimeError(f"Failed to load weights after trying all channel configs: {error_msg}")
                        continue
        else:
            # 其他类型的错误，直接抛出
            raise RuntimeError(f"Failed to load model weights: {error_msg}")
    
    # 将模型移至GPU（如果可用）
    if torch.cuda.is_available():
        net.cuda()
    
    return net, model_file


def infer_net_input_channels(net):
    """Infer first-layer input channels from loaded weights (avoids assuming 2 ch when filename lacks ModeBoth)."""
    name = net.__class__.__name__
    if name == 'DinkNet50':
        return int(net.firstconv.weight.shape[1])
    if name == 'UnetModel':
        return int(net.in_channels)
    if name == 'UNetPlusPlus':
        return int(net.num_channels)
    if hasattr(net, 'firstconv') and hasattr(net.firstconv, 'weight'):
        return int(net.firstconv.weight.shape[1])
    raise RuntimeError(f'Cannot infer input channels from network type {name}')


def normalize_gui_mt_mode(argv_val):
    """Normalize GUI argv TE/TM/BOTH to strings used by load_test_data."""
    if argv_val is None or str(argv_val).strip() == '':
        return None
    s = str(argv_val).strip()
    u = s.upper()
    if u == 'BOTH':
        return 'Both'
    if u == 'TE':
        return 'TE'
    if u == 'TM':
        return 'TM'
    return s


################################################
########    LOADING TESTING DATA       ########
################################################

# 导入训练数据加载模块中的降采样函数与空间网格预处理（与训练一致）
from skimage.measure import block_reduce
from func.DataLoad_Train import ensure_grid_shape, _get_raw_grid_shape

def decimate(a, axis):
    """Downsample block mean; same as training."""
    return np.mean(a, axis=axis)

def validate_test_data(resistivity_data, phase_data=None, file_id='test', mode='TE'):
    """
    Validate apparent resistivity and phase (aligned with DataLoad_Train.validate_data).
    Returns valid, invalid_info, resistivity_data, phase_data.
    """
    invalid_info = []
    valid = True
    
    # 检查视电阻率数据是否有负数或异常值（与训练时一致）
    if np.any(np.logical_or(resistivity_data <= 0, resistivity_data >= 100000)):
        valid = False
        invalid_info.append(f"{mode} apparent resistivity has non-positive or out-of-range values")
        # 修复：将无效值限制在有效范围内
        resistivity_data = np.maximum(resistivity_data, 1e-6)
        resistivity_data = np.minimum(resistivity_data, 99999)
        print(f"Clamped invalid {mode} apparent resistivity values")
    
    # 检查相位数据是否在0-90之间（如果提供，与训练时一致）
    if phase_data is not None:
        if np.any(phase_data < 0) or np.any(phase_data > 90):
            valid = False
            invalid_info.append(f"{mode} phase values outside [0, 90]")
            # 修复：将无效值限制在有效范围内
            phase_data = np.maximum(phase_data, 0)
            phase_data = np.minimum(phase_data, 90)
            print(f"Clamped invalid {mode} phase values")
    
    return valid, invalid_info, resistivity_data, phase_data

def load_test_data(test_data_file=None, mt_mode='TM'):
    """
    Load test tensors with the same pipeline as training (see DataLoad_Train).

    Returns test_set, data_dsp_dim, label_dsp_dim, file_info.
    """
    print('***************** Loading Testing DataSet *****************')
    
    # 确保Inchannels变量已定义
    global Inchannels, data_dsp_blk, label_dsp_blk, DataDim, ModelDim
    if 'Inchannels' not in globals() or Inchannels is None:
        # 根据MT模式自动设置输入通道数
        if mt_mode == 'Both':
            Inchannels = 4
        else:
            Inchannels = 2
        print(f"Warning: Inchannels not defined, using default value: {Inchannels} based on MT mode")
    
    # 获取降采样参数（与训练时一致）
    if 'data_dsp_blk' not in globals() or data_dsp_blk is None:
        data_dsp_blk = (1, 1)  # 默认不降采样
    if 'label_dsp_blk' not in globals() or label_dsp_blk is None:
        label_dsp_blk = (1, 1)  # 默认不降采样
    
    # 使用传入的mt_mode参数
    if mt_mode is None:
        mt_mode = MT_Mode if 'MT_Mode' in globals() else 'TM'
        print(f"Using default MT mode: {mt_mode}")
    
    # 根据MT模式确定输入通道数
    if mt_mode == 'Both':
        expected_channels = 4
    else:
        expected_channels = 2
    
    if expected_channels != Inchannels:
        print(f"Warning: MT mode {mt_mode} requires {expected_channels} channels, but Inchannels={Inchannels}. Adjusting...")
        Inchannels = expected_channels

    # 与训练一致：目标网格来自 DataDim；原始行列来自 ParamConfig.RawGridShape（无需预测时再输入）
    _raw_gs = _get_raw_grid_shape()
    if 'DataDim' not in globals() or DataDim is None:
        DataDim = [32, 32]
    _th, _tw = int(DataDim[0]), int(DataDim[1])
    
    if test_data_file is not None and test_data_file and os.path.exists(test_data_file):
        # 使用指定的测试数据文件
        print(f"Use specified test data file: {test_data_file}")
        
        # 获取文件基础名称（不含扩展名）
        file_base = os.path.splitext(os.path.basename(test_data_file))[0]
        
        try:
            # ========== 数据加载和预处理（与训练时完全一致） ==========
            
            if mt_mode == 'TE' or mt_mode == 'TM':
                # TE或TM模式：2通道（视电阻率 + 相位）
                print(f"Loading {mt_mode} mode data (2 channels)")
                
                # 步骤1：读取原始数据（与训练时完全一致）
                print(f"Step 1: Loading {mt_mode} resistivity data from: {test_data_file}")
                raw_resistivity = np.loadtxt(test_data_file, encoding='utf-8')
                
                # 读取相位数据
                phase_file = None
                if mt_mode == 'TE' and TE_Phase_Dir and os.path.exists(TE_Phase_Dir):
                    phase_file = os.path.join(TE_Phase_Dir, f"{file_base}.txt")
                elif mt_mode == 'TM' and TM_Phase_Dir and os.path.exists(TM_Phase_Dir):
                    phase_file = os.path.join(TM_Phase_Dir, f"{file_base}.txt")
                
                raw_phase = None
                if phase_file and os.path.exists(phase_file):
                    print(f"Step 1: Loading {mt_mode} phase data from: {phase_file}")
                    raw_phase = np.loadtxt(phase_file, encoding='utf-8')
                else:
                    print(f"Warning: Phase file not found: {phase_file}, using zeros for phase channel")
                    raw_phase = np.zeros_like(np.asarray(raw_resistivity), dtype=np.float64)
                
                # 步骤2：数据预处理（与训练时完全一致）
                # 2.1 原始网格先插值到 DataDim，再转置（与 DataLoad_Train.ensure_grid_shape 一致）
                print(f"Step 2.1: Resize to DataDim=({_th}, {_tw}) then transpose (RawGridShape={_raw_gs})")
                train_data1 = ensure_grid_shape(raw_resistivity, (_th, _tw), _raw_gs).T
                train_data2 = ensure_grid_shape(raw_phase, (_th, _tw), _raw_gs).T
                
                # 2.2 数据验证（与训练时一致）
                print(f"Step 2.2: Validate data")
                valid, invalid_info, train_data1, train_data2 = validate_test_data(train_data1, train_data2, file_base, mt_mode)
                if not valid:
                    print(f"Warning: validation issues:")
                    for info in invalid_info:
                        print(f"  - {info}")
                
                # 2.3 对视电阻率进行对数转换（与DataLoad_Train.py第114行一致）
                print(f"Step 2.3: Apply log10 to resistivity data")
                train_data1 = np.log10(train_data1)
                
                # 2.4 相位数据不进行归一化（与DataLoad_Train.py第116行一致，不除以90.0）
                print(f"Step 2.4: Phase data kept as is (no normalization)")
                
                # 2.5 组织数据格式（与DataLoad_Train.py第118-119行一致）
                print(f"Step 2.5: Organize data format")
                data1_set = np.array([train_data1, train_data2])
                data1_set = np.transpose(data1_set, (1, 2, 0))  # 转换为(height, width, channels)
                
                # 2.6 对每个通道进行降采样（与DataLoad_Train.py第121-131行完全一致）
                print(f"Step 2.6: Downsample each channel using block_reduce")
                data_set = None
                for k in range(0, Inchannels):
                    data11_set = np.float32(data1_set[:, :, k])
                    data11_set = np.float32(data11_set)  # 确保float32类型
                    # Data downsampling（与训练时完全一致）
                    data11_set = block_reduce(data11_set, block_size=data_dsp_blk, func=decimate)
                    data_dsp_dim = data11_set.shape  # 记录降采样后的维度
                    # Flatten为1D（与训练时完全一致）
                    data11_set = data11_set.reshape(1, data_dsp_dim[0] * data_dsp_dim[1])
                    
                    if k == 0:
                        data_set = data11_set
                    else:
                        data_set = np.append(data_set, data11_set, axis=0)
                
                print(f"Data downsampled from ({_th}, {_tw}) to {data_dsp_dim} with block size {data_dsp_blk}")
                print(f"Data set shape after processing: {data_set.shape} (channels, flattened)")
                
                # 2.7 组织为最终格式（与训练时完全一致）
                # 训练时最终shape: (samples, channels, data_dsp_dim[0] * data_dsp_dim[1])
                print(f"Step 2.7: Reshape to final format matching training")
                test_set = data_set.reshape(1, Inchannels, data_dsp_dim[0] * data_dsp_dim[1])
                print(f"Final test_set shape: {test_set.shape} (samples, channels, flattened_data)")
                
            elif mt_mode == 'Both':
                # Both模式：4通道（TE电阻率 + TE相位 + TM电阻率 + TM相位）
                print(f"Loading Both mode data (4 channels)")
                
                # 步骤1：读取所有数据
                te_resistivity_file = test_data_file
                te_phase_file = os.path.join(TE_Phase_Dir, f"{file_base}.txt") if TE_Phase_Dir and os.path.exists(TE_Phase_Dir) else None
                
                # 尝试找到TM数据文件
                tm_resistivity_file = None
                tm_phase_file = None
                if TM_Resistivity_Dir and os.path.exists(TM_Resistivity_Dir):
                    tm_resistivity_file = os.path.join(TM_Resistivity_Dir, f"{file_base}.txt")
                    if not os.path.exists(tm_resistivity_file):
                        tm_files = [f for f in os.listdir(TM_Resistivity_Dir) if f.endswith('.txt')]
                        if tm_files:
                            tm_resistivity_file = os.path.join(TM_Resistivity_Dir, tm_files[0])
                
                if TM_Phase_Dir and os.path.exists(TM_Phase_Dir):
                    tm_phase_file = os.path.join(TM_Phase_Dir, f"{file_base}.txt")
                    if not os.path.exists(tm_phase_file):
                        tm_files = [f for f in os.listdir(TM_Phase_Dir) if f.endswith('.txt')]
                        if tm_files:
                            tm_phase_file = os.path.join(TM_Phase_Dir, tm_files[0])
                
                # 读取所有数据
                raw_te_resistivity = np.loadtxt(te_resistivity_file, encoding='utf-8')
                _te0 = np.asarray(raw_te_resistivity, dtype=np.float64)
                raw_te_phase = np.loadtxt(te_phase_file, encoding='utf-8') if te_phase_file and os.path.exists(te_phase_file) else np.zeros_like(_te0)
                raw_tm_resistivity = np.loadtxt(tm_resistivity_file, encoding='utf-8') if tm_resistivity_file and os.path.exists(tm_resistivity_file) else np.zeros_like(_te0)
                _tm0 = np.asarray(raw_tm_resistivity, dtype=np.float64)
                raw_tm_phase = np.loadtxt(tm_phase_file, encoding='utf-8') if tm_phase_file and os.path.exists(tm_phase_file) else np.zeros_like(_tm0)
                
                # 步骤2：数据预处理（与训练流程完全一致）
                # 2.1 插值到 DataDim 后转置
                train_data1 = ensure_grid_shape(raw_te_resistivity, (_th, _tw), _raw_gs).T
                train_data2 = ensure_grid_shape(raw_te_phase, (_th, _tw), _raw_gs).T
                train_data3 = ensure_grid_shape(raw_tm_resistivity, (_th, _tw), _raw_gs).T
                train_data4 = ensure_grid_shape(raw_tm_phase, (_th, _tw), _raw_gs).T
                
                # 2.2 数据验证
                valid_te, invalid_info_te, train_data1, train_data2 = validate_test_data(train_data1, train_data2, file_base, 'TE')
                valid_tm, invalid_info_tm, train_data3, train_data4 = validate_test_data(train_data3, train_data4, file_base, 'TM')
                
                if not valid_te or not valid_tm:
                    print(f"Warning: validation issues:")
                    if not valid_te:
                        for info in invalid_info_te:
                            print(f"  - {info}")
                    if not valid_tm:
                        for info in invalid_info_tm:
                            print(f"  - {info}")
                
                # 2.3 对数转换（与训练时完全一致）
                train_data1 = np.log10(train_data1)
                train_data3 = np.log10(train_data3)
                # 相位数据不进行归一化
                
                data1_set = np.array([train_data1, train_data2, train_data3, train_data4])
                data1_set = np.transpose(data1_set, (1, 2, 0))
                
                # 2.4 对每个通道进行降采样
                data_set = None
                for k in range(0, Inchannels):
                    data11_set = np.float32(data1_set[:, :, k])
                    data11_set = np.float32(data11_set)
                    # Data downsampling
                    data11_set = block_reduce(data11_set, block_size=data_dsp_blk, func=decimate)
                    data_dsp_dim = data11_set.shape
                    data11_set = data11_set.reshape(1, data_dsp_dim[0] * data_dsp_dim[1])
                    if k == 0:
                        data_set = data11_set
                    else:
                        data_set = np.append(data_set, data11_set, axis=0)
                
                print(f"Data downsampled from ({_th}, {_tw}) to {data_dsp_dim} with block size {data_dsp_blk}")
                
                # 2.5 组织为最终格式
                test_set = data_set.reshape(1, Inchannels, data_dsp_dim[0] * data_dsp_dim[1])
                print(f"Final test_set shape: {test_set.shape}")
            
            # 计算label_dsp_dim（与训练时完全一致）
            # 训练时：label_dsp_dim是通过对32x32的模型数据进行block_reduce后得到的实际shape
            if 'ModelDim' in globals() and ModelDim:
                # 模拟训练时的计算方式：对ModelDim进行block_reduce
                label_height = ModelDim[0] // label_dsp_blk[0] if label_dsp_blk[0] > 0 else ModelDim[0]
                label_width = ModelDim[1] // label_dsp_blk[1] if label_dsp_blk[1] > 0 else ModelDim[1]
                label_dsp_dim = (label_height, label_width)
                print(f"Calculated label_dsp_dim from ModelDim={ModelDim} and label_dsp_blk={label_dsp_blk}: {label_dsp_dim}")
            else:
                # 默认值：假设ModelDim=[32,32], label_dsp_blk=(1,1)
                label_dsp_dim = (32, 32)
                print(f"Using default label_dsp_dim: {label_dsp_dim}")
            
            # 验证data_dsp_dim和label_dsp_dim的关系
            # 训练时：data_dsp_dim[0]*data_dsp_dim[1] 应该等于 label_dsp_dim[0]*label_dsp_dim[1]
            data_total = data_dsp_dim[0] * data_dsp_dim[1]
            label_total = label_dsp_dim[0] * label_dsp_dim[1]
            if data_total != label_total:
                print(f"Warning: data_dsp_dim total elements ({data_total}) != label_dsp_dim total elements ({label_total})")
                print(f"This may cause issues during prediction. Training requires these to match for view() operation.")
            else:
                print(f"Data and label dimensions match (total elements: {data_total})")
            
            print(f"Test data loaded successfully:")
            print(f"  Shape: {test_set.shape}")
            print(f"  Data DSP Dim: {data_dsp_dim}")
            print(f"  Label DSP Dim: {label_dsp_dim}")
            
            file_info = {
                "filename": os.path.basename(test_data_file),
                "path": test_data_file,
                "mt_mode": mt_mode
            }
            return test_set, data_dsp_dim, label_dsp_dim, file_info
                
        except Exception as e:
            print(f"Failed to load test data file: {test_data_file}", e)
            error_msg = f"Failed to load test data file: {test_data_file}\n"
            error_msg += f"Detail: {str(e)}"
            raise RuntimeError(error_msg)
    else:
        # 如果没有指定测试数据文件，尝试从数据目录中查找
        error_msg = "Error: a valid test data file path is required\n"
        error_msg += "Pass the test data file when running this script"
        print(error_msg)
        raise RuntimeError(error_msg)


################################################
########            PREDICTION          ########
################################################
def predict(net, test_set, data_dsp_dim, label_dsp_dim):
    """
    Run forward pass; tensor layout matches training (flattened input then view to label_dsp_dim).
    """
    
    print() 
    print('*******************************************') 
    print('*******************************************') 
    print('            START PREDICTION               ') 
    print('*******************************************') 
    print('*******************************************') 
    print()
    
    # 设置为评估模式
    net.eval()
    
    # 记录开始时间
    since = time.time()
    
    # 准备数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 获取模型类型名称
    model_class_name = net.__class__.__name__
    
    # 添加详细的调试信息
    print(f"\n{'='*60}")
    print(f"Input statistics:")
    print(f"{'='*60}")
    print(f"  raw test_set shape: {test_set.shape}")
    print(f"  expected: (samples, channels, flattened_data)")
    print(f"  value range: min={np.min(test_set):.6f}, max={np.max(test_set):.6f}")
    print(f"  mean: {np.mean(test_set):.6f}, std: {np.std(test_set):.6f}")
    
    # 分别检查每个通道的数据范围
    for ch in range(test_set.shape[1]):
        ch_data = test_set[0, ch]
        print(f"  channel {ch+1} (flattened, length={len(ch_data)}):")
        print(f"    min={ch_data.min():.6f}, max={ch_data.max():.6f}")
        print(f"    mean={ch_data.mean():.6f}, std={ch_data.std():.6f}")
        # 如果是第一个通道（通常是电阻率，已经log10转换），显示对应的原始电阻率范围
        if ch == 0:
            if ch_data.min() > -20 and ch_data.max() < 20:
                orig_min = 10 ** ch_data.min()
                orig_max = 10 ** ch_data.max()
                print(f"    approx. linear resistivity range: {orig_min:.2e} - {orig_max:.2e} Ohm·m")
    
    print(f"  model: {model_class_name}")
    print(f"  device: {device}")
    print(f"  data_dsp_dim: {data_dsp_dim}")
    print(f"  label_dsp_dim: {label_dsp_dim}")
    print(f"{'='*60}\n")
    
    # ========== 关键：与训练时完全一致的reshape ==========
    # 训练时数据流程：
    # 1. 数据加载后shape: (samples, channels, data_dsp_dim[0] * data_dsp_dim[1]) - flatten的1D
    # 2. 训练时reshape: images.view(batch, channels, label_dsp_dim[0], label_dsp_dim[1])
    # 注意：view要求总元素数相同，所以 data_dsp_dim[0]*data_dsp_dim[1] 必须等于 label_dsp_dim[0]*label_dsp_dim[1]
    
    batch_size = test_set.shape[0]
    in_channels = test_set.shape[1]
    flattened_size = test_set.shape[2]  # data_dsp_dim[0] * data_dsp_dim[1]
    target_total = label_dsp_dim[0] * label_dsp_dim[1]
    
    print(f"Input data: batch={batch_size}, channels={in_channels}, flattened_size={flattened_size}")
    print(f"Target label dimensions: {label_dsp_dim[0]}x{label_dsp_dim[1]} (total: {target_total})")
    
    # 转换为tensor（与训练时一致）
    test_tensor = torch.from_numpy(test_set).float().to(device)
    
    # 如果总元素数相同，可以直接view（与训练时完全一致）
    if flattened_size == target_total:
        # 与训练时完全一致：view到label_dsp_dim
        print(f"Reshaping using view (total elements match: {flattened_size} == {target_total})")
        test_tensor = test_tensor.view(batch_size, in_channels, label_dsp_dim[0], label_dsp_dim[1])
        print(f"Reshaped test_tensor shape: {test_tensor.shape}")
    else:
        # 如果总元素数不同，需要使用插值调整大小
        print(f"Warning: Total elements differ ({flattened_size} vs {target_total}), using interpolation")
        # 先reshape到data_dsp_dim
        test_tensor_2d = test_tensor.view(batch_size, in_channels, data_dsp_dim[0], data_dsp_dim[1])
        # 转换为numpy进行插值
        test_array = test_tensor_2d.cpu().numpy()
        from scipy.ndimage import zoom
        reshaped_data = np.zeros((batch_size, in_channels, label_dsp_dim[0], label_dsp_dim[1]), dtype=np.float32)
        for b in range(batch_size):
            for c in range(in_channels):
                # 使用zoom进行插值调整大小（使用线性插值）
                scale_factors = (label_dsp_dim[0] / data_dsp_dim[0], label_dsp_dim[1] / data_dsp_dim[1])
                reshaped_data[b, c] = zoom(test_array[b, c], scale_factors, order=1)
        test_tensor = torch.from_numpy(reshaped_data).float().to(device)
        print(f"Reshaped test_tensor using interpolation: {test_tensor.shape}")
    
    print(f"Final test_tensor shape for model input: {test_tensor.shape}")
    print(f"Expected shape: (batch={batch_size}, channels={in_channels}, height={label_dsp_dim[0]}, width={label_dsp_dim[1]})")
    
    # 进行预测
    with torch.no_grad():  
        # 检查模型类型，根据不同模型类型传入正确的参数
        if model_class_name == 'UnetModel':
            # UnetModel需要两个参数：图像和label_dsp_dim
            outputs = net(test_tensor, label_dsp_dim)
        else:
            # DinkNet、UNetPlusPlus等模型只需要一个参数：图像
            outputs = net(test_tensor)
        
        print(f"Model output shape: {outputs.shape}")
        print(f"Model output dtype: {outputs.dtype}")
        print(f"Model output value range: min={outputs.min().item():.6f}, max={outputs.max().item():.6f}")
        
        # 处理输出 - 与训练时的标签格式保持一致
        # 训练时标签格式：(batch, Nclasses, height, width)
        # 如果输出是4D张量(batch, channels, height, width)
        if len(outputs.shape) == 4:
            print(f"Output is 4D tensor: {outputs.shape}")
            # 如果Nclasses=1，squeeze掉channels维度，得到(batch, height, width)
            if outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)  # (batch, height, width)
                print(f"After squeezing channel dimension: {outputs.shape}")
            else:
                # 如果有多个通道，保持原样
                print(f"Keeping multiple channels: {outputs.shape}")
                pass
        elif len(outputs.shape) == 3:
            # 已经是(batch, height, width)格式
            print(f"Output is already 3D tensor: {outputs.shape}")
            pass
        elif len(outputs.shape) == 2:
            # 如果是2D，可能是(height, width)
            print(f"Warning: Output is 2D tensor: {outputs.shape}, expected 3D or 4D")
            # 尝试reshape到期望的形状
            if outputs.shape[0] == label_dsp_dim[0] and outputs.shape[1] == label_dsp_dim[1]:
                # 单样本，添加batch维度
                outputs = outputs.unsqueeze(0)
                print(f"Reshaped 2D to 3D by adding batch dimension: {outputs.shape}")
            else:
                # 尝试reshape
                total_elements = outputs.numel()
                expected_elements = label_dsp_dim[0] * label_dsp_dim[1]
                if total_elements == expected_elements:
                    outputs = outputs.view(1, label_dsp_dim[0], label_dsp_dim[1])
                    print(f"Reshaped 2D to 3D using view: {outputs.shape}")
                else:
                    print(f"Error: Cannot reshape 2D output. Total elements: {total_elements}, Expected: {expected_elements}")
        else:
            # 其他情况，尝试reshape
            print(f"Warning: Unexpected output shape {outputs.shape}, attempting to reshape")
            total_elements = outputs.numel()
            expected_elements = label_dsp_dim[0] * label_dsp_dim[1]
            if total_elements == expected_elements:
                outputs = outputs.view(1, label_dsp_dim[0], label_dsp_dim[1])
                print(f"Reshaped to expected shape: {outputs.shape}")
            else:
                print(f"Error: Cannot reshape. Total elements: {total_elements}, Expected: {expected_elements}")
        
        # 移动到CPU并转换为numpy数组
        predictions = outputs.cpu().numpy()
        
        # 确保predictions是2D或3D数组
        if len(predictions.shape) == 2:
            # 单样本，添加batch维度
            predictions = predictions[np.newaxis, :, :]
        elif len(predictions.shape) == 1:
            # 如果是1D，reshape为2D
            predictions = predictions.reshape(label_dsp_dim[0], label_dsp_dim[1])
            predictions = predictions[np.newaxis, :, :]
        
        print(f"Final predictions shape: {predictions.shape}")
        
        # 打印详细的预测结果统计信息
        print("\n" + "="*60)
        print("Prediction stats (model output, log10 resistivity):")
        print("="*60)
        for i in range(predictions.shape[0]):
            pred_2d = predictions[i]
            current_min = pred_2d.min()
            current_max = pred_2d.max()
            current_mean = pred_2d.mean()
            current_std = pred_2d.std()
            print(f"channel {i+1} log10 range:")
            print(f"  min={current_min:.6f}, max={current_max:.6f}")
            print(f"  mean={current_mean:.6f}, std={current_std:.6f}")
            print(f"  shape={pred_2d.shape}")
            
            # 计算对应的电阻率范围（用于参考）
            if current_min > -20 and current_max < 20:
                resistivity_min = 10 ** current_min
                resistivity_max = 10 ** current_max
                print(f"  approx. linear resistivity: {resistivity_min:.2e} - {resistivity_max:.2e} Ohm·m")
                print(f"  (saved file converts log10 to linear resistivity)")
        print("="*60 + "\n")
    
    # 记录耗时
    time_elapsed = time.time() - since
    print(f'Prediction completed in {time_elapsed:.2f} seconds')
    
    return predictions


def _apply_prediction_spatial_fix_2d(arr):
    """
    Apply ParamConfig.PredictionOutputSpatialFix (transpose/rotate) before saving, to match map/orientation conventions.
    """
    if arr is None or not hasattr(arr, 'ndim') or arr.ndim != 2:
        return arr
    try:
        from ParamConfig import PredictionOutputSpatialFix as fix
    except (ImportError, AttributeError):
        fix = 'none'
    fix = (fix or 'none').strip().lower()
    if fix == 'none' or fix == '':
        return arr
    if fix == 'transpose':
        return np.ascontiguousarray(arr.T)
    if fix in ('rot90_cw', 'rot90_clockwise', 'cw'):
        return np.ascontiguousarray(np.rot90(arr, k=-1))
    if fix in ('rot90_ccw', 'ccw'):
        return np.ascontiguousarray(np.rot90(arr, k=1))
    print(f"Warning: unknown PredictionOutputSpatialFix={fix!r}, ignored.")
    return arr


################################################
########          SAVE RESULTS          ########
################################################
def save_prediction_results(predictions, file_info, data_dsp_dim):
    """
    Write prediction grids to results_dir; returns (result_file_path, config_file_path).
    """
    # 创建结果目录（如果不存在）
    os.makedirs(results_dir, exist_ok=True)
    
    # 生成结果文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.splitext(file_info["filename"])[0]
    result_filename = f"pred_{base_filename}_{timestamp}.txt"
    result_file_path = os.path.join(results_dir, result_filename)
    
    # 从file_info中获取MT模式
    mt_mode = file_info.get("mt_mode", MT_Mode if 'MT_Mode' in globals() else "TM")
    
    # 保存预测结果
    # 检查是否为Both模式且有4个通道
    if mt_mode == 'Both' and len(predictions) >= 4:
        # 对于Both模式的4通道数据，分别保存每个通道
        with open(result_file_path, 'w') as f:
            for i in range(min(4, len(predictions))):
                f.write(f"# Channel {i+1}\n")
                if i == 0 or i == 2:  # 电阻率通道
                    limited_predictions = np.clip(predictions[i], -20, 20)
                    channel_data = np.power(10, limited_predictions)
                    channel_data = np.clip(channel_data, 1e-8, 1e8)
                else:  # 相位通道
                    channel_data = predictions[i]
                channel_data = _apply_prediction_spatial_fix_2d(channel_data)
                np.savetxt(f, channel_data, fmt='%.6f')
                if i < min(4, len(predictions)) - 1:
                    f.write('\n')
    else:
        # 对于其他模式，保存第一个通道的数据
        # 注意：predictions[0]已经是log10格式的预测值（与训练时标签格式一致）
        # 限制预测值范围，防止出现inf值
        limited_predictions = np.clip(predictions[0], -20, 20)
        # 应用指数转换，还原原始电阻率范围（因为训练时标签也是log10格式）
        restored_predictions = np.power(10, limited_predictions)
        # 进一步限制结果范围
        restored_predictions = np.clip(restored_predictions, 1e-8, 1e8)
        restored_predictions = _apply_prediction_spatial_fix_2d(restored_predictions)
        print(f"\n{'='*60}")
        print("Save pipeline (log10 -> linear):")
        print(f"{'='*60}")
        print(f"model output (log10):")
        print(f"  min={predictions[0].min():.6f}, max={predictions[0].max():.6f}")
        print(f"after clip (log10):")
        print(f"  min={limited_predictions.min():.6f}, max={limited_predictions.max():.6f}")
        print(f"linear resistivity (Ohm·m):")
        print(f"  min={restored_predictions.min():.6e}, max={restored_predictions.max():.6e}")
        print(f"log10 of linear (check):")
        print(f"  min={np.log10(restored_predictions.min()):.6f}, max={np.log10(restored_predictions.max()):.6f}")
        print(f"{'='*60}\n")
        np.savetxt(result_file_path, restored_predictions, fmt='%.6f')
    
    print(f"Prediction results saved to: {result_file_path}")
    
    # 保存配置信息
    config_filename = f"pred_config_{base_filename}_{timestamp}.json"
    config_file_path = os.path.join(results_dir, config_filename)
    
    actual_data_dim = list(data_dsp_dim) if hasattr(data_dsp_dim, '__iter__') else [16, 16]
    
    config_data = {
        "original_file": file_info["filename"],
        "original_path": file_info["path"],
        "prediction_file": result_filename,
        "prediction_path": result_file_path,
        "timestamp": timestamp,
        "data_dim": actual_data_dim,
        "model_type": ModelName if 'ModelName' in globals() else "DinkNet",
        "mt_mode": mt_mode
    }
    
    with open(config_file_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)
    
    return result_file_path, config_file_path


################################################
########            MAIN FUNCTION       ########
################################################
def main():
   
    try:
        # 1. 加载模型
        selected_model_path = None
        if len(sys.argv) > 2 and sys.argv[2] != "No model file selected" and sys.argv[2]:
            selected_model_path = sys.argv[2]
            print(f"Model path selected from GUI: {selected_model_path}")
            
            # 从模型文件名推断模型类型
            if selected_model_path:
                model_filename = os.path.basename(selected_model_path)
                if 'DinkNet' in model_filename or 'dinknet' in model_filename.lower():
                    model_type = 'DinkNet'
                elif 'UnetPlusPlus' in model_filename or 'unetplusplus' in model_filename.lower():
                    model_type = 'UnetPlusPlus'
                elif 'Unet' in model_filename or 'unet' in model_filename.lower():
                    model_type = 'UnetModel'
                else:
                    model_type = 'DinkNet'
            else:
                model_type = ModelName if 'ModelName' in globals() and ModelName else 'DinkNet'
        else:
            model_type = ModelName if 'ModelName' in globals() and ModelName else 'DinkNet'
            print(f"Using default model selection logic, model type: {model_type}")
        
        # 调用load_model函数
        net, model_file = load_model(model_type=model_type, model_path=selected_model_path)
        model_in_ch = infer_net_input_channels(net)
        print(f"Inferred model input channels from weights: {model_in_ch}")
        
        # 2. 确定 MT 模式：必须以「已加载网络的输入通道数」为准，文件名 / GUI 仅作 TE/TM 细分
        mt_mode = 'TM'  # 默认 TM（2 通道时与文件名/GUI 再对齐）
        if selected_model_path:
            model_filename = os.path.basename(selected_model_path)
            if 'ModeTE' in model_filename:
                mt_mode = 'TE'
                print(f"MT mode hint from model filename: {mt_mode}")
            elif 'ModeTM' in model_filename:
                mt_mode = 'TM'
                print(f"MT mode hint from model filename: {mt_mode}")
            elif 'ModeBoth' in model_filename:
                mt_mode = 'Both'
                print(f"MT mode hint from model filename: {mt_mode}")
        
        gui_mt_mode = normalize_gui_mt_mode(sys.argv[3] if len(sys.argv) > 3 else None)
        
        if model_in_ch == 4:
            mt_mode = 'Both'
            if gui_mt_mode and gui_mt_mode != 'Both':
                print(
                    f"Note: Model requires 4 input channels; loading test data in Both (TE+TM) layout. "
                    f"GUI mode {gui_mt_mode} does not change channel count."
                )
        elif model_in_ch == 2:
            if gui_mt_mode in ('TE', 'TM'):
                mt_mode = gui_mt_mode
                print(f"Using MT mode from GUI for 2-channel model: {mt_mode}")
            elif gui_mt_mode == 'Both':
                raise RuntimeError(
                    "This model uses 2 input channels; it is incompatible with Both mode. "
                    "Use a 4-channel model (filename with ModeBoth) or select TE/TM in the GUI."
                )
            else:
                print(f"Using MT mode from model filename (2-channel model): {mt_mode}")
        else:
            print(f"Warning: Unusual model input channel count: {model_in_ch}")
        
        # 3. 加载测试数据
        test_data_file = None
        if len(sys.argv) > 1 and sys.argv[1]:
            test_data_file = sys.argv[1]
        
        print(f"Loading test data with MT mode: {mt_mode}")
        test_set, data_dsp_dim, label_dsp_dim, file_info = load_test_data(test_data_file=test_data_file, mt_mode=mt_mode)
        
        # 验证数据通道数与模型匹配
        expected_channels = 4 if mt_mode == 'Both' else 2
        actual_channels = test_set.shape[1]
        if actual_channels != expected_channels:
            print(f"Warning: Data has {actual_channels} channels, but model expects {expected_channels} channels for {mt_mode} mode")
            print(f"This may cause prediction errors. Please check model and data compatibility.")
        
        # 4. 进行预测
        predictions = predict(net, test_set, data_dsp_dim, label_dsp_dim)
        
        # 5. 保存预测结果
        result_file_path, config_file_path = save_prediction_results(predictions, file_info, data_dsp_dim)
        
        # 6. 返回结果文件路径，供GUI使用
        print(f"Prediction completed successfully!")
        print(f"Result file: {result_file_path}")
        print(f"Config file: {config_file_path}")
        
        # 输出特殊格式的结果路径，方便GUI解析
        print(f"RESULT_PATH:{result_file_path}")
        
    except Exception as e:
        print(f"Prediction process failed: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
