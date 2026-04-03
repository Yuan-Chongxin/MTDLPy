# -*- coding: utf-8 -*-
"""
Load training data.

Created July 2021.

Author: ycx

"""

import numpy as np
from skimage.measure import block_reduce
import skimage
import scipy.io
import pandas as pd
from IPython.core.debugger import set_trace
from scipy.interpolate import lagrange
import os
import math
import cv2


def _get_raw_grid_shape():
    try:
        from ParamConfig import RawGridShape as rgs
        return rgs
    except (ImportError, AttributeError):
        return None


def _get_raw_grid_transpose():
    try:
        from ParamConfig import RawGridTransposeBeforeResize as t
        return bool(t)
    except (ImportError, AttributeError):
        return False


def _infer_1d_grid_shape(length, target_hw, raw_hw=None):
    """
    When 1D length matches neither the target grid nor RawGridShape, infer (rows, cols) by factorization
    so flattened 1D data can be reshaped to 2D then interpolated to DataDim/ModelDim.
    Prefer raw_hw if product matches; else perfect square; else factor pair with aspect ratio closest to target.
    """
    th, tw = int(target_hw[0]), int(target_hw[1])
    if length <= 0:
        raise ValueError("Spatial data length is 0; cannot infer grid shape")
    if raw_hw is not None:
        rh, rw = int(raw_hw[0]), int(raw_hw[1])
        if length == rh * rw:
            return rh, rw
    if length == th * tw:
        return th, tw
    s = int(math.isqrt(length))
    if s * s == length:
        return s, s
    target_ar = (th / float(tw)) if tw > 0 else 1.0
    best_h, best_w = None, None
    best_score = float("inf")
    for h in range(1, s + 1):
        if length % h != 0:
            continue
        w = length // h
        for hh, ww in ((h, w), (w, h)):
            ar = hh / float(ww) if ww > 0 else 1.0
            score = abs(math.log(ar + 1e-9) - math.log(target_ar + 1e-9))
            if score < best_score:
                best_score = score
                best_h, best_w = hh, ww
    if best_h is None:
        raise ValueError(
            "1D spatial data length %d cannot be factored into integer rows x cols; use multi-column text "
            "(2D from np.loadtxt) or set ParamConfig RawGridShape=(rows,cols) with rows*cols=%d"
            % (length, length)
        )
    return best_h, best_w


def ensure_grid_shape(arr, target_hw, raw_hw=None):
    """
    Reshape apparent resistivity / phase / model grids to 2D (target_h, target_w) for .T and further steps.
    target_hw is (rows, cols) like numpy shape; cv2.resize dsize is (cols, rows).
    - 2D: any (h,w) is linearly interpolated to target; RawGridShape not required.
    - 1D: reshape to target if length matches; else reshape using RawGridShape product if it matches;
      else infer shape, reshape, then interpolate (shared by training and MT_test).
    If RawGridTransposeBeforeResize is True and 2D shape equals raw_hw, transpose before resize.
    Default (13,11) with flag True: (13,11) input is transposed to (11,13) then resized; labels at (11,13) skip that .T.
    """
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    a = np.asarray(arr)
    if a.ndim == 1:
        need = target_h * target_w
        if a.size == need:
            a = a.reshape(target_h, target_w)
        elif raw_hw is not None:
            rh, rw = int(raw_hw[0]), int(raw_hw[1])
            if a.size == rh * rw:
                a = a.reshape(rh, rw)
            else:
                ih, iw = _infer_1d_grid_shape(a.size, target_hw, raw_hw=None)
                a = a.reshape(ih, iw)
        else:
            ih, iw = _infer_1d_grid_shape(a.size, target_hw, raw_hw=None)
            a = a.reshape(ih, iw)
    elif a.ndim == 2:
        pass
    else:
        raise ValueError("Expected 1D or 2D spatial data; got shape=%s" % (a.shape,))

    if raw_hw is not None and _get_raw_grid_transpose():
        rh, rw = int(raw_hw[0]), int(raw_hw[1])
        if a.shape == (rh, rw):
            a = np.ascontiguousarray(a.T)

    if a.shape[0] != target_h or a.shape[1] != target_w:
        a = cv2.resize(
            np.ascontiguousarray(a, dtype=np.float32),
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR,
        )
    return np.asarray(a, dtype=np.float64)


def validate_data(resistivity_data, phase_data, model_data, file_id, mode):
    """
    Validate apparent resistivity, phase, and resistivity model data.

    Returns:
    - valid: bool
    - invalid_files: list of human-readable issue strings
    """
    invalid_files = []
    valid = True
    
    # 检查视电阻率数据是否有负数或异常值
    if np.any(np.logical_or(resistivity_data <= 0, resistivity_data >= 100000)):
        valid = False
        invalid_files.append(f"{mode} apparent resistivity file (ID: {file_id}) has non-positive or out-of-range values")
        
    # 检查相位数据是否在0-90之间
    if np.any(phase_data < 0) or np.any(phase_data > 90):
        valid = False
        invalid_files.append(f"{mode} phase file (ID: {file_id}) has values outside [0, 90]")
        
    # 检查电阻率模型数据是否有负数或异常值
    if np.any(np.logical_or(model_data <= 0, model_data >= 100000)):
        valid = False
        invalid_files.append(f"Resistivity model file (ID: {file_id}) has non-positive or out-of-range values")
    
    return valid, invalid_files

def DataLoad_Train(train_size, train_data_dir, data_dim, in_channels, model_dim, data_dsp_blk, label_dsp_blk, start,
                   datafilename, dataname, truthfilename, truthname, 
                   TE_Resistivity_Dir, TE_Phase_Dir, TM_Resistivity_Dir, TM_Phase_Dir,
                  Resistivity_Model_Dir, MT_Mode):
    """
    Load training data for TE, TM, or Both; validate apparent resistivity, phase, and model.
    Spatial pipeline: apparent data often (13,11), transpose inside ensure to (11,13), resize to data_dim, then training .T on inputs.
    Labels: (11,13) without extra transpose, resize to model_dim; no second .T on labels in loader.

    Returns train_set, label_set, data_dsp_dim, label_dsp_dim, valid_count.
    """
    import time
    start_time = time.time()
    print(f"[DataLoad_Train] Loading training data, up to {train_size} samples...")
    
    invalid_files_list = []
    valid_count = 0
    _raw_gs = _get_raw_grid_shape()
    _th, _tw = int(data_dim[0]), int(data_dim[1])
    _mh, _mw = int(model_dim[0]), int(model_dim[1])

    # 添加错误处理，确保函数在KeyboardInterrupt时也能提供有意义的信息
    try:
        if in_channels == 2 and MT_Mode == 'TE':
            for i in range(start, start + train_size):
                # 每处理10个数据打印一次进度，方便监控
                if (i - start) % 10 == 0 and (i - start) > 0:
                    print(f"[DataLoad_Train] Processed {i - start} samples, valid: {valid_count}")
                
                # 加载原始数据用于校验
                # 确保目录路径以斜杠结尾
                te_res_dir = TE_Resistivity_Dir + '/' if not TE_Resistivity_Dir.endswith('/') and not TE_Resistivity_Dir.endswith('\\') else TE_Resistivity_Dir
                te_ph_dir = TE_Phase_Dir + '/' if not TE_Phase_Dir.endswith('/') and not TE_Phase_Dir.endswith('\\') else TE_Phase_Dir
                res_mod_dir = Resistivity_Model_Dir + '/' if not Resistivity_Model_Dir.endswith('/') and not Resistivity_Model_Dir.endswith('\\') else Resistivity_Model_Dir
                
                filename_seis1 = te_res_dir + str(i) + '.txt'
                #print(filename_seis1)
                raw_resistivity = np.loadtxt(filename_seis1, encoding='utf-8')
                
                filename_seis2 = te_ph_dir + str(i) + '.txt'
                #print(filename_seis2)
                raw_phase = np.loadtxt(filename_seis2, encoding='utf-8')
                
                filename1_label1 = res_mod_dir + 'zz' + str(i) + '.txt'
                #print(filename1_label1)
                raw_model = np.loadtxt(filename1_label1, encoding='utf-8')
                
                # 数据校验
                is_valid, invalid_files = validate_data(raw_resistivity, raw_phase, raw_model, i, 'TE')
                if is_valid:
                    # 数据有效，继续处理（任意原始网格先插值到 data_dim，与 ModelDim 一致）
                    train_data1 = ensure_grid_shape(raw_resistivity, (_th, _tw), _raw_gs).T
                    train_data1 = np.log10(train_data1)
                    
                    train_data2 = ensure_grid_shape(raw_phase, (_th, _tw), _raw_gs).T
                    
                    data1_set = np.array([train_data1, train_data2])
                    data1_set = np.transpose(data1_set, (1, 2, 0))
                    
                    for k in range(0, in_channels):
                        data11_set= np.float32(data1_set[:, :, k])
                        data11_set = np.float32(data11_set)
                        # Data downsampling
                        data11_set = block_reduce(data11_set, block_size=data_dsp_blk, func=decimate)
                        data_dsp_dim = data11_set.shape
                        data11_set = data11_set.reshape(1, data_dsp_dim[0] * data_dsp_dim[1])
                        if k == 0:
                            data_set = data11_set
                        else:
                            data_set = np.append(data_set, data11_set, axis=0)
                    
                    train_label1 = ensure_grid_shape(raw_model, (_mh, _mw), _raw_gs)
                    # 对电阻率label进行对数转换，保持与输入数据处理一致性
                    train_label1 = np.log10(train_label1)
                    train_label1 = block_reduce(train_label1, block_size=label_dsp_blk, func=np.max)
                    label_dsp_dim = train_label1.shape
                    train_label1 = train_label1.reshape(1, label_dsp_dim[0] * label_dsp_dim[1])
                    train_label1 = np.float32(train_label1)
                    
                    if valid_count == 0:
                        # 第一次初始化数组
                        train_set = data_set
                        label_set = train_label1
                    else:
                        # 使用np.append添加新数据
                        train_set = np.append(train_set, data_set, axis=0)
                        label_set = np.append(label_set, train_label1, axis=0)
                    valid_count += 1
                else:
                    # 数据无效，记录需要删除的文件
                    invalid_files_list.extend(invalid_files)
                    print(f"Skipping invalid sample (ID: {i})")
                    
        elif in_channels == 2 and MT_Mode == 'TM':
            for i in range(start, start + train_size):
                # 每处理10个数据打印一次进度，方便监控
                if (i - start) % 10 == 0 and (i - start) > 0:
                    print(f"[DataLoad_Train] Processed {i - start} samples, valid: {valid_count}")
                    
                # 加载原始数据用于校验
                # 确保目录路径以斜杠结尾
                tm_res_dir = TM_Resistivity_Dir + '/' if not TM_Resistivity_Dir.endswith('/') and not TM_Resistivity_Dir.endswith('\\') else TM_Resistivity_Dir
                tm_ph_dir = TM_Phase_Dir + '/' if not TM_Phase_Dir.endswith('/') and not TM_Phase_Dir.endswith('\\') else TM_Phase_Dir
                res_mod_dir = Resistivity_Model_Dir + '/' if not Resistivity_Model_Dir.endswith('/') and not Resistivity_Model_Dir.endswith('\\') else Resistivity_Model_Dir
                
                filename_seis1 = tm_res_dir + str(i) + '.txt'
                #print(filename_seis1)
                raw_resistivity = np.loadtxt(filename_seis1, encoding='utf-8')
                
                filename_seis2 = tm_ph_dir + str(i) + '.txt'
                #print(filename_seis2)
                raw_phase = np.loadtxt(filename_seis2, encoding='utf-8')
                
                filename1_label1 = res_mod_dir + 'zz' +str(i) + '.txt'
                #print(filename1_label1)
                raw_model = np.loadtxt(filename1_label1, encoding='utf-8')
                
                # 数据校验
                is_valid, invalid_files = validate_data(raw_resistivity, raw_phase, raw_model, i, 'TM')
                if is_valid:
                    # 数据有效，继续处理
                    train_data1 = ensure_grid_shape(raw_resistivity, (_th, _tw), _raw_gs).T
                    train_data1 = np.log10(train_data1)
                    
                    train_data2 = ensure_grid_shape(raw_phase, (_th, _tw), _raw_gs).T
                    
                    data1_set = np.array([train_data1, train_data2])
                    data1_set = np.transpose(data1_set, (1, 2, 0))
                    
                    # 对数据进行降采样处理并定义data_dsp_dim
                    for k in range(0, in_channels):
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
                    
                    train_label1 = ensure_grid_shape(raw_model, (_mh, _mw), _raw_gs)
                    # 对电阻率label进行对数转换，保持与输入数据处理一致性
                    train_label1 = np.log10(train_label1)
                    train_label1 = block_reduce(train_label1, block_size=label_dsp_blk, func=np.max)
                    label_dsp_dim = train_label1.shape
                    train_label1 = train_label1.reshape(1, label_dsp_dim[0] * label_dsp_dim[1])
                    train_label1 = np.float32(train_label1)
                    
                    if valid_count == 0:
                        train_set = data_set
                        label_set = train_label1
                    else:
                        train_set = np.append(train_set, data_set, axis=0)
                        label_set = np.append(label_set, train_label1, axis=0)
                    valid_count += 1
                else:
                    # 数据无效，记录需要删除的文件
                    invalid_files_list.extend(invalid_files)
                    print(f"Skipping invalid sample (ID: {i})")
        elif in_channels == 4:
            for i in range(start, start + train_size):
                # 每处理10个数据打印一次进度，方便监控
                if (i - start) % 10 == 0 and (i - start) > 0:
                    print(f"[DataLoad_Train] Processed {i - start} samples, valid: {valid_count}")
                    
                # 加载原始数据用于校验
                # 确保目录路径以斜杠结尾
                te_res_dir = TE_Resistivity_Dir + '/' if not TE_Resistivity_Dir.endswith('/') and not TE_Resistivity_Dir.endswith('\\') else TE_Resistivity_Dir
                te_ph_dir = TE_Phase_Dir + '/' if not TE_Phase_Dir.endswith('/') and not TE_Phase_Dir.endswith('\\') else TE_Phase_Dir
                tm_res_dir = TM_Resistivity_Dir + '/' if not TM_Resistivity_Dir.endswith('/') and not TM_Resistivity_Dir.endswith('\\') else TM_Resistivity_Dir
                tm_ph_dir = TM_Phase_Dir + '/' if not TM_Phase_Dir.endswith('/') and not TM_Phase_Dir.endswith('\\') else TM_Phase_Dir
                res_mod_dir = Resistivity_Model_Dir + '/' if not Resistivity_Model_Dir.endswith('/') and not Resistivity_Model_Dir.endswith('\\') else Resistivity_Model_Dir
                
                filename_seis1 = te_res_dir + str(i) + '.txt'
                #print(filename_seis1)
                raw_te_resistivity = np.loadtxt(filename_seis1, encoding='utf-8')
                
                filename_seis2 = te_ph_dir + str(i) + '.txt'
                #print(filename_seis2)
                raw_te_phase = np.loadtxt(filename_seis2, encoding='utf-8')
                
                filename_seis3 = tm_res_dir + str(i) + '.txt'
                #print(filename_seis3)
                raw_tm_resistivity = np.loadtxt(filename_seis3, encoding='utf-8')
                
                filename_seis4 = tm_ph_dir + str(i) + '.txt'
                #print(filename_seis4)
                raw_tm_phase = np.loadtxt(filename_seis4, encoding='utf-8')
                
                filename1_label1 = res_mod_dir + 'zz' +str(i) + '.txt'
                #print(filename1_label1)
                raw_model = np.loadtxt(filename1_label1, encoding='utf-8')
                
                # 数据校验 - Both模式下，任何一个模式数据无效则整个数据无效
                te_valid, te_invalid_files = validate_data(raw_te_resistivity, raw_te_phase, raw_model, i, 'TE')
                tm_valid, tm_invalid_files = validate_data(raw_tm_resistivity, raw_tm_phase, raw_model, i, 'TM')
                
                if te_valid and tm_valid:
                    # 数据有效，继续处理
                    train_data1 = ensure_grid_shape(raw_te_resistivity, (_th, _tw), _raw_gs).T
                    train_data1 = np.log10(train_data1)
                    
                    train_data2 = ensure_grid_shape(raw_te_phase, (_th, _tw), _raw_gs).T
                    
                    train_data3 = ensure_grid_shape(raw_tm_resistivity, (_th, _tw), _raw_gs).T
                    train_data3 = np.log10(train_data3)
                    
                    train_data4 = ensure_grid_shape(raw_tm_phase, (_th, _tw), _raw_gs).T
                    
                    data1_set = np.array([train_data1, train_data2, train_data3, train_data4])
                    data1_set = np.transpose(data1_set, (1, 2, 0))
                    
                    # 对数据进行降采样处理并定义data_dsp_dim
                    for k in range(0, in_channels):
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
                    
                    train_label1 = ensure_grid_shape(raw_model, (_mh, _mw), _raw_gs)
                    # 对电阻率label进行对数转换，保持与输入数据处理一致性
                    train_label1 = np.log10(train_label1)
                    train_label1 = block_reduce(train_label1, block_size=label_dsp_blk, func=np.max)
                    label_dsp_dim = train_label1.shape
                    train_label1 = train_label1.reshape(1, label_dsp_dim[0] * label_dsp_dim[1])
                    train_label1 = np.float32(train_label1)
                    
                    if valid_count == 0:
                        train_set = data_set
                        label_set = train_label1
                    else:
                        train_set = np.append(train_set, data_set, axis=0)
                        label_set = np.append(label_set, train_label1, axis=0)
                    valid_count += 1
                else:
                    # 数据无效，记录需要删除的文件
                    if not te_valid:
                        invalid_files_list.extend(te_invalid_files)
                    if not tm_valid:
                        invalid_files_list.extend(tm_invalid_files)
                    print(f"Skipping invalid sample (ID: {i})")
        
        # 如果valid_count为0，说明没有有效的训练数据，抛出异常
        if valid_count == 0:
            raise ValueError("No valid training samples; check data files")
        
        print(f"Reshaping dataset...")
        # 调整train_set的形状为有效的训练数据个数
        train_set = train_set.reshape((valid_count, in_channels, data_dsp_dim[0] * data_dsp_dim[1]))
        label_set = label_set.reshape((valid_count, 1, label_dsp_dim[0] * label_dsp_dim[1]))
        
        print(f"Reshape done")
        print(f"train_set shape: {train_set.shape}")
        print(f"label_set shape: {label_set.shape}")
        print(f"Load time: {time.time() - start_time:.2f} s")
        
        # 打印校验结果
        print(f"\nValidation summary:")
        print(f"Requested samples: {train_size}")
        print(f"Valid samples: {valid_count}")
        print(f"Invalid samples: {train_size - valid_count}")
        
        if invalid_files_list:
            print(f"\nInvalid entries:")
            for file_info in invalid_files_list:
                print(f"- {file_info}")
        else:
            print("All loaded files passed validation")

        # 根据用户要求，不返回归一化相关参数，只返回5个必要参数
        return train_set, label_set, data_dsp_dim, label_dsp_dim, valid_count
        
    except KeyboardInterrupt:
        print(f"\n[DataLoad_Train] Load interrupted by user")
        print(f"Processed {i - start + 1}/{train_size} samples")
        print(f"Valid samples loaded: {valid_count}")
        raise
    except Exception as e:
        print(f"\n[DataLoad_Train] Load error: {str(e)}")
        raise


# 改进的降采样函数，直接返回块的平均值
def decimate(a, axis):
    """
    Block mean for downsampling; used with block_reduce.
    """
    return np.mean(a, axis=axis)


def updateFile(file, old_str, new_str):
    """
    Write lines with old_str replaced to a .bak file, remove original, rename .bak to file.
    """
    with open(file, "r", encoding="utf-8") as f1, open("%s.bak" % file, "w", encoding="utf-8") as f2:
        for line in f1:
            if old_str in line:
                line = line.replace(old_str, new_str)
            f2.write(line)
    os.remove(file)
    os.rename("%s.bak" % file, file)


def normalize_data(data, min_val, max_val):
    """Scale data to [0, 1] using min_val and max_val."""
    return (data - min_val) / (max_val - min_val)

def denormalize_data(data, min_val, max_val):
    """Inverse of normalize_data."""
    return data * (max_val - min_val) + min_val

def get_normal_data(data1):
    """Simple normalization helper (legacy)."""
    amin = 0.1
    amax = 1000000
    return normalize_data(data1, amin, amax)
