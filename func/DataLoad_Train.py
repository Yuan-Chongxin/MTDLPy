# -*- coding: utf-8 -*-
"""
读取训练数据

创建于2021年7月

作者：ycx

"""

import numpy as np
from skimage.measure import block_reduce
import skimage
import scipy.io
import pandas as pd
from IPython.core.debugger import set_trace
from scipy.interpolate import lagrange
import os
import cv2

def validate_data(resistivity_data, phase_data, model_data, file_id, mode):
    """
    校验视电阻率、相位和电阻率模型数据是否有效
    
    参数:
    - resistivity_data: 视电阻率数据
    - phase_data: 相位数据
    - model_data: 电阻率模型数据
    - file_id: 文件编号
    - mode: 模式（'TE'或'TM'）
    
    返回:
    - 布尔值：数据是否有效
    - 列表：需要删除的文件路径
    """
    invalid_files = []
    valid = True
    
    # 检查视电阻率数据是否有负数或异常值
    if np.any(np.logical_or(resistivity_data <= 0, resistivity_data >= 100000)):
        valid = False
        invalid_files.append(f"{mode}视电阻率文件(ID: {file_id})包含负数或值异常")
        
    # 检查相位数据是否在0-90之间
    if np.any(phase_data < 0) or np.any(phase_data > 90):
        valid = False
        invalid_files.append(f"{mode}相位文件(ID: {file_id})值不在0-90区间")
        
    # 检查电阻率模型数据是否有负数或异常值
    if np.any(np.logical_or(model_data <= 0, model_data >= 100000)):
        valid = False
        invalid_files.append(f"电阻率模型文件(ID: {file_id})包含负数或值异常")
    
    return valid, invalid_files

def DataLoad_Train(train_size, train_data_dir, data_dim, in_channels, model_dim, data_dsp_blk, label_dsp_blk, start,
                   datafilename, dataname, truthfilename, truthname, 
                   TE_Resistivity_Dir, TE_Phase_Dir, TM_Resistivity_Dir, TM_Phase_Dir,
                  Resistivity_Model_Dir, MT_Mode):
    """
    加载训练数据，支持TE、TM或Both模式，并对视电阻率、相位和电阻率模型进行校验
    
    参数:
    - train_size: 训练数据大小
    - in_channels: 输入通道数（2对应TE或TM模式，4对应Both模式）
    - MT_Mode: MT模式（'TE'、'TM'或'Both'）
    - 其他参数为数据路径和处理参数
    
    返回:
    - train_set: 训练数据集
    - label_set: 标签数据集
    - data_dsp_dim: 数据降采样维度
    - label_dsp_dim: 标签降采样维度
    - valid_count: 有效的训练数据个数
    """
    import time
    start_time = time.time()
    print(f"[DataLoad_Train] 开始加载训练数据，预计处理 {train_size} 个样本...")
    
    invalid_files_list = []
    valid_count = 0
    
    # 添加错误处理，确保函数在KeyboardInterrupt时也能提供有意义的信息
    try:
        if in_channels == 2 and MT_Mode == 'TE':
            for i in range(start, start + train_size):
                # 每处理10个数据打印一次进度，方便监控
                if (i - start) % 10 == 0 and (i - start) > 0:
                    print(f"[DataLoad_Train] 已处理 {i - start} 个样本，有效数据: {valid_count}")
                
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
                    # 数据有效，继续处理
                    train_data1 = np.reshape(raw_resistivity, (32, 32)).T
                    train_data1 = np.log10(train_data1)
                    
                    train_data2 = np.reshape(raw_phase, (32, 32)).T
                    
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
                    
                    train_label1 = raw_model
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
                    print(f"跳过无效数据(ID: {i})")
                    
        elif in_channels == 2 and MT_Mode == 'TM':
            for i in range(start, start + train_size):
                # 每处理10个数据打印一次进度，方便监控
                if (i - start) % 10 == 0 and (i - start) > 0:
                    print(f"[DataLoad_Train] 已处理 {i - start} 个样本，有效数据: {valid_count}")
                    
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
                    train_data1 = np.reshape(raw_resistivity, (32, 32)).T
                    train_data1 = np.log10(train_data1)
                    
                    train_data2 = np.reshape(raw_phase, (32, 32)).T
                    
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
                    
                    train_label1 = raw_model
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
                    print(f"跳过无效数据(ID: {i})")
        elif in_channels == 4:
            for i in range(start, start + train_size):
                # 每处理10个数据打印一次进度，方便监控
                if (i - start) % 10 == 0 and (i - start) > 0:
                    print(f"[DataLoad_Train] 已处理 {i - start} 个样本，有效数据: {valid_count}")
                    
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
                    train_data1 = np.reshape(raw_te_resistivity, (32, 32)).T
                    train_data1 = np.log10(train_data1)
                    
                    train_data2 = np.reshape(raw_te_phase, (32, 32)).T
                    
                    train_data3 = np.reshape(raw_tm_resistivity, (32, 32)).T
                    train_data3 = np.log10(train_data3)
                    
                    train_data4 = np.reshape(raw_tm_phase, (32, 32)).T
                    
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
                    
                    train_label1 = raw_model
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
                    print(f"跳过无效数据(ID: {i})")
        
        # 如果valid_count为0，说明没有有效的训练数据，抛出异常
        if valid_count == 0:
            raise ValueError("没有有效的训练数据，请检查数据文件质量")
        
        print(f"正在调整数据集形状...")
        # 调整train_set的形状为有效的训练数据个数
        train_set = train_set.reshape((valid_count, in_channels, data_dsp_dim[0] * data_dsp_dim[1]))
        label_set = label_set.reshape((valid_count, 1, label_dsp_dim[0] * label_dsp_dim[1]))
        
        print(f"数据集形状调整完成")
        print(f"训练集形状: {train_set.shape}")
        print(f"标签集形状: {label_set.shape}")
        print(f"数据加载耗时: {time.time() - start_time:.2f} 秒")
        
        # 打印校验结果
        print(f"\n数据校验完成:")
        print(f"原始训练数据个数: {train_size}")
        print(f"有效训练数据个数: {valid_count}")
        print(f"无效数据个数: {train_size - valid_count}")
        
        if invalid_files_list:
            print(f"\n无效文件列表:")
            for file_info in invalid_files_list:
                print(f"- {file_info}")
        else:
            print("所有数据文件均有效")

        # 根据用户要求，不返回归一化相关参数，只返回5个必要参数
        return train_set, label_set, data_dsp_dim, label_dsp_dim, valid_count
        
    except KeyboardInterrupt:
        print(f"\n[DataLoad_Train] 数据加载被用户中断!")
        print(f"已处理 {i - start + 1}/{train_size} 个样本")
        print(f"已加载 {valid_count} 个有效数据")
        raise
    except Exception as e:
        print(f"\n[DataLoad_Train] 数据加载出错: {str(e)}")
        raise


# 改进的降采样函数，直接返回块的平均值
def decimate(a, axis):
    """
    简化的降采样函数，直接返回块的平均值
    这比原来的实现更可靠，适用于block_reduce函数
    """
    return np.mean(a, axis=axis)


def updateFile(file, old_str, new_str):
    """
    将替换的字符串写到一个新的文件中，然后将原文件删除，新文件改为原来文件的名字
    :param file: 文件路径
    :param old_str: 需要替换的字符串
    :param new_str: 替换的字符串
    :return: None
    """
    with open(file, "r", encoding="utf-8") as f1, open("%s.bak" % file, "w", encoding="utf-8") as f2:
        for line in f1:
            if old_str in line:
                line = line.replace(old_str, new_str)
            f2.write(line)
    os.remove(file)
    os.rename("%s.bak" % file, file)


def normalize_data(data, min_val, max_val):
    """标准化数据到[0,1]区间
    
    参数:
    - data: 输入数据
    - min_val: 最小值
    - max_val: 最大值
    
    返回:
    - 标准化后的数据
    """
    return (data - min_val) / (max_val - min_val)

def denormalize_data(data, min_val, max_val):
    """反标准化数据
    
    参数:
    - data: 标准化后的数据
    - min_val: 原始最小值
    - max_val: 原始最大值
    
    返回:
    - 反标准化后的数据
    """
    return data * (max_val - min_val) + min_val

def get_normal_data(data1):
    """简单归一化函数，保留用于兼容性"""
    amin = 0.1
    amax = 1000000
    return normalize_data(data1, amin, amax)
