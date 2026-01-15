# -*- coding: utf-8 -*-
"""
Load testing data set

Created on Nov 2021

@author: 袁崇鑫

"""

import numpy as np
from skimage.measure import block_reduce
import skimage
import scipy.io
import pandas as pd
import cv2
import os
import random
import matplotlib.pyplot as plt


def DataLoad_Test(test_size,test_data_dir,data_dim,in_channels,model_dim,data_dsp_blk,label_dsp_blk,start,datafilename,dataname,truthfilename,truthname):
    print(f"正在加载测试数据，测试数据目录: {test_data_dir}")
    for i in range(start,start+test_size):
        # 使用传入的test_data_dir路径，而不是硬编码路径
        filename_seis = os.path.join(test_data_dir, f"{i}.dat")
        # np.set_printoptions(threshold=np.inf)

        # 数据验证和预处理函数（与DataLoad_Train.py保持一致）
        def validate_data(data, file_id='test'):
            """校验数据是否有效"""
            invalid_info = []
            valid = True
            
            # 检查数据是否有负数或异常值
            if np.any(np.logical_or(data <= 0, data >= 100000)):
                valid = False
                invalid_info.append(f"数据包含负数或异常值")
                # 修复：将无效值限制在有效范围内
                data = np.maximum(data, 1e-6)
                data = np.minimum(data, 99999)
                print(f"已自动修复数据中的无效值")
            
            return valid, invalid_info, data
        
        odd_data = []
        even_data = []
        
        try:
            with open(filename_seis, 'r') as file:
                lines = file.readlines()
                for hh, line in enumerate(lines):
                    if hh > 7:  # 跳过标题行
                        columns = line.split()
                        if len(columns) >= 9:
                            value = float(columns[8])
                            if (hh - 1) % 2 == 0:
                                even_data.append(value)
                            else:
                                # 直接使用原始值，不进行log转换
                                odd_data.append(value)
            
            my_array1 = np.array(odd_data)
            my_array2 = np.array(even_data)
            my_array1 = my_array1.reshape(32, 32).T
            my_array2 = my_array2.reshape(32, 32).T
            
            # 对数据进行验证
            valid, invalid_info, my_array1 = validate_data(my_array1, f"test_{i}")
            
            if not valid:
                print(f"警告: 数据校验发现问题:")
                for info in invalid_info:
                    print(f"  - {info}")
            
            # 使用与训练数据相同的降采样块大小
            data1_set = block_reduce(my_array1, block_size=data_dsp_blk, func=np.mean)
            data_dsp_dim = data1_set.shape
            data1_set = data1_set.reshape(1, data_dsp_dim[0] * data_dsp_dim[1])
            train1_set = np.float32(data1_set)
        except Exception as e:
            print(f"加载测试数据文件 {filename_seis} 时出错: {str(e)}")
            # 创建默认数据以避免程序崩溃
            train1_set = np.zeros((1, 32*32), dtype=np.float32)
            data_dsp_dim = (32, 32)

        # 读取label
        # 使用truthfilename构建标签文件路径
        if truthfilename:
            filename_label = os.path.join(truthfilename, f"{i}.dat")
        else:
            # 默认使用与测试数据相同的目录
            filename_label = os.path.join(test_data_dir, f"label_{i}.dat")
        
        print(f"加载标签文件: {filename_label}")
        
        try:
            data2_set = np.loadtxt(filename_label)
            
            # 确保标签数据是二维的
            if len(data2_set.shape) == 1:
                # 尝试重塑为合适的形状
                size = int(np.sqrt(data2_set.size))
                data2_set = data2_set.reshape(size, size)
            
            # 直接使用原始值，不进行归一化
            
            # 使用与训练数据相同的降采样参数
            data2_set = block_reduce(data2_set, block_size=label_dsp_blk, func=np.mean)
            label_dsp_dim = data2_set.shape
            data2_set = data2_set.reshape(1, label_dsp_dim[0] * label_dsp_dim[1])
            data2_set = np.float32(data2_set)
        except Exception as e:
            print(f"加载标签文件 {filename_label} 时出错: {str(e)}")
            # 创建默认标签数据以避免程序崩溃
            label_dsp_dim = (16, 16)
            data2_set = np.zeros((1, label_dsp_dim[0] * label_dsp_dim[1]), dtype=np.float32)

        if i == start:
           test_set =  train1_set
           label_set = data2_set
        else:
           test_set = np.append(test_set, train1_set, axis=0)
           label_set = np.append(label_set, data2_set, axis=0)

    test_set  = test_set.reshape((test_size, in_channels, data_dsp_dim[0], data_dsp_dim[1]))
    label_set = label_set.reshape((test_size, 1, label_dsp_dim[0], label_dsp_dim[1]))
    
    print(f"测试数据加载完成，形状: {test_set.shape}")
    print(f"标签数据形状: {label_set.shape}")
    print(f"数据降采样维度: {data_dsp_dim}")
    print(f"标签降采样维度: {label_dsp_dim}")

    # 返回与DataLoad_Train.py相同的参数结构，包括min/max值（为了兼容性）
    data_min_vals = [np.min(test_set[:, channel, :, :]) for channel in range(in_channels)]
    data_max_vals = [np.max(test_set[:, channel, :, :]) for channel in range(in_channels)]
    label_min_val = np.min(label_set)
    label_max_val = np.max(label_set)
    
    print("已禁用归一化处理，使用原始数据进行测试")
    
    return test_set, label_set, data_dsp_dim, label_dsp_dim, data_min_vals, data_max_vals, label_min_val, label_max_val

# downsampling function by taking the middle value
def decimate(a,axis):
   idx = np.round((np.array(a.shape)[np.array(axis).reshape(1,-1)]+1.0)/2.0-1).reshape(-1)
   downa = np.array(a)[:,:,idx[0].astype(int)]
   return downa

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

def add_gaussian_noise(x, y):
    """
    对输入数据加入高斯噪声
    :param x: x轴数据
    :param y: y轴数据
    :return:
    """
    temp_x = np.float64(np.copy(x))
    h = x.shape[0]
    w = x.shape[1]
    noise = np.random.randn(h, w) * y

    noisy_image = np.zeros(temp_x.shape, np.float64)
    if len(temp_x.shape) == 2:
        noisy_x = temp_x + noise
    else:
        noisy_x[:, :, 0] = temp_x[:, :, 0] + noise
        noisy_x[:, :, 1] = temp_x[:, :, 1] + noise
        noisy_x[:, :, 2] = temp_x[:, :, 2] + noise
    """
    print('min,max = ', np.min(noisy_image), np.max(noisy_image))
    print('type = ', type(noisy_image[0][0][0]))
    """
    return noisy_x
