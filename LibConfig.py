# -*- coding: utf-8 -*-
"""
Import libraries

@author: 袁崇鑫

"""

################################################
########            LIBARIES            ########
################################################

import numpy as np
import torch
import os, sys
sys.path.append(os.getcwd())
import time
import pdb
import argparse
import torch
import torch.nn as nn
import scipy.io
import torch.utils.data as data_utils
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

# 解决libpng警告：iCCP: known incorrect sRGB profile
# 在导入matplotlib之前设置警告抑制
import warnings
import logging

# 抑制所有matplotlib相关的字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*findfont.*')
warnings.filterwarnings('ignore', message='.*Font family.*not found.*')
warnings.filterwarnings('ignore', message='.*findfont: Font family.*')
# 设置matplotlib字体管理器的日志级别为ERROR
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

import matplotlib
matplotlib.rcParams['image.cmap'] = 'viridis'
matplotlib.use('Agg')
import matplotlib.pylab as plt
import matplotlib.font_manager as fm

# 智能配置字体，避免字体警告
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    available_fonts = [f.name for f in fm.fontManager.ttflist]
font_candidates = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial", "DejaVu Sans"]
selected_font = None
for font_name in font_candidates:
    if font_name in available_fonts:
        selected_font = font_name
        break
if selected_font:
    plt.rcParams["font.family"] = [selected_font]
else:
    plt.rcParams["font.family"] = ["sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 注意：警告抑制已经在导入matplotlib之前设置，这里不需要重复设置
from mpl_toolkits.axes_grid1 import make_axes_locatable
from func.utils import *
from func.UnetModel import UnetModel
from func.dunet import DUNet
from func.segbase import SegBaseModel
from func.enet import ENet
#from func.swnet import ENet
from func.DataLoad_Train import DataLoad_Train
from func.DataLoad_Test import DataLoad_Test
from func.utils import turn, PSNR, SSIM
from func.dinknet import *
# 已移除所有Transformer相关的导入