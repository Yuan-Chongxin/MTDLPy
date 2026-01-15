# -*- coding: utf-8 -*-
"""
文件位置设置

创建于2021年7月

作者：ycx

"""

from LibConfig import *
from ParamConfig import *

####################################################
####                   FILENAMES               ####
####################################################

# Data filename

datafilename = 'model'
dataname = 'model'
truthfilename = 'Rec'
truthname = 'model'

###################################################
########               PATHS              #########
###################################################

# 使用当前文件的绝对路径作为基础，确保路径的稳定性
current_file_path = os.path.abspath(__file__)
main_dir = os.path.dirname(os.path.dirname(current_file_path))  # 向上两级目录，即h:\O\DLTool

## Check the main directory
if len(main_dir) == 0:
    raise Exception('Please specify path to correct directory!')

## Data path
if os.path.exists('./data/'):
    data_dir = main_dir + '/data/'  # Replace your data path here
else:
    os.makedirs('./data/')
    data_dir = main_dir + '/data/'

# Define training/testing data directory

train_data_dir = data_dir + 'train_data/'  # Replace your training data path here
test_data_dir = data_dir + 'test_data/'  # Replace your testing data path here

# Define directory for simulate data and SEG data respectively


## Create Results and Models path

# 直接构建完整的结果和模型目录路径
results_dir = main_dir + '/dl/results/'
models_dir = main_dir + '/dl/models/'

# 确保这些目录存在，如果不存在则创建
# 首先确保主目录结构存在
os.makedirs(os.path.dirname(models_dir), exist_ok=True)
# 然后创建具体的结果和模型目录
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Create Model name

tagM = 'MT'
tagM0 = '_UnetModel'
tagM1 = '_TrainSize' + str(TrainSize)
tagM2 = '_Epoch' + str(Epochs)
tagM3 = '_BatchSize' + str(BatchSize)
tagM4 = '_LR' + str(LearnRate)

modelname = tagM + tagM0 + tagM1 + tagM2 + tagM3 + tagM4
# Change here to set the model as the pre-trained initialization
premodelname = ''

