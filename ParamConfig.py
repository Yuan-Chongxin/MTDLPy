# -*- coding: utf-8 -*-
"""
Parameter Configuration

Created in July 2021

Author: ycx

"""


####################################################
#####            MAIN PARAMETERS               #####
####################################################

ReUse         = True      # Whether to reuse the pre-trained model
#DataDim       = [96,96]
DataDim       = [32,32] # Size of the training data
data_dsp_blk  = (1,1)     # Downsampling ratio of input
ModelDim      = [32,32] # Size of the output model
# 磁盘上网格 (行, 列)：当 .txt 为单列/单行 1D 且长度=行×列时，用其 reshape 再插值到 DataDim/ModelDim。
# 若为二维多列文本（np.loadtxt 得二维），可任意 (h,w)，不依赖本项。1D 且长度与下式都不匹配时，DataLoad_Train.ensure_grid_shape 会按因子分解自动推断行列再插值。
RawGridShape  = (13, 11)
# True：在插值前对「形状恰好等于 RawGridShape」的二维数组先 .T（得到 11×13），与原先 32×32 流程的轴向一致
RawGridTransposeBeforeResize = True
# 约定（勿随意改注释含义）：视电阻率/相位二维读入常为 (13,11)，须先经上式 .T 才成 11 行×13 列 (11,13)，再 resize 到如 32×32。
# 反演模型标签：不做转置即为 11 行×13 列 (11,13)，再插值到 ModelDim（如 32×32）；训练代码对标签无 ensure 后的 .T。
#ModelDim      = [96,96]
label_dsp_blk = (1,1)     # Downsampling ratio of output
dh            = 10        # Space interval
DataFormat    = 'depth'   # Data format: 'depth' or 'width'

# 预测结果写入 results 前对 2D 数组的空间修正（与训练时预处理轴向一致）：
# 训练里视电阻率/相位：ensure_grid_shape(...).T；标签：ensure_grid_shape(...)（规范 11×13 不经 RawGrid 转置）且无训练用 .T。
# 网络输出与「标签」同索引约定；若剖面与观测/地质图相比整体偏了约 90°，可改为下面其一后重新预测：
#   'none'       — 与训练标签 txt 完全一致（默认，兼容旧流程）
#   'transpose'  — 保存前 .T（交换行列，常与「仅 ensure、无训练用 .T」的观测显示对齐）
#   'rot90_cw'   — 保存前顺时针旋转 90°（np.rot90(..., k=-1)）
PredictionOutputSpatialFix = 'none'


####################################################
####             NETWORK PARAMETERS             ####
####################################################

Epochs        = 200      # Number of epoch













TrainSize     = 0.80      # Training data size ratio



































































































ValSize       = 0.20      # Validation data size ratio




































































































TestSize      = 0.002      # Test data size ratio
TestBatchSize = 20
EarlyStop         = 10       # Early stopping threshold (0 means no early stopping)














BatchSize         = 8       # Number of batch size













LearnRate         = 0.00000001      # Learning rate













Optimizer         = 'Adam'      # Optimizer type (Adam, SGD, RMSprop, AdamW)












WeightDecay         = 0.000000      # Weight decay (L2 regularization)
Nclasses          = 1        # Number of output channels
# Inchannels will be automatically set based on MT_Mode defined below
# No longer need SaveEpoch parameter, model will auto-save based on validation loss
DisplayStep       = 10       # Display training information every x step

# Physical Constraints - No longer used
# Phys_Con and related parameters have been removed

# File Paths
DataDir       = 'data/'   # Data directory
ModelDir      = 'm:\DLTool\inversion/dl/models/' # Model directory
ResultDir     = 'm:\DLTool\inversion/dl/results/' # Result directory

# MT Mode Specific Directories
TE_Resistivity_Dir = 'M:/DLTool/TE_S' # TE mode apparent resistivity data directory










# 数据目录路径（实际数据位于G盘）
TE_Phase_Dir = 'M:/DLTool/TE_X' # TE mode phase data directory



TM_Resistivity_Dir = 'H:/sdzl' # TM mode apparent resistivity data directory



































TM_Phase_Dir = 'H:/xw' # TM mode phase data directory











Resistivity_Model_Dir = 'H:/dzl' # Resistivity model directory






















































# MT Mode Selection
MT_Mode = 'Both'  # MT mode: 'TE', 'TM', or 'Both'
# Input channels are automatically set based on MT mode: 2 for TE or TM single mode, 4 for mixed mode
Inchannels        = 4        # Number of input channels, i.e. the number of shots
# Pre-trained Model
PreModel      = 'model'   # Pre-trained model name
# Device Configuration
Device        = 'cuda'    # Device to use: 'cuda' or 'cpu'
# GUI Configuration Parameters
ModelName     = 'DinkNet' # Name of the model to use

# 使用正斜杠避免Unicode转义错误
ModelsDir     = 'm:\DLTool\inversion/dl/models/'   # Directory for saving models
ResultsDir    = 'm:\DLTool\inversion/dl/results/'  # Directory for saving results
