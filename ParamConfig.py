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
#ModelDim      = [96,96]
label_dsp_blk = (1,1)     # Downsampling ratio of output
dh            = 10        # Space interval
DataFormat    = 'depth'   # Data format: 'depth' or 'width'


####################################################
####             NETWORK PARAMETERS             ####
####################################################

Epochs        = 200      # Number of epoch






TrainSize     = 0.80      # Training data size ratio




























































































ValSize       = 0.20      # Validation data size ratio





























































































TestSize      = 0.002      # Test data size ratio
TestBatchSize = 20
EarlyStop         = 20       # Early stopping threshold (0 means no early stopping)







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
TE_Resistivity_Dir = 'L:/DLTool/dl/data/TE/resistivity' # TE mode apparent resistivity data directory



# 数据目录路径（实际数据位于G盘）
TE_Phase_Dir = 'L:/DLTool/dl/data/TE/phase' # TE mode phase data directory

TM_Resistivity_Dir = 'M:/sdzl' # TM mode apparent resistivity data directory





















TM_Phase_Dir = 'M:/xw' # TM mode phase data directory







Resistivity_Model_Dir = 'M:/dzl' # Resistivity model directory




































# MT Mode Selection
MT_Mode = 'TM'  # MT mode: 'TE', 'TM', or 'Both'
# Input channels are automatically set based on MT mode: 2 for TE or TM single mode, 4 for mixed mode
Inchannels        = 2        # Number of input channels, i.e. the number of shots
# Pre-trained Model
PreModel      = 'model'   # Pre-trained model name
# Device Configuration
Device        = 'cuda'    # Device to use: 'cuda' or 'cpu'
# GUI Configuration Parameters
ModelName     = 'DinkNet' # Name of the model to use

# 使用正斜杠避免Unicode转义错误
ModelsDir     = 'm:\DLTool\inversion/dl/models/'   # Directory for saving models
ResultsDir    = 'm:\DLTool\inversion/dl/results/'  # Directory for saving results
