# -*- coding: utf-8 -*-
"""
网络训练

创建于2021年7月

作者：ycx

"""

################################################
########        IMPORT LIBARIES         ########
################################################

# 确保正确导入配置模块，添加错误处理和路径调整
import sys
import os
import traceback
import torch
import torch.utils.data as data_utils
import numpy as np

# 添加当前脚本所在目录到Python路径，确保能正确导入配置模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 添加详细的环境信息打印
print("[DEBUG INFO]")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
    print(f"CUDA内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"NumPy版本: {np.__version__}")
print(f"Python版本: {sys.version}")
print("[DEBUG INFO END]")
print()

# 导入自定义模块
print("[DEBUG] 开始导入自定义模块...")
import func.DataLoad_Train as DLTrain
from func.dinknet import DinkNet50
print("[DEBUG] 已导入DataLoad_Train和DinkNet50")
print("[DEBUG] 所有自定义模块导入完成")

# 确保我们可以正常运行，即使配置模块导入失败
try:
    from ParamConfig import *
    from PathConfig import *
    from LibConfig import *
    CONFIG_MODULES_AVAILABLE = True
    print(f'[MT_TRAIN] 成功导入配置模块')
except ImportError as e:
    print(f'[MT_TRAIN] 警告: 导入配置模块失败: {e}')
    CONFIG_MODULES_AVAILABLE = False
    # 设置默认值以确保程序能继续运行
    ReUse = True
    DataDim = [32,32]
    data_dsp_blk = (1,1)
    ModelDim = [32,32]
    label_dsp_blk = (1,1)
    dh = 10
    DataFormat = 'depth'
    Epochs = 10  # 确保训练10个epoch
    TrainSize = 0.1  # 增加训练集大小
    TestSize = 0.01
    TestBatchSize = 20
    EarlyStop = 5
    BatchSize = 4  # 减小batch size以减少内存使用
    LearnRate = 0.0001
    Nclasses = 1
    Inchannels = 2  # 根据实际数据设置
    ValSize = 0.2  # 添加验证集大小默认值
    SaveEpoch = 1
    DisplayStep = 1
    # Phys_Con and lambdaphy parameters have been removed
    DataDir = 'data/'
    # 使用绝对路径确保正确性
    ModelDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/')
    ResultDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/')
    ModelsDir = ModelDir
    ResultsDir = ResultDir
    TE_Resistivity_Dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/TE/resistivity/')
    TE_Phase_Dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/TE/phase/')
    TM_Resistivity_Dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/TM/resistivity/')
    TM_Phase_Dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/TM/phase/')
    Resistivity_Model_Dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/models/')
    MT_Mode = 'TM'  # 设置为TM模式
    PreModel = 'model'
    Device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ModelName = 'DinkNet'  # 默认使用DinkNet模型
    # UsePhysicsConstraint parameter has been removed

# 打印配置信息
print(f"[DEBUG] 模型名称: {ModelName}")
print(f"[DEBUG] 输入通道: {Inchannels}")
print(f"[DEBUG] 设备: {Device}")
print(f"[DEBUG] 批大小: {BatchSize}")
print(f"[DEBUG] 学习率: {LearnRate}")
print(f"[DEBUG] 训练轮数: {Epochs}")

# 确保模型保存目录存在
if not os.path.exists(ModelsDir):
    os.makedirs(ModelsDir)
    print(f'[MT_TRAIN] 已创建模型保存目录: {ModelsDir}')
else:
    print(f'[MT_TRAIN] 模型保存目录已存在: {ModelsDir}')

# 确保结果保存目录存在
if not os.path.exists(ResultsDir):
    os.makedirs(ResultsDir)
    print(f'[MT_TRAIN] 已创建结果保存目录: {ResultsDir}')
else:
    print(f'[MT_TRAIN] 结果保存目录已存在: {ResultsDir}')

# 定义训练和结果目录
models_dir = ModelsDir  # 使用之前定义的ModelsDir
results_dir = ResultsDir  # 使用之前定义的ResultsDir
print(f'[DEBUG] 模型保存目录: {models_dir}')
print(f'[DEBUG] 结果保存目录: {results_dir}')
from math import log
import subprocess
import os
import sys
import argparse

# 解析命令行参数
# def parse_args():
#     parser = argparse.ArgumentParser(description='MT Training')
#     parser.add_argument('--input_files', nargs='+', default=None, help='List of input data files')
#     parser.add_argument('--label_files', nargs='+', default=None, help='List of label data files')
#     return parser.parse_known_args()

# # 获取命令行参数
# args, unknown = parse_args()
# # input_files = args.input_files
# # label_files = args.label_files

# # 打印传递的参数，便于调试
# print(f"Command line arguments received:")
# # print(f"- input_files: {input_files}")
# # print(f"- label_files: {label_files}")
# print(f"- unknown args: {unknown}")

# 验证文件路径是否存在
# if input_files:
#     print(f"Number of input file groups: {len(input_files) if isinstance(input_files, list) else 1}")
# if label_files:
#     print(f"Number of label files: {len(label_files) if isinstance(label_files, list) else 1}")
################################################
########             NETWORK            ########
################################################

# 导入用于数据分割的库
from sklearn.model_selection import train_test_split
# Here indicating the GPU you want to use. if you don't have GPU, just leave it.


cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
# device = 'cpu'

# 根据ModelName参数动态选择和创建模型
if ModelName == 'UnetModel':
    net = UnetModel(n_classes=Nclasses, in_channels=Inchannels)
elif ModelName == 'DinkNet':
    net = DinkNet50(num_classes=Nclasses, num_channels=Inchannels)
# elif ModelName == 'DUNet':
#     net = DUNet(in_channels=Inchannels, n_classes=Nclasses)
# elif ModelName == 'ENet':
#     net = ENet(num_classes=Nclasses, in_channels=Inchannels)
# 模型初始化代码
else:
    # 默认使用UnetModel
    print(f"[MT_TRAIN WARNING] Unknown model: {ModelName}, using UnetModel as default")
    net = UnetModel(n_classes=Nclasses, in_channels=Inchannels)

print(f"[MT_TRAIN] Using model: {ModelName}")

if torch.cuda.is_available():
    net.cuda()

# Optimizer we want to use
# 使用默认的Adam优化器
optimizer = torch.optim.Adam(net.parameters(), lr=LearnRate)

# If ReUse, it will load saved model from premodelfilepath and continue to train
if ReUse:
    print('***************** Loading the pre-trained model *****************')
    print('')
    
    # 检查premodelname是否为空
    if not premodelname:
        print(f'[MT_TRAIN INFO] premodelname is empty, skipping pre-trained model loading')
        ReUse = False
    else:
        premodel_file = models_dir + premodelname + '.pkl'
        try:
            if os.path.exists(premodel_file):
                ##Load generator parameters
                net.load_state_dict(torch.load(premodel_file))
                net = net.to(device)
                print(f'[MT_TRAIN] Successfully loaded pre-trained model: {str(premodel_file)}')
            else:
                print(f'[MT_TRAIN WARNING] Pre-trained model file not found: {str(premodel_file)}')
                ReUse = False
        except Exception as e:
            print(f'[MT_TRAIN ERROR] Failed to load pre-trained model: {str(e)}')
            ReUse = False

################################################
########    LOADING TRAINING DATA       ########
################################################
print('***************** Loading Training DataSet *****************')
# 将ParamConfig中的TrainSize比例转换为实际文件个数
import os

# 尝试从实际数据目录中统计文件数量
# 根据MT_Mode选择合适的目录来统计文件数
try:
    if MT_Mode == 'TE' and os.path.exists(TE_Resistivity_Dir):
        # 统计TE模式下的文件数量
        total_files = len([f for f in os.listdir(TE_Resistivity_Dir) if f.endswith('.txt') and f[:-4].isdigit()])
    elif MT_Mode == 'TM' and os.path.exists(TM_Resistivity_Dir):
        # 统计TM模式下的文件数量
        total_files = len([f for f in os.listdir(TM_Resistivity_Dir) if f.endswith('.txt') and f[:-4].isdigit()])
    elif MT_Mode == 'Both':
        # 对于Both模式，取两个目录中文件数量较小的值
        te_files = len([f for f in os.listdir(TE_Resistivity_Dir) if f.endswith('.txt') and f[:-4].isdigit()]) if os.path.exists(TE_Resistivity_Dir) else 0
        tm_files = len([f for f in os.listdir(TM_Resistivity_Dir) if f.endswith('.txt') and f[:-4].isdigit()]) if os.path.exists(TM_Resistivity_Dir) else 0
        total_files = min(te_files, tm_files)
    else:
        # 如果目录不存在或模式不正确，使用默认值
        total_files = 1000
        print(f'[MT_TRAIN WARNING] 无法访问数据目录或MT_Mode设置不正确，使用默认文件数: {total_files}')
        print(f'[MT_TRAIN] 当前MT_Mode: {MT_Mode}')
        print(f'[MT_TRAIN] TE_Resistivity_Dir: {TE_Resistivity_Dir} 存在: {os.path.exists(TE_Resistivity_Dir)}')
        print(f'[MT_TRAIN] TM_Resistivity_Dir: {TM_Resistivity_Dir} 存在: {os.path.exists(TM_Resistivity_Dir)}')
except Exception as e:
    total_files = 1000
    print(f'[MT_TRAIN ERROR] 统计文件数量时出错: {str(e)}，使用默认文件数: {total_files}')

# 根据TrainSize比例计算实际训练文件个数
# 使用round函数来处理浮点数精度问题
print(f'[MT_TRAIN] 总文件数: {total_files}, TrainSize比例: {TrainSize}, 计算前: {total_files * TrainSize}')
train_size_files = round(total_files * TrainSize)
# 确保至少有一个训练文件
if train_size_files < 1:
    train_size_files = 1
print(f'[MT_TRAIN] 计算得到训练文件个数: {train_size_files}')
print(f'[MT_TRAIN] 实际使用的训练比例: {train_size_files/total_files:.6f}')

# 对于测试或调试目的，限制最大文件数量
max_train_files = 1000000  # 可以根据需要调整这个值
if train_size_files > max_train_files:
    print(f'[MT_TRAIN WARNING] 限制训练文件数量为 {max_train_files} 进行快速测试')
    train_size_files = max_train_files
train_set, label_set, data_dsp_dim, label_dsp_dim, valid_count = DataLoad_Train(train_size=train_size_files, train_data_dir=train_data_dir, \
                                                                 data_dim=DataDim, in_channels=Inchannels, \
                                                                 model_dim=ModelDim, data_dsp_blk=data_dsp_blk, \
                                                                 label_dsp_blk=label_dsp_blk, start=1, \
                                                                 datafilename=datafilename, dataname=dataname,\
                                                                 truthfilename=truthfilename, truthname=truthname,\
                                                                TE_Resistivity_Dir=TE_Resistivity_Dir,TE_Phase_Dir=TE_Phase_Dir,\
                                                                TM_Resistivity_Dir=TM_Resistivity_Dir,TM_Phase_Dir=TM_Phase_Dir,\
                                                                Resistivity_Model_Dir=Resistivity_Model_Dir, MT_Mode=MT_Mode)
print(f'有效的训练数据个数: {valid_count}')

# 分割训练集和验证集
print(f'[MT_TRAIN] 准备分割训练集和验证集...')
# 使用ValSize参数来计算验证集比例
# 确保导入ValSize参数
if 'ValSize' not in dir():
    ValSize = 0.2  # 默认值
print(f'[MT_TRAIN] 训练集大小参数(TrainSize): {round(TrainSize, 2):.2f}')
print(f'[MT_TRAIN] 验证集大小参数(ValSize): {round(ValSize, 2):.2f}')

# 是的，这个比例可以任意设置！现在我们直接使用ValSize参数作为验证集占已加载训练数据的比例
# 这样用户可以通过ParamConfig.py中的ValSize参数任意设置验证集比例
val_ratio = ValSize  # 直接使用ValSize参数作为验证集比例
print(f'[MT_TRAIN] 验证集比例(相对于已加载的训练数据): {val_ratio:.2f}')

# 添加合理性检查
if val_ratio <= 0 or val_ratio >= 1:
    print(f'[MT_TRAIN WARNING] 验证集比例 {val_ratio} 不合理，将调整为0.2')
    val_ratio = 0.2
    
# 确保有足够的数据进行分割
min_val_samples = 5  # 最小验证样本数
min_val_ratio = min_val_samples / valid_count if valid_count > 0 else 0.01
if valid_count <= 10:  # 如果有效数据太少
    print(f'[MT_TRAIN WARNING] 有效数据量太少: {valid_count}，调整验证集比例')
    val_ratio = min(val_ratio, 0.5)  # 最多使用一半数据作为验证集
    if val_ratio < min_val_ratio:
        val_ratio = min_val_ratio
        print(f'[MT_TRAIN] 调整验证集比例为: {val_ratio:.6f}，确保至少有{min_val_samples}个验证样本')

# 分割数据
X_train, X_val, y_train, y_val = train_test_split(
    train_set, label_set, test_size=val_ratio, random_state=42, shuffle=True
)
print(f'[MT_TRAIN] 训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}')
print(f'[MT_TRAIN] 验证集实际占总训练数据比例: {len(X_val)/(len(X_train)+len(X_val)):.2f}')

# Change data type (numpy --> tensor)
# 处理多通道数据，并确保数据类型为float32以匹配模型期望
# 数据预处理代码
# 原始处理方式
if len(train_set.shape) == 4 and train_set.shape[1] > 1:
    # 多通道情况，数据形状为 [样本数, 通道数, 高度, 宽度]
    train_dataset = data_utils.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_dataset = data_utils.TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
else:
    # 单通道情况，添加通道维度
    train_dataset = data_utils.TensorDataset(torch.from_numpy(X_train).unsqueeze(1).float(), torch.from_numpy(y_train).unsqueeze(1).float())
    val_dataset = data_utils.TensorDataset(torch.from_numpy(X_val).unsqueeze(1).float(), torch.from_numpy(y_val).unsqueeze(1).float())
    
train_loader = data_utils.DataLoader(train_dataset, batch_size=BatchSize, shuffle=True)
val_loader = data_utils.DataLoader(val_dataset, batch_size=BatchSize, shuffle=False)

################################################
########            TRAINING            ########
################################################

print()
print('*******************************************')
print('*******************************************')
print('           START TRAINING                  ')
print('*******************************************')
print('*******************************************')
print()

# 初始化损失记录变量
loss3 = []
val_losses = []

print(f'[MT_TRAIN CONFIG] Original data dimention:{str(DataDim)}')
print(f'[MT_TRAIN CONFIG] Downsampled data dimention:{str(data_dsp_dim)}')
print(f'[MT_TRAIN CONFIG] Original label dimention:{str(ModelDim)}')
print(f'[MT_TRAIN CONFIG] Downsampled label dimention:{str(label_dsp_dim)}')
print(f'[MT_TRAIN CONFIG] Training size:{train_size_files}')
print(f'[MT_TRAIN CONFIG] Traning batch size:{int(BatchSize)}')
print(f'[MT_TRAIN CONFIG] Number of epochs:{int(Epochs)}')
print(f'[MT_TRAIN CONFIG] Learning rate:{float(LearnRate):.5f}')
# Physics constraint configuration has been removed
print(f'[MT_TRAIN CONFIG] Number of training samples: {len(train_loader.dataset)}')

# Initialization
# loss3 = []  # 已在上面初始化，删除重复初始化
# val_losses = []  # 已在上面初始化，删除重复初始化
# 正确计算每轮的步数：训练集样本数除以批次大小
step = max(1, len(train_loader))  # 确保step至少为1
print(f"[MT_TRAIN] 每轮训练步数: {step}")
start = time.time()

# 设置学习率调度器
# 简化为不使用调度器
print(f"[MT_TRAIN] Not using learning rate scheduler for simplicity")
scheduler = None

# 记录训练开始时间
start_training_time = time.time()

# 初始化最佳模型参数
best_val_loss = float('inf')
best_model_path = None
best_epoch = 0

print('[DEBUG TRAIN] 即将开始epoch循环，总epoch数:', Epochs)
for epoch in range(Epochs):
    print(f'[DEBUG EPOCH] 真正进入epoch循环: {epoch+1}/{Epochs}')
    print(f'[MT_TRAIN EPOCH] Starting epoch {epoch+1}/{Epochs}')
    epoch_loss = 0.0
    since = time.time()
    for i, (images, labels) in enumerate(train_loader):
        print(f'[MT_TRAIN BATCH] Processing batch {i+1}/{len(train_loader)}')
        iteration = epoch * step + i + 1
        # Set Net with train condition
        net.train()

        # BatchSize1 = int(images.shape[0])
        # 获取实际批次大小，解决最后一个批次可能不足BatchSize的问题
        actual_batch_size = images.size(0)
        # Reshape data size using actual batch size
        images = images.view(actual_batch_size, Inchannels, label_dsp_dim[0], label_dsp_dim[1])
        labels = labels.view(actual_batch_size, Nclasses, label_dsp_dim[0], label_dsp_dim[1])
        images = images.to(device)
        labels = labels.to(device)

        # Zero the gradient buffer
        optimizer.zero_grad()

        # Forward prediction
        if ModelName == 'UnetModel':
            outputs = net(images, label_dsp_dim)
        else:
            outputs = net(images)
        # Calculate the MSE
        # loss = F.mse_loss(outputs, labels, reduction='sum') / (label_dsp_dim[0] * label_dsp_dim[1] * BatchSize)
        loss1 = F.mse_loss(outputs, labels, reduction='sum') / (label_dsp_dim[0] * label_dsp_dim[1] * actual_batch_size)
        
       
        
        # 直接使用数据损失作为最终损失
        loss = loss1

        # 初始化混合精度训练相关变量（如果尚未定义）
        if 'use_amp' not in locals():
            use_amp = False
            scaler = None
        
        # 检查和修复NaN或Inf值
        if torch.isnan(images).any() or torch.isinf(images).any():
            print(f'[CRITICAL] 输入数据包含NaN或Inf值，尝试修复')
            images = torch.nan_to_num(images, nan=0.0, posinf=1e3, neginf=-1e3)
        
        # 标准前向传播
        try:
            if ModelName == 'UnetModel':
                outputs = net(images, label_dsp_dim)
            else:
                outputs = net(images)
            
            # 检查输出是否包含NaN或Inf
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f'[CRITICAL] 模型输出包含NaN或Inf值，尝试修复')
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e-1, neginf=-1e-1)
            
            # 计算损失
            loss1 = F.mse_loss(outputs, labels, reduction='sum') / (label_dsp_dim[0] * label_dsp_dim[1] * actual_batch_size)
            loss = loss1
        except Exception as e:
            print(f'[EXCEPTION] 前向传播异常: {str(e)}，跳过此批次')
            optimizer.zero_grad()
            continue
        
        # 检查损失值
        if np.isnan(float(loss.item())) or np.isinf(float(loss.item())):
            print(f'[CRITICAL ERROR] 检测到NaN/Inf损失值: {loss.item()}')
            optimizer.zero_grad()
            continue
        
        epoch_loss += loss.item()
        
        try:
            # 标准反向传播
            loss.backward()
        except Exception as e:
            print(f'[EXCEPTION] 反向传播异常: {str(e)}，跳过此批次')
            optimizer.zero_grad()
            continue
        
        # 应用梯度裁剪
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)
        
        # 保存训练日志和模型
        if (i+1) % 1 == 0:  # 每个批次都打印，方便调试
            print(f'[MT_TRAIN] Epoch {epoch+1}/{Epochs}, Batch {i+1}/{step}, Loss: {loss.item():.6f}')
            # 打印输入和输出的形状信息
            print(f'  Input shape: {images.shape}, Output shape: {outputs.shape}, Label shape: {labels.shape}')
        # 物理约束已经完全移除，不再需要else分支
        
        # 梯度检查代码
        
        # 确保反向传播成功
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f'[CRITICAL ERROR] 损失值包含NaN或Inf: {loss.item()}')
            raise ValueError('Loss contains NaN or Inf')

        # 优化器更新
        optimizer.step()
        # 验证优化器步骤后的参数
        with torch.no_grad():
            param_check = next(net.parameters())
            if torch.isnan(param_check).any():
                print('[CRITICAL ERROR] 优化器更新后参数包含NaN')
                raise ValueError('Parameters contain NaN after optimization')
        
        # 学习率调度器逻辑
        if scheduler is not None:
            scheduler.step()
            
            # 打印当前学习率以便监控
            current_lr = scheduler.get_last_lr()[0]
            if epoch % 10 == 0 and (i+1) % 20 == 0:
                print(f"[MT_TRAIN] Current learning rate: {current_lr:.8f}")

        # Print loss
        if iteration % DisplayStep == 0:
            print('Epoch: {}/{}, Iteration: {}/{} --- Training Loss:{:.8f}'.format(epoch + 1, \
                                                                                    Epochs, iteration, \
                                                                                    step * Epochs, loss.item()))
        
    # Epoch完成后打印信息 - 已移到batch循环外部
    print('[DEBUG TRAIN] Batch循环已完成，开始处理epoch完成逻辑')
    print('Epoch: {:d} finished ! Loss: {:.8f}'.format(epoch + 1, epoch_loss / len(train_loader)))
    # loss3 = np.append(loss3, epoch_loss / len(train_loader))  # 使用np.append会将列表转换为numpy数组
    loss3.append(epoch_loss / len(train_loader))  # 保持与val_losses一致，使用列表append
    time_elapsed = time.time() - since
    print('Epoch consuming time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    # 这个特殊格式的输出会被ml_trainer.py解析
    print(f"Epoch: {epoch+1} finished, Loss: {epoch_loss/len(train_loader):.6f}, Time: {time_elapsed:.2f}s")
    
    # 学习率调度器更新（已在step级别处理）
    
    # 进行验证 - 已移到batch循环外部
    net.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            # 获取实际批次大小
            val_batch_size = val_images.size(0)
            # Reshape data size
            val_images = val_images.view(val_batch_size, Inchannels, label_dsp_dim[0], label_dsp_dim[1])
            val_labels = val_labels.view(val_batch_size, Nclasses, label_dsp_dim[0], label_dsp_dim[1])
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            
            # Forward prediction
            if ModelName == 'UnetModel':
                val_outputs = net(val_images, label_dsp_dim)
            else:
                val_outputs = net(val_images)
            
            # Calculate the validation loss
            batch_val_loss = F.mse_loss(val_outputs, val_labels, reduction='sum') / (label_dsp_dim[0] * label_dsp_dim[1] * val_batch_size)
            val_loss += batch_val_loss.item()
    
    # 计算平均验证损失
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f'[MT_TRAIN VALIDATION] Epoch {epoch+1}/{Epochs} - Validation Loss: {avg_val_loss:.8f}')
    
    # 保存验证损失最小的模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        
        # 从LearnRate中移除小数点，避免文件名问题
        lr_str = str(LearnRate).replace('.', '')
        best_model_path = models_dir + f'{ModelName}_TrainSize{TrainSize}_Epoch{Epochs}_BatchSize{BatchSize}_LR{lr_str}_Mode{MT_Mode}_best_val_epoch{epoch + 1}.pkl'
        
        # 删除之前的最佳模型（如果存在）
        if os.path.exists(best_model_path):
            os.remove(best_model_path)
        
        # 保存新的最佳模型
        torch.save(net.state_dict(), best_model_path)
        print(f'[MT_TRAIN SAVE] 新的最佳模型已保存到 {best_model_path}, 验证损失: {best_val_loss:.8f}')
        
        # 根据用户要求，不保存归一化参数
        print(f'[MT_TRAIN SAVE] 最佳模型已保存，不使用归一化参数')
        

# Record the consuming time
time_elapsed = time.time() - start_training_time
print('[MT_TRAIN COMPLETE] Training complete in {:.0f}m  {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


# 保存最终模型
# 在文件名中包含更多模型配置信息
# 确保文件名中包含明确的ModelName和MT模式信息
# 从LearnRate中移除小数点，避免文件名问题
lr_str = str(LearnRate).replace('.', '')
final_model_path = models_dir + f'{ModelName}_TrainSize{TrainSize}_Epoch{Epochs}_BatchSize{BatchSize}_LR{lr_str}_Mode{MT_Mode}_final.pkl'
torch.save(net.state_dict(), final_model_path)
print(f'[MT_TRAIN COMPLETE] Final model saved to {final_model_path}')

# 显示最佳模型信息
print('[DEBUG TRAIN] 训练循环已完成，实际执行的epoch数:', epoch+1)
print(f'[MT_TRAIN SUMMARY] 最佳模型在第{best_epoch}轮保存，验证损失: {best_val_loss:.8f}')
if best_model_path:
    print(f'[MT_TRAIN SUMMARY] 最佳模型路径: {best_model_path}')
    # 将最佳模型复制为best.pkl，便于后续使用
    best_pkl_path = models_dir + f'{ModelName}_best.pkl'
    if os.path.exists(best_pkl_path):
        os.remove(best_pkl_path)
    import shutil
    shutil.copy(best_model_path, best_pkl_path)
    print(f'[MT_TRAIN SUMMARY] 最佳模型已复制为: {best_pkl_path}')

# 保存训练和验证损失曲线
# print('[MT_TRAIN RESULTS] 保存训练和验证损失曲线...')
# from func.utils import SaveTrainResults
# import matplotlib.pyplot as plt
# font2 = {'family': 'Times New Roman', 'size': 16}
# font3 = {'family': 'Times New Roman', 'size': 18}
# SaveTrainResults(loss3, val_losses, results_dir, font2, font3)
# print(f'[MT_TRAIN RESULTS] 损失曲线已保存到 {results_dir}LossCurve.png')

print('[MT_TRAIN EXIT] Training process completed successfully')





