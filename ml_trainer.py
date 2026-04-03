#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Machine Learning Trainer Module for MTDLPy.

Author: AI Assistant
Creation Time: 2024
"""

import os
import sys
import time
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 确保中文正常显示
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
matplotlib.use('Agg')  # 非交互式后端，适合后台运行
import matplotlib.pyplot as plt
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

# 定义一个函数来检查CUDA是否可用，而不是在模块导入时自动执行
def get_available_device(verbose=False):
    """Return cuda if available, else cpu."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")
    return device

# 默认设备，不自动打印
DEVICE = get_available_device(verbose=False)

class MTDataSet(Dataset):
    """MT dataset: load input/label files."""
    def __init__(self, input_files, label_files):
        self.input_files = input_files
        self.label_files = label_files
        
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        # 处理多通道输入数据
        if isinstance(self.input_files[0], list):
            # 多通道情况
            input_tensors = []
            for channel_files in self.input_files:
                input_path = channel_files[idx]
                input_data = np.loadtxt(input_path) if os.path.splitext(input_path)[1] == '.txt' else np.load(input_path)
                
                # 数据预处理
                input_tensor = torch.from_numpy(input_data).float().unsqueeze(0)
                
                input_tensors.append(input_tensor)
            
            # 沿通道维度拼接
            input_tensor = torch.cat(input_tensors, dim=0)
        else:
            # 单通道情况
            input_path = self.input_files[idx]
            input_data = np.loadtxt(input_path) if os.path.splitext(input_path)[1] == '.txt' else np.load(input_path)
            
            # 数据预处理
            input_tensor = torch.from_numpy(input_data).float().unsqueeze(0)
        
        # 加载标签数据
        label_path = self.label_files[idx]
        label_data = np.loadtxt(label_path) if os.path.splitext(label_path)[1] == '.txt' else np.load(label_path)
        
        # 数据预处理
        label_tensor = torch.from_numpy(label_data).float().unsqueeze(0)
        
        return input_tensor, label_tensor



class BaseModel(nn.Module):
    """Base model class."""
    def __init__(self, in_channels=1, out_channels=1):
        super(BaseModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(self, x):
        raise NotImplementedError("Subclass must implement forward method")

class DinkNet50(BaseModel):
    """DinkNet50 stub."""
    def __init__(self, in_channels=1, out_channels=1, num_classes=1):
        super(DinkNet50, self).__init__(in_channels, out_channels)
        self.num_classes = num_classes
        
        # 这里实现DinkNet50的网络结构
        # 为了示例，这里提供一个简化版本
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid() if out_channels == 1 else nn.Softmax(dim=1)
    
    def forward(self, x):
        # 简化版本的前向传播
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.upconv1(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x

class UNetModel(BaseModel):
    """U-Net stub."""
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetModel, self).__init__(in_channels, out_channels)
        
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 中间部分
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 解码器部分
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid() if out_channels == 1 else nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

class MTTrainer:
    """Trainer: spawns MT_train.py subprocess."""
    def __init__(self, model_name='DinkNet50', in_channels=1, out_channels=1, device=None, verbose=False):
        self.model_name = model_name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device if device is not None else DEVICE
        
        # 只有在明确要求时才打印设备信息
        if verbose:
            print(f"Using device: {self.device}")
        
        # 初始化模型
        self.model = self._create_model()
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
        # 训练参数
        self.epochs = 100
        self.batch_size = 32
        self.learning_rate = 0.001
        self.early_stop = None
        self.save_epoch = 10
        self.display_step = 10
        
        # 物理约束相关参数已移除
        
        # 训练结果
        self.total_loss = []
        self.data_loss = []
        self.physics_loss = []
        self.validation_losses = []  # 新增：保存验证损失
        
        # 状态标志
        self.stopped = False
        self.paused = False
        self.was_stopped = True  # 初始值设为True，表示训练未开始或未成功完成
        self.train_thread = None
        
        # 回调函数
        self.update_callback = None
        self.progress_callback = None
        self.loss_callback = None
        
        # 数据文件路径
        self.input_files = []
        self.label_files = []
        
        # 保存路径
        self.models_dir = "models/"
        self.results_dir = "results/"
    
    def _create_model(self):
        """Create model instance by name."""
        if self.model_name == 'DinkNet50':
            model = DinkNet50(self.in_channels, self.out_channels)
        elif self.model_name == 'UNet':
            model = UNetModel(self.in_channels, self.out_channels)
        elif self.model_name == 'UnetPlusPlus':
            from func.unetplusplus import UNetPlusPlus
            model = UNetPlusPlus(num_classes=self.out_channels, num_channels=self.in_channels)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        
        return model.to(self.device)
    
    def set_params(self, epochs=None, batch_size=None, learning_rate=None, early_stop=None, save_epoch=None, display_step=None,
                   
                   input_files=None, label_files=None, models_dir=None, results_dir=None):
        """Set training hyperparameters."""
        if epochs is not None:
            self.epochs = epochs
        if batch_size is not None:
            self.batch_size = batch_size
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if early_stop is not None:
            self.early_stop = early_stop
        if save_epoch is not None:
            self.save_epoch = save_epoch
        if display_step is not None:
            self.display_step = display_step
        
        # 物理约束相关参数已移除
        
        # 数据文件路径
        if input_files is not None:
            self.input_files = input_files
        if label_files is not None:
            self.label_files = label_files
        
        # 保存路径
        if models_dir is not None:
            self.models_dir = models_dir
        if results_dir is not None:
            self.results_dir = results_dir
    
    def set_callbacks(self, update_callback=None, progress_callback=None, loss_callback=None):
        """Register UI callbacks."""
        self.update_callback = update_callback
        self.progress_callback = progress_callback
        self.loss_callback = loss_callback
    
    def _log(self, message):
        """Append log line."""
        if self.update_callback:
            self.update_callback(message)
        print(message)
    
    def load_data(self):
        """Load training data into DataLoader."""
        try:
            # 检查是否有输入文件和标签文件
            if not self.input_files or not self.label_files:
                raise ValueError("No input files or label files specified")
            
            # 检查文件数量是否匹配
            if isinstance(self.input_files[0], list):
                # 多通道情况，每个通道都应该有相同数量的文件
                for i, channel_files in enumerate(self.input_files):
                    if len(channel_files) != len(self.label_files):
                        raise ValueError(f"Channel {i} files count ({len(channel_files)}) doesn't match label files count ({len(self.label_files)})")
            else:
                # 单通道情况
                if len(self.input_files) != len(self.label_files):
                    raise ValueError(f"Input files count ({len(self.input_files)}) doesn't match label files count ({len(self.label_files)})")
            
            # 创建数据集
            self.dataset = MTDataSet(self.input_files, self.label_files)
            self.dataloader = DataLoader(
                self.dataset, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=0  # 在GUI环境中，建议设置为0以避免多进程问题
            )
            
            # 初始化优化器
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            self._log(f"Successfully loaded {len(self.input_files)} pairs of training data")
            return True
        except Exception as e:
            self._log(f"Error loading data: {str(e)}")
            return False
    
    def load_pretrained_model(self, model_path):
        """Load pretrained weights if path given."""
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self._log(f"Pretrained model loaded successfully from {model_path}")
                return True
            except Exception as e:
                self._log(f"Error loading pretrained model: {str(e)}")
                return False
        else:
            self._log(f"Pretrained model file not found: {model_path}")
            return False
    
    # 物理约束损失方法已移除
    
    def save_model(self, save_path, epoch=None):
        """Save state_dict to disk."""
        try:
            # 确保保存目录存在
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # 保存模型状态字典
            torch.save(self.model.state_dict(), save_path)
            
            if epoch is not None:
                self._log(f"Model saved at epoch {epoch}: {save_path}")
            else:
                self._log(f"Model saved: {save_path}")
            
            return True
        except Exception as e:
            self._log(f"Error saving model: {str(e)}")
            return False
    
    def _train_in_thread(self):
        """Run MT_train.py in a background thread."""
        try:
            import subprocess
            import sys
            import time
            
            # 确保模型保存目录存在
            if not os.path.exists(self.models_dir):
                os.makedirs(self.models_dir)
                self._log(f"Created models directory: {self.models_dir}")
            else:
                self._log(f"Models directory exists: {self.models_dir}")
            
            # 获取MT_train.py的绝对路径
            mt_train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MT_train.py')
            
            self._log(f"Starting MT_train.py")
            self._log(f"MT_train.py path: {mt_train_path}")
            
            # 检查MT_train.py文件是否存在
            if not os.path.exists(mt_train_path):
                self._log(f"Error: MT_train.py not found at {mt_train_path}")
                return
            
            # 构建命令行参数
            cmd_args = [sys.executable, mt_train_path]
            
            # 添加输入和标签文件参数（如果有）
            if self.input_files:
                cmd_args.extend(['--input_files'])
                cmd_args.extend(self.input_files)
                self._log(f"Added {len(self.input_files)} input files")
                
            if self.label_files:
                cmd_args.extend(['--label_files'])
                cmd_args.extend(self.label_files)
                self._log(f"Added {len(self.label_files)} label files")
            
            self._log(f"Command: {' '.join(cmd_args)}")
            
            # 启动MT_train.py进程
            try:
                process = subprocess.Popen(
                    cmd_args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # 将stderr重定向到stdout
                    text=True,
                    bufsize=1,
                    shell=False
                )
                self._log(f"MT_train.py started, PID: {process.pid}")
            except Exception as proc_error:
                self._log(f"Failed to start MT_train.py: {str(proc_error)}")
                return
            
            # 实时读取和处理输出
            start_time = time.time()
            epoch = 0
            training_completed = False
            output_received = False
            last_output_time = time.time()
            
            # 添加已处理epoch的集合，确保每个epoch只记录一次损失
            processed_epochs = set()
            
            # 重置损失列表，用于GUI显示
            self.total_loss = []
            self.data_loss = []
            self.physics_loss = []
            
            # 设置非阻塞读取模式
            process.stdout._line_buffering = True
            
            while process.poll() is None:
                # 检查是否停止训练
                if self.stopped:
                    self.was_stopped = True
                    self._log("Stopping training...")
                    try:
                        process.terminate()
                        process.wait(timeout=5)
                    except:
                        try:
                            process.kill()
                        except:
                            pass
                    self._log("Training stopped")
                    return
                
                # 检查是否暂停训练
                while self.paused:
                    if self.stopped:
                        self.was_stopped = True
                        self._log("Stopping training...")
                        try:
                            process.terminate()
                            process.wait(timeout=5)
                        except:
                            try:
                                process.kill()
                            except:
                                pass
                        self._log("Training stopped")
                        return
                    time.sleep(0.5)
                
                # 非阻塞读取输出
                import select
                ready_to_read, _, _ = select.select([process.stdout], [], [], 0.1)  # 0.1秒超时
                
                if ready_to_read:
                    line = process.stdout.readline()
                    if line:
                        output_received = True
                        last_output_time = time.time()
                        line = line.strip()
                        
                        # 在单独的线程中更新GUI，避免阻塞训练线程
                        def update_gui():
                            self._log(line)
                            
                            # 检测训练完成标志
                            if "Training complete" in line or "Final model saved" in line or "[MT_TRAIN EXIT]" in line:
                                nonlocal training_completed
                                training_completed = True
                            
                            # 解析epoch信息以更新进度条和损失曲线
                            # 只处理MT_train.py中专门设计的特殊格式输出行
                            if "Epoch:" in line and "finished, Loss:" in line and "Loss1:" in line and "Loss2:" in line and "Time:" in line:
                                try:
                                    # 这是MT_train.py中专门为解析设计的格式: "Epoch: X finished, Loss: Y, Loss1: Z, Loss2: W, Time: T"
                                    # 提取epoch信息
                                    # 使用更精确的分割方式，确保正确解析epoch号
                                    try:
                                        # 先找到'Epoch: '后面的数字部分
                                        epoch_start = line.find('Epoch: ') + len('Epoch: ')
                                        epoch_end = line.find(' finished')
                                        if epoch_start > 0 and epoch_end > epoch_start:
                                            epoch_part = line[epoch_start:epoch_end].strip()
                                            if epoch_part.isdigit():
                                                nonlocal epoch
                                                epoch = int(epoch_part)
                                                
                                                # 更新进度条
                                                if self.progress_callback and self.epochs > 0:
                                                    progress = int((epoch / self.epochs) * 100)
                                                    self.progress_callback(min(progress, 100))
                                    except Exception as e:
                                        self._log(f"Epoch parse error: {str(e)}")
                                
                                    try:
                                        # 提取总损失值
                                        loss_part = line.split("Loss:")[1].split(",")[0].strip()
                                        total_loss = float(loss_part)
                                    
                                        # 确保只在epoch结束时记录一次损失值
                                        # 使用集合来确保每个epoch只处理一次
                                        if epoch not in processed_epochs:
                                            # 这是新的epoch，添加到已处理集合
                                            processed_epochs.add(epoch)
                                            
                                            # 将损失值添加到列表中
                                            self.total_loss.append(total_loss)
                                            
                                            # 清空其他损失列表以节省内存
                                            self.data_loss = []
                                            self.physics_loss = []
                                            
                                            # 立即更新图表，使用最新的损失列表
                                            if self.loss_callback:
                                                # 传递训练损失和验证损失
                                                self.loss_callback(self.total_loss, self.validation_losses, [])
                                    except Exception as parse_error:
                                        self._log(f"Loss parse error: {str(parse_error)}")
                                except Exception as general_error:
                                    self._log(f"Train line handler error: {str(general_error)}")
                            
                            # 解析验证损失信息 - 独立检查每一行
                            if '[MT_TRAIN VALIDATION]' in line:
                                try:
                                    # 提取验证损失值
                                    val_loss_str = line.split('Validation Loss:')[1].strip()
                                    val_loss = float(val_loss_str)
                                    
                                    # 添加到验证损失列表
                                    self.validation_losses.append(val_loss)
                                    
                                    # 立即更新图表，使用最新的损失列表
                                    if self.loss_callback:
                                        self.loss_callback(self.total_loss, self.validation_losses, [])
                                except (IndexError, ValueError):
                                    self._log(f"Val loss parse error: {line}")
                                except Exception as val_error:
                                    self._log(f"Val line handler error: {str(val_error)}")
                        
                        # 创建并启动GUI更新线程
                        gui_thread = threading.Thread(target=update_gui)
                        gui_thread.daemon = True
                        gui_thread.start()
                
                # 检查是否长时间没有输出
                current_time = time.time()
                if output_received and current_time - last_output_time > 10:
                    def log_warning():
                        self._log("Warning: no output for a while; check MT_train.py")
                    warning_thread = threading.Thread(target=log_warning)
                    warning_thread.daemon = True
                    warning_thread.start()
                    last_output_time = current_time
                elif not output_received and current_time - start_time > 10:
                    def log_startup_warning():
                        self._log("Warning: no output from MT_train.py within 10s of start")
                        self._log("Check that MT_train.py runs correctly")
                    startup_warning_thread = threading.Thread(target=log_startup_warning)
                    startup_warning_thread.daemon = True
                    startup_warning_thread.start()
                    output_received = True  # 只显示一次警告
                
                # 短暂休眠，避免CPU占用过高
                time.sleep(0.01)
            
            # 训练完成处理
            total_time_elapsed = time.time() - start_time
            self.was_stopped = False  # 无论返回代码如何，都设置为训练完成
            self._log(f"Training process ended, elapsed {total_time_elapsed // 60:.0f}m {total_time_elapsed % 60:.0f}s")
            self._log(f"MT_train.py exit code: {process.returncode}")
            
            # 检查是否收到任何输出
            if not output_received:
                self._log("Warning: no stdout from MT_train.py during run")
                self._log("Possible causes:")
                self._log("1. Wrong path to MT_train.py")
                self._log("2. Wrong Python interpreter")
                self._log("3. MT_train.py crashed early")
                self._log("4. Invalid CLI arguments")
            
            # 只在真正训练完成时才设置进度条为100%
            if self.progress_callback and training_completed:
                self.progress_callback(100)
            
        except Exception as e:
            self._log(f"Error launching MT_train.py: {str(e)}")
            self._log("Check MT_train.py exists and is executable")
            import traceback
            self._log(f"Traceback: {traceback.format_exc()}")
            self.was_stopped = False
    
    def train(self):
        """Start MT_train.py on a worker thread."""
        # 如果已经在训练中，不重复启动
        if self.train_thread and self.train_thread.is_alive():
            self._log("Training already in progress")
            return False
            
        # 重置状态标志
        self.stopped = False
        self.paused = False
        self.was_stopped = True
        
        # 创建并启动训练线程
        self.train_thread = threading.Thread(target=self._train_in_thread)
        self.train_thread.daemon = True  # 设置为守护线程，主线程结束时自动终止
        self.train_thread.start()
        
        self._log("Training thread started")
        return True
    
    def pause_train(self):
        """Pause training."""
        if self.train_thread and self.train_thread.is_alive() and not self.paused:
            self.paused = True
            self._log("Training paused")
            return True
        return False
    
    def resume_train(self):
        """Resume training."""
        if self.paused:
            self.paused = False
            self._log("Training resumed")
            return True
        return False
    
    def stop_train(self):
        """Stop training."""
        if self.train_thread and self.train_thread.is_alive():
            self.stopped = True
            # 设置paused为False，确保训练线程可以继续执行并检测到stopped标志
            self.paused = False
            # 等待训练线程结束，最多等待10秒
            self.train_thread.join(timeout=10)
            self._log("Training stopped")
            return True
        return False
    
    def is_training(self):
        """Return True if worker thread is alive."""
        return self.train_thread is not None and self.train_thread.is_alive()

    
    def pause(self):
        """Set paused flag."""
        self.paused = True
    
    def resume(self):
        """Clear paused flag."""
        self.paused = False
    
    def stop(self):
        """Request stop."""
        self.stopped = True
        self.paused = False

# 用于测试的主函数
if __name__ == "__main__":
    # 这里可以添加测试代码
    print("ML Trainer Module Loaded Successfully")