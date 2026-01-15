# -*- coding: utf-8 -*-
"""


创建于2021年7月

作者：袁崇鑫

"""
import torch
import numpy as np
import torch.nn as nn
from math import log10
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import scipy.io
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
import csv
import cv2

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list


def turn(GT):
    dim = GT.shape
    for j in range(0,dim[1]):
        for i in range(0,dim[0]//2):
            temp    = GT[i,j]
            GT[i,j] = GT[dim[0]-1-i,j]
            GT[dim[0]-1-i,j] = temp
    return GT 


def PSNR(prediction, target):
    prediction = Variable(torch.from_numpy(prediction))
    target     = Variable(torch.from_numpy(target))
    zero       = torch.zeros_like(target)   
    criterion  = nn.MSELoss(size_average=True)    
    MSE        = criterion (prediction, target)
    total      = criterion (target, zero)
    psnr       = 10. * log10(total.item() / MSE.item())
    return psnr

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window     = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1    = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2    = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    L  = 255
    C1 = (0.01*L) ** 2
    C2 = (0.03*L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)



def SSIM(img1, img2, window_size=11, size_average=True):
    img1 = Variable(torch.from_numpy(img1))
    img2 = Variable(torch.from_numpy(img2))
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def SaveTrainResults(train_loss, val_loss, SavePath, font2, font3):
    """保存训练结果：仅保存损失数值到txt文件，不再单独保存损失函数曲线"""
    # 保存训练和验证损失到txt文件
    np.savetxt(SavePath + 'train_loss.txt', train_loss)
    np.savetxt(SavePath + 'val_loss.txt', val_loss)

def SaveTestResults(TotPSNR,TotSSIM,Prediction,GT,SavePath,Testsize):
    data = {}
    data['TotPSNR'] = TotPSNR
    data['TotSSIM'] = TotSSIM
    data['GT']      = GT
    data['Prediction'] = Prediction
    for i in range(0, Testsize):
        np.savetxt(SavePath + f'prediction_{i+1}.txt', Prediction[i])

    # 可选：保存完整的测试结果字典
    # save_dict(SavePath + 'TestResults', data)



def save_dict(filename, dic):
        '''save dict into json file'''
        with open(filename, 'w') as json_file:
            json.dump(dic, json_file, ensure_ascii=False, cls=JsonEncoder)


    #scipy.io.savetxt(SavePath+'TestResults',data)
    
    
def PlotComparison(pd,gt,label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath):
    PD = pd.reshape(label_dsp_dim[0],label_dsp_dim[1])
    GT = gt.reshape(label_dsp_dim[0],label_dsp_dim[1])
    fig1,ax1 = plt.subplots(figsize=(6, 4))
    im1 = ax1.imshow(GT, cmap = 'rainbow', vmin=minvalue, vmax=maxvalue)
    #im1     = ax1.imshow(GT,extent=[0,label_dsp_dim[1]*label_dsp_blk[1]*dh/1000., \
                             # 0,label_dsp_dim[0]*label_dsp_blk[0]*dh/1000.],vmin=minvalue,vmax=maxvalue)
    divider = make_axes_locatable(ax1)
    cax1    = divider.append_axes("right",size="5%",pad=0.05)
    plt.colorbar(im1,ax=ax1,cax=cax1).set_label('lgρ(Ω·m)')
    plt.tick_params(labelsize=12)
    for label in  ax1.get_xticklabels()+ax1.get_yticklabels():
        label.set_fontsize(14)
    ax1.set_xlabel('Position (km)',font2)
    ax1.set_ylabel('Depth (km)',font2)
    ax1.set_title('Ground truth',font3)
    ax1.invert_yaxis()
    plt.subplots_adjust(bottom=0.15,top=0.92,left=0.08,right=0.98)
    plt.savefig(SavePath+'GT',transparent=True)
    
    fig2,ax2=plt.subplots(figsize=(6, 4))

    im2 = ax2.imshow(PD, cmap = 'rainbow', vmin=minvalue, vmax=maxvalue)
    #im2=ax2.imshow(PD,extent=[0,label_dsp_dim[1]*label_dsp_blk[1]*dh/1000., \
                              #0,label_dsp_dim[0]*label_dsp_blk[0]*dh/1000.],vmin=minvalue,vmax=maxvalue)

    plt.tick_params(labelsize=12)  
    for label in  ax2.get_xticklabels()+ax2.get_yticklabels():
        label.set_fontsize(14)   
    ax2.set_xlabel('Position (km)',font2)
    ax2.set_ylabel('Depth (km)',font2)
    ax2.set_title('Prediction',font3)
    ax2.invert_yaxis()
    plt.subplots_adjust(bottom=0.15,top=0.92,left=0.08,right=0.98)
    plt.savefig(SavePath+'PD',transparent=True)
    #plt.show(fig1)
   # plt.show(fig2)
    plt.close()
   
