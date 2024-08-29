'''
Useful tools in this project.
Including:
    PSNR calculation,
    SSIM calculation,
    parameter initialization,
    ...
Updated 2022/3/7 By Aurora Yin
'''

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity
from collections import OrderedDict
import pytorch_ssim


####################### PSNR & SSIM #########################
'''
batch_PSNR-- PSNR calculation
img_gt: ground truth image
img_test: result image out from net
data_range: always 255 or 1.
'''
def batch_PSNR(img_gt, img_test, average=False):
    # Img_gt = img_gt.data.cpu().numpy().astype(np.float32)
    # Img_test = img_test.data.cpu().numpy().astype(np.float32)
    # PSNR = 0
    # for i in range(Img_gt.shape[0]):
    #     PSNR += peak_signal_noise_ratio(Img_gt[i,:,:,:], Img_test[i,:,:,:], data_range=data_range)
    # result = PSNR/Img_gt.shape[0]   # calculate average PSNR in one batch
    # return result

    PSNR = []
    for im1, im2 in zip(img_gt, img_test):
        imdff = torch.clamp(im2, 0, 1) - torch.clamp(im1, 0, 1)
        rmse = (imdff ** 2).mean().sqrt()
        psnr = 20 * torch.log10(1 / rmse)
        PSNR.append(psnr)
    return sum(PSNR) / len(PSNR) if average else sum(PSNR)

'''
paras same as PSNR
multi_ch: if RGB then True
'''
def batch_SSIM(img_gt, img_test, multi_ch):
    Img_gt = img_gt.data.cpu().numpy().astype(np.float32)
    Img_test = img_test.data.cpu().numpy().astype(np.float32)
    SSIM = 0
    for i in range(Img_gt.shape[0]):
        SSIM += structural_similarity(Img_gt[i, :, :, :], Img_test[i, :, :, :], win_size=3, multichannel=multi_ch)
        # SSIM += structural_similarity(Img_gt, Img_test)

    result = SSIM / Img_gt.shape[0]  # calculate average SSIM in one batch
    return result

def myPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def mySSIM(tar_img, prd_img):
    return pytorch_ssim.ssim(tar_img, prd_img)


####################### Data Augmentation ########################
'''
data augmentation
'''
def data_augmentation(image, mode):
    # first transpose to standard format, [channels, height, width] => [height, width, channels]
    # then do corresponding works
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1)) # HWC => CWH


class Augment_RGB_torch:
    def __init__(self):
        pass
    def transform0(self, torch_tensor):
        return torch_tensor
    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])
        return torch_tensor
    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        return torch_tensor


######################### Loss Functions ##########################
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class MS_SSIM_L1_LOSS(nn.Module):
    """
    Have to use cuda, otherwise the speed is too slow.
    Both the group and shape of input image should be attention on.
    I set 255 and 1 for gray image as default.
    """

    def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range=255.0,
                 K=(0.01, 0.03),  # c1,c2
                 alpha=0.84,  # weight of ssim and l1 loss
                 compensation=200.0,  # final factor for total loss
                 cuda_dev=0,  # cuda device choice
                 channel=3):  # RGB image should set to 3 and Gray image should be set to 1
        super(MS_SSIM_L1_LOSS, self).__init__()
        self.channel = channel
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation = compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros(
            (self.channel * len(gaussian_sigmas), 1, filter_size, filter_size))  # 创建了(3*5, 1, 33, 33)个masks
        for idx, sigma in enumerate(gaussian_sigmas):
            if self.channel == 1:
                # only gray layer
                g_masks[idx, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            elif self.channel == 3:
                # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
                g_masks[self.channel * idx + 0, 0, :, :] = self._fspecial_gauss_2d(filter_size,
                                                                                   sigma)  # 每层mask对应不同的sigma
                g_masks[self.channel * idx + 1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
                g_masks[self.channel * idx + 2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            else:
                raise ValueError
        self.g_masks = g_masks.cuda(cuda_dev)  # 转换为cuda数据类型

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)
        # Outer product of input and vec2. If input is a vector of size nn and vec2 is a vector of size mm,
        # then out must be a matrix of size (n \times m)(n×m).

    def forward(self, x, y):
        b, c, h, w = x.shape
        assert c == self.channel

        mux = F.conv2d(x, self.g_masks, groups=c, padding=self.pad)  # 图像为96*96，和33*33卷积，出来的是64*64，加上pad=16,出来的是96*96
        muy = F.conv2d(y, self.g_masks, groups=c, padding=self.pad)  # groups 是分组卷积，为了加快卷积的速度

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=c, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=c, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=c, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l = (2 * muxy + self.C1) / (mux2 + muy2 + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)
        if self.channel == 3:
            lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]  # 亮度对比因子
            PIcs = cs.prod(dim=1)
        elif self.channel == 1:
            lM = l[:, -1, :, :]
            PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM * PIcs  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, C, H, W]
        # average l1 loss in num channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-self.channel, length=self.channel),
                               groups=c, padding=self.pad).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation * loss_mix

        return loss_mix.mean()


###################### Model Utils #########################
'''
weights_init-- Method for parameter initializations
Used in DnCNN_Pro
Use kaiming_normal initialization for Conv and Linear; 
Pay attention to the initialization of BatchNorm
'''
def weights_init(m):
    classname = m.__class__.__name__    #get name of network level, such as 'Conv2d'
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname == 'Linear':
        # nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        nn.init.constant_(m.bias.data, 0.)
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02) # Uniform distribution
        nn.init.normal_(m.weight.data, 1., 0.02)   # Gaussian distribution
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025) # DnCNN_Pro
        nn.init.constant(m.bias.data, 0.)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    torch.save(state, model_out_path)

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    # checkpoint = torch.load(weights, map_location=lambda storage, loc: storage.cuda(1))
    try:
        print("Loading weights ... ...")
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, False)

def load_checkpoint_multigpu(model, weights):
    # checkpoint = torch.load(weights, map_location=lambda storage, loc: storage.cuda(1))
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    # checkpoint = torch.load(weights, map_location=lambda storage, loc: storage.cuda(1))
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    # checkpoint = torch.load(weights, map_location=lambda storage, loc: storage.cuda(1))
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr


import csv

def create_csv(path, csv_head):
    with open(path, 'w', newline = '', encoding='utf-8_sig') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(csv_head)#写入表头

def write_csv(path, data):   #file_name为写入CSV文件的路径，datas为要写入数据列表
    # file_csv = codecs.open(path, 'w+', 'utf-8')  #追加
    with open(path, 'a', newline='', encoding='utf-8_sig') as f:
        writer = csv.writer(f)
        writer.writerow(data)
