# '''
# Testing model
#     Save test results to corresponding folder.
#
# Updated 2022/3/13 By Aurora Yin
# '''
# import os
# import time
# import argparse
# import glob
#
# import matplotlib.pyplot as plt
# import numpy
# import numpy as np
# import cv2
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# from torchvision.utils import save_image
# import torchvision.transforms as tf
#
# from model import *
# from utils import *
# from dataPro import *
#
# def main():
#     channel = 3
#     layers_num = 20
#     load_fp = '../net_paras_1.pth'
#     input_path = '../dataset/SIDD_Small_sRGB_Only/Data/'
#     result_path = '../Test_result/'
#
#     # ImageNames = os.listdir(input_path)
#
#     noise_level = 25
#     psnr_test = 0
#     ssim_test = 0
#
#     print("Test start \nLoading model ...")
#     torch.cuda.empty_cache()
#     net = DnCNN_Pro(channels=channel, num_of_layers=layers_num)
#     device_ids = [i for i in range(torch.cuda.device_count())]
#     if len(device_ids) >= 1:
#         model = nn.DataParallel(net, device_ids=device_ids).cuda()
#
#     model = nn.DataParallel(net).cuda()
#     model.load_state_dict(torch.load(load_fp))
#     model.eval()
#
#     ## Prepare test data ##
#     # files = glob.glob(os.path.join(input_path, '*'))
#     all_files = glob.glob(os.path.join(input_path, '*', '*.PNG'))
#     all_files.sort()
#
#     noisy_files, files = [], []  # files: store clean images
#     for file_ in all_files:
#         filename = os.path.split(file_)[-1]
#         if 'GT' in filename:
#             files.append(file_)
#         if 'NOISY' in filename:
#             noisy_files.append(file_)
#
#     # ImageNames = os.listdir(files)
#     # ImageNames = [os.path.join(result_path, 'SIDD_dn', x) for x in files]
#
#     index = 0
#     # process each test image
#     for f in files:
#         img = cv2.imread(f)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         print(img.shape)
#         img = img.transpose(2, 0, 1)
#         img = np.expand_dims(img, 0)
#         print(img.shape)
#         img = np.float32(normalize(img))
#         ground_truth = torch.Tensor(img)
#
#         noise = torch.FloatTensor(ground_truth.size()).normal_(mean=0, std=noise_level / 255.)
#         noisy_img = ground_truth + noise
#
#         ground_truth = Variable(ground_truth.cuda())
#         noisy_img = Variable(noisy_img.cuda())
#
#         with torch.no_grad():
#             start_t = time.time()
#             denoise_img = torch.clamp(noisy_img - model(noisy_img), 0., 1.)
#             if not os.path.exists(result_path):
#                 os.makedirs(result_path)
#             # save_image(denoise_img, result_path+str(ImageNames[index]))
#             index += 1
#
#         psnr = batch_PSNR(ground_truth, denoise_img, 1.)
#         ssim = batch_SSIM(ground_truth, denoise_img, True)
#         psnr_test += psnr
#         ssim_test += ssim
#
#         print("[%s]   PSNR: %.4f  SSIM: %.4f" % (f, psnr, ssim))
#
#         ground_truth = tf.ToPILImage()(np.squeeze(ground_truth))
#         noisy_img = tf.ToPILImage()(np.squeeze(noisy_img))
#         denoise_img = tf.ToPILImage()(np.squeeze(denoise_img))
#         noise = tf.ToPILImage()(np.squeeze(noise))
#         # cv2.imshow('ground truth', numpy.array(ground_truth))
#         # cv2.imshow('noisy img', noisy_img)
#         # cv2.imwrite(result_path+index+'_noisy.png', noisy_img)
#         # cv2.imshow('de-noised img', denoise_img)
#         # cv2.imwrite(result_path+index+'_denoised.png', denoise_img)
#         # cv2.imshow('noise', noise)
#         # cv2.waitKey(0)
#
#         plt.figure(figsize=(15, 10), dpi=200)
#         plt.axis('off')
#         plt.subplot(1, 4, 1)
#         plt.axis('off')
#         plt.imshow(ground_truth)
#         plt.subplot(1, 4, 2)
#         plt.axis('off')
#         plt.imshow(noisy_img)
#         plt.subplot(1, 4, 3)
#         plt.axis('off')
#         plt.imshow(denoise_img)
#         plt.subplot(1, 4, 4)
#         plt.axis('off')
#         plt.imshow(noise)
#         plt.savefig(result_path+'small_SIDD_'+index+'.png')
#         plt.show()
#
#     psnr_test = psnr_test/len(files)
#     ssim_test = ssim_test/len(files)
#
#     end_t = time.time()
#     print("\nFinish tesing! Time: %.2f" % (end_t-start_t))
#     print("Average PSNR: %.4f  |  Average SSIM: %.4f" % (psnr_test, ssim_test))
#
#
# if __name__ == "__main__":
#     main()
#
#
import cv2
import numpy as np
import os,sys
import argparse
from tqdm import tqdm
from natsort import natsorted
from einops import rearrange, repeat

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import scipy.io as sio
import utils

from model import *
from datasets import *

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', default='../dataset/denoising/sidd/test/',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='../results/denoising/sidd/patches-test',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='./log/AtUNet_logs/model_log/model_best.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0,1,2,3', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='AtUNet', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')
parser.add_argument('--win_size', type=int, default=8, help='number of data loading workers')
parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='FFN', help='ffn/FFN token mlp')
parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')

args = parser.parse_args()

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

assert os.path.exists(args.input_dir)
test_dataset = ValDataset(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=32, drop_last=False)

if args.arch == 'DnCNN_Pro':
    channel = 3
    layers_num = 20
    model = DnCNN_Pro(channels=channel, num_of_layers=layers_num)
elif args.arch == 'AtUNet':
    model = AtUNet(img_size=args.train_ps, embed_dim=args.embed_dim, win_size=args.win_size,
                   token_projection=args.token_projection, token_mlp=args.token_mlp)
else:
    raise Exception("Arch error!")

device = torch.device("cuda")

model = torch.nn.DataParallel(model)

utils.load_checkpoint(model, args.weights)
print("===>Testing using weights: ", args.weights)

model.cuda()
model.eval()
torch.cuda.empty_cache()
with torch.no_grad():
    psnr_set = []
    ssim_set = []

    for ii, data_test in enumerate(test_loader, 0):

        gt_img = data_test[0].numpy().squeeze().transpose((1,2,0))
        input_img = data_test[1].cuda()
        filenames = data_test[2]
        # print(filenames)
        # print(input_img.size())

        restored_img = model(input_img)
        restored_img = torch.clamp(restored_img,0,1).cpu().numpy().squeeze().transpose((1,2,0))
        psnr = psnr_loss(restored_img, gt_img)
        ssim = ssim_loss(restored_img, gt_img, multichannel = True)
        psnr_set.append(psnr)
        ssim_set.append(ssim)
        print("#%d PSNR: %f, SSIM: %f " % (ii, psnr, ssim))

        cv2.imwrite(os.path.join(args.result_dir,'result_'+filenames[0]+'.png'), img_as_ubyte(restored_img))
        torch.cuda.empty_cache()

psnr_set = sum(psnr_set)/len(test_dataset)
ssim_set = sum(ssim_set)/len(test_dataset)
print("------- Testing achieve PSNR: %f, SSIM: %f " %(psnr_set,ssim_set))




