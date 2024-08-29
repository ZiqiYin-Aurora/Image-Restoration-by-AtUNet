
import warnings

from numpy import mean

warnings.filterwarnings("ignore")

import os, sys
import argparse
from glob import glob

from tqdm import tqdm
from natsort import natsorted

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import utils
from utils_test import my_patchify, my_unpatchify

from model import *
from datasets import *

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--arch', default='AtUNet', type=str, help='arch')
parser.add_argument('--mode', default='denoise', type=str, help='image restoration mode')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--input_dir', default='../dataset/SIDD_test',
                    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='../results/denoising/sidd',
                    type=str, help='Directory for results')
parser.add_argument('--weights', default='./log/AtUNet_logs/model_log/model_best.pth',
                    type=str, help='Path to weights')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')
parser.add_argument('--win_size', type=int, default=8, help='number of data loading workers')
parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str, default='FFN', help='ffn/FFN token mlp')
parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
parser.add_argument('--test_ps', type=int, default=256, help='patch size of training sample')


args = parser.parse_args()

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

if args.arch == 'DnCNN_Pro':
    channel = 3
    layers_num = 20
    model_restoration = DnCNN_Pro(channels=channel, num_of_layers=layers_num)
elif args.arch == 'AtUNet':
    model_restoration = AtUNet(img_size=args.train_ps, embed_dim=args.embed_dim, win_size=args.win_size,
                               token_projection=args.token_projection, token_mlp=args.token_mlp)
else:
    raise Exception("Arch error!")

device = torch.device("cuda")

model_restoration = torch.nn.DataParallel(model_restoration)

if args.mode == 'denoise':
    weight = './log/AtUNet_logs/model_log/model_best.pth'
elif args.mode == 'derain':
    weight = './log_derain/AtUNet_logs/model_log/model_best.pth'
elif args.mode == 'deblur':
    weight = './log_deblur/AtUNet_logs/model_log/model_best_GoPro.pth'

utils.load_checkpoint(model_restoration, weight)
print("===>Testing using weights: ", weight)

model_restoration.cuda()
model_restoration.eval()
torch.cuda.empty_cache()

assert os.path.exists(args.input_dir)
dirs = natsorted(glob(os.path.join(args.input_dir, '*')))
index = 0
psnr_test = []
ssim_test = []

for d in dirs:
    assert os.path.exists(d)
    folder=os.path.split(d)[-1]
    # print(d)
    # print(folder)

    files = natsorted(glob(os.path.join(d, '*'))) ###.png
    res_dir = os.path.join(args.result_dir, folder)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    # print(res_dir)

    csv_path = os.path.join(res_dir, 'psnr&ssim.csv')
    csv_head = ['Image', 'PSNR', 'SSIM']
    create_csv(csv_path, csv_head)
    write_csv(csv_path, ['------', 'Patches', '-------'])

    for f in files:
        # print(f)
        filename = os.path.split(f)[-1]
        if 'GT' in filename or 'mean' in filename:
            gt_path = os.path.join(args.input_dir, folder, filename)
        if ('NOISY' in filename) or ('RAINY' in filename) or ('BLUR' in filename) or ('real' in filename):
            noisy_path = os.path.join(args.input_dir, folder, filename)
            noisy_fn = filename
    # print(noisy_path)
    my_patchify(noisy_path, gt_path, ps=args.test_ps, res_path=res_dir)

    if not os.path.exists(os.path.join(res_dir, 'restored_patches')):
        os.makedirs(os.path.join(res_dir, 'restored_patches'))

    test_dataset = ValDataset(res_dir)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=32, drop_last=False)

    with torch.no_grad():
        # psnr_val_rgb = []
        # ssim_val_rgb = []
        print("--------------- Image " + str(index+1) + " ------------------")
        for ii, data_test in enumerate(test_loader, 0):
            gt_img = data_test[0].numpy().squeeze().transpose((1, 2, 0))
            input_img = data_test[1].cuda()
            filenames = data_test[2]
            # print(input_img.size())

            restored_img = model_restoration(input_img)
            restored_img = torch.clamp(restored_img, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))

            psnr = psnr_loss(restored_img, gt_img)
            ssim = ssim_loss(restored_img, gt_img, multichannel=True)
            # psnr_val_rgb.append(psnr)
            # ssim_val_rgb.append(ssim)
            csv_data = ['%s'%filenames[0], '%.4f'%psnr, '%.4f'%ssim]
            write_csv(csv_path, csv_data)

            if not cv2.imwrite(os.path.join(res_dir, 'restored_patches', filenames[0] + '.png'), img_as_ubyte(restored_img)):
                raise Exception("Could not write the image")
            # + '-psnr:{} ssim:{}'.format(round(psnr,2),round(ssim,2))
            torch.cuda.empty_cache()

    my_unpatchify(res_dir, ps=args.test_ps)

    # psnr_val_rgb = sum(psnr_val_rgb) / len(test_dataset)
    # ssim_val_rgb = sum(ssim_val_rgb) / len(test_dataset)
    # print("Patches Average PSNR: %f, SSIM: %f " % (psnr_val_rgb, ssim_val_rgb))

    restored = cv2.imread(os.path.join(res_dir, 'restored_img.png'))
    gt = cv2.imread(os.path.join(res_dir, 'gt_img.png'))
    psnr_whole = psnr_loss(restored, gt)
    ssim_whole = ssim_loss(restored, gt, multichannel=True)
    psnr_test.append(psnr_whole)
    ssim_test.append(ssim_whole)

    write_csv(csv_path, ['------', 'Whole Image', '------'])
    write_csv(csv_path, ['%s'%folder, '%.4f'%psnr_whole, '%.4f'%ssim_whole])

    print("Whole image PSNR=%.4f, SSIM=%.4f" % (psnr_whole, ssim_whole))
    # os.rename(os.path.join(res_dir, 'restored_img.png'), os.path.join(res_dir, 'restored_'+'psnr:{} ssim:{}'.format(round(psnr_whole,2),round(ssim_whole,2))+'.png'))
    print("------"+folder+"/"+noisy_fn+" DONE!!------")
    index += 1

print("------- Testing achieve PSNR: %f, SSIM: %f " %(mean(psnr_test),mean(ssim_test)))
print("--------Finish restoring %d files!---------" % index)