'''
Generate patches and save, according to paras given.
Updated 2022/4/5 By Aurora Yin
'''

from glob import glob
from tqdm import tqdm
import numpy as np
import os
from natsort import natsorted
import cv2
import h5py
from joblib import Parallel, delayed
import multiprocessing
import argparse
from noiseAdd import *

# if __name__ == '__main__':
def crop_as_square(f):
    img = cv2.imread(f, cv2.IMREAD_COLOR)
    # 根据最短边进行图像裁剪
    sp = img.shape
    height = sp[0]  # height(height) of image
    width = sp[1]  # width(colums) of image
    if height >= width:
        shorter = width
    else:
        shorter = height
    cropped = img[0:shorter, 0:shorter]  # 裁剪坐标为[y0:y1, x0:x1]
    cropped = cv2.resize(cropped, dsize=(1024,1024),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
    return cropped

parser = argparse.ArgumentParser(description='Generate patches from Full Resolution images')
parser.add_argument('--src_dir', default='../dataset/SIDD_test/Data', type=str, help='Directory for full resolution images')
parser.add_argument('--tar_dir', default='../dataset/denoising/sidd/test',type=str, help='Directory for image patches')
parser.add_argument('--num_cores', default=10, type=int, help='Number of CPU Cores')
parser.add_argument('--imgtype', default='*.jpg', type=str, help='Type of image in your dataset, e.g. *.jpg')

args = parser.parse_args()

src = args.src_dir
tar = args.tar_dir
type = args.imgtype
NUM_CORES = args.num_cores

input_Dir = os.path.join(tar, 'input')
clean_Dir = os.path.join(tar, 'groundtruth')

if os.path.exists(tar):
    os.system("rm -r {}".format(tar))

os.makedirs(input_Dir)
os.makedirs(clean_Dir)

noisy_h5f = h5py.File(os.path.join(tar,'noisy.h5'), 'w')
gt_h5f = h5py.File(os.path.join(tar,'groundtruth.h5'), 'w')


# get sorted folders
files = natsorted(glob(os.path.join(src, '*', '*.PNG')))
noisy_files, clean_files = [], []
ig = 1
ii = 1

for file_ in files:
    filename = os.path.split(file_)[-1]

    if 'GT' in filename:
        clean_files.append(file_)
        img = cv2.imread(file_)
        # img = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        img = crop_as_square(file_)
        gt_h5f.create_dataset('{}'.format(ig), data=img)
        # cv2.imwrite(os.path.join(tar, 'groundtruth', '{}.png'.format(ig)), img)
        ig += 1
    if 'NOISY' in filename:
        noisy_files.append(file_)
        img = cv2.imread(file_)
        # img = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        img = crop_as_square(file_)
        noisy_h5f.create_dataset('{}'.format(ii), data=img)
        # cv2.imwrite(os.path.join(tar, 'input', '{}.png'.format(ii)), img)
        ii += 1

gt_h5f.close()
noisy_h5f.close()

print("DONE! noisy=%d, clean=%d" % (len(noisy_files), len(clean_files)))

