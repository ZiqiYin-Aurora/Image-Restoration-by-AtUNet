

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

parser = argparse.ArgumentParser(description='Making dataset for real testing data')
parser.add_argument('--src_dir', default='./dataset/test', type=str, help='Directory for full resolution images')
parser.add_argument('--tar_dir', default='../dataset/denoising/sidd/test',type=str, help='Directory for image patches')
parser.add_argument('--imgtype', default='*.jpg', type=str, help='Type of image in your dataset, e.g. *.jpg')
parser.add_argument('--num_cores', default=10, type=int, help='Number of CPU Cores')

args = parser.parse_args()
src = args.src_dir
tar = args.tar_dir
type = args.imgtype
NUM_CORES = args.num_cores

h5f = h5py.File(os.path.join(tar,'test_data.h5'), 'w')

f = natsorted(glob(os.path.join(src, type)))

def create_dataset(i):
    for file_ in f:
        filename = os.path.split(file_)[-1]
        img = cv2.imread(file_)
        h5f.create_dataset(filename+type, data=img)

Parallel(n_jobs=NUM_CORES)(delayed(create_dataset)(i) for i in tqdm(range(len(f))))
h5f.close()


