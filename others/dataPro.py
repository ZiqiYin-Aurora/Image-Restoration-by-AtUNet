'''
Processing image data, make datasets.
Updated 2022/3/23 By Aurora Yin
'''

import os
import os.path
import numpy as np
import torch
import random
import cv2
import h5py
import glob
from torchvision import transforms as tf
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from utils import data_augmentation
from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter("../essay/essay_logs")

def normalize(x):
    return x/255.

'''
ps: patch size

'''

def extract_patches(datapath, ps, aug_times, stride=1):
    # files = glob.glob(os.path.join(datapath, 'simple_train_data', '*.jpg'))
    files = glob.glob(datapath)
    patchDir = '../dataset/in_patches'

    # ## SIDD ###
    # all_files = glob.glob(os.path.join(datapath, 'SIDD_Small_sRGB_Only/Data', '*', '*.png'))
    # all_files.sort()
    # ## SIDD ###


    # ## SIDD ###
    # noisy_files, files = [], []  #files: store clean images
    # for file_ in all_files:
    #     filename = os.path.split(file_)[-1]
    #     if 'GT' in filename:
    #         files.append(file_)
    #     if 'NOISY' in filename:
    #         noisy_files.append(file_)
    # ## SIDD ###

    # scales = [1, 0.9, 0.8, 0.7]
    scales = [1]
    patches = []
    total_patch_num = 0

    for n in range(len(files)):
        # read image
        file = files[n]
        print(os.path.exists(file))
        # cv2.imshow('image', file)
        img = cv2.imread(file)
        h, w, c = img.shape
        # print(img.shape)

        for s in scales:
            h_scaled, w_scaled = int(h*s), int(w*s)
            img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
            img_scaled = img_scaled.transpose(2, 0, 1)
            # img_scaled = np.expand_dims(img_scaled, axis=0)
            img_scaled = np.float32(normalize(img_scaled))

            k = 0
            c, w, h = img_scaled.shape
            # print(img_scaled.shape)

            patch = img_scaled[:, 0:w-ps+0+1:stride, 0:h-ps+0+1:stride]
            # patch_num = patch.shape[1] * patch.shape[2]

            # ### AaCNN ###
            # index = 0
            # for i in range(0, w-ps+1, stride):
            #     for j in range(0, h-ps+1, stride):
            #         patch = img_scaled[:, i:i+ps, j:j+ps]
            #         # patches.append(patch)
            #         for a in range(0, aug_times):
            #             patch = data_augmentation(patch, mode=np.random.randint(1, 8))
            #             patches.append(patch)
            #             # writer.add_image("Patches", patch, index)
            #             index += 1
            #             if index == 50:
            #                 break
            #         if index == 50:
            #             break
            #     if index == 50:
            #         break
            # ### AaCNN ###


            ## AtUNet ###
            for j in range(5):
                rr = np.random.randint(0, h - ps)
                cc = np.random.randint(0, w - ps)
                # patch = img_scaled[:, rr:rr + ps, cc:cc + ps]
                # print(patch.shape)
                patch = img[rr:rr + ps, cc:cc + ps, :]
                # print(patch.shape)
                cv2.imwrite(os.path.join(patchDir, '{}_{}.png'.format(n + 1, j + 1)), patch)
                # clean_patch = clean_img[rr:rr + PS, cc:cc + PS, :]
                patch = patch.transpose(2, 0, 1)
                # patch = np.float32(normalize(patch))
                patches.append(patch)
            ## AtUNet ###

            patch_num = len(patches)
            print("file: %s scale %.1f #samples: %d" % (files[n], s, patch_num*aug_times))
            total_patch_num += patch_num*aug_times
    # writer.close()
    print("len(patches): %d, total_patch_num: %d" % (len(patches), total_patch_num))
    return patches, total_patch_num



def prepare_data(path, patch_size, stride, aug_times=1):
    # train
    print('process training data')
    h5f = h5py.File('train_data.h5', 'w')
    train_num = 0
    train_path = os.path.join(path, 'simple_train_data', '*.jpg')
    patches, patch_num = extract_patches(train_path, patch_size, aug_times, stride = stride)

    for i in range(len(patches)):
        data = patches[i]
        h5f.create_dataset(str(train_num), data=data)
        train_num += 1
        # h5f.create_dataset(str(train_num)+"_aug_%d" % (j+1), data=data)
    print("finish ")
    h5f.close()

    ##### val #####
    print('\nprocess validation data')
    h5f = h5py.File('val_data.h5', 'w')
    # files = glob.glob(os.path.join(path, 'simple_val_data', '*.jpg'))
    val_num = 0
    val_path = os.path.join(path, 'simple_val_data', '*.jpg')
    _patches, _patch_num = extract_patches(val_path, patch_size, aug_times, stride=stride)
    # files = glob.glob(os.path.join(path, '*', '*.PNG'))
    # files.sort()
    # h5f = h5py.File('val.h5', 'w')
    # val_num = 0
    #
    # for i in range(len(files)):
    #     print("file: %s" % files[i])
    #     img = cv2.imread(files[i])
    #     # img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    #     crop_img = tf.Compose([
    #         tf.ToPILImage(),
    #         tf.CenterCrop(256),
    #         tf.ToTensor()
    #     ])
    #
    #     img = crop_img(img)
    #     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     print(img.shape)
    #     # img = img.transpose(2, 0, 1)
    #     # img = np.expand_dims(img[:,:,0], 0)
    #     # print(img.shape)
    #     # img = np.float32(normalize(img))
    #
    #     h5f.create_dataset(str(val_num), data=img)
    #     val_num += 1
    for i in range(len(_patches)):
        data = _patches[i]
        h5f.create_dataset(str(val_num), data=data)
        val_num += 1
        # h5f.create_dataset(str(train_num)+"_aug_%d" % (j+1), data=data)
    print("finish ")
    h5f.close()

    print('training set, # samples %d\n' % train_num)
    print('val set, # samples %d\n' % val_num)


class MyDataset(Dataset):
    def __init__(self, train=True):
        super(MyDataset, self).__init__()
        self.train = train

        if self.train:
            h5f = h5py.File('train_data.h5', 'r')
        else:
            h5f = h5py.File('val_data.h5', 'r')

        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File('train_data.h5', 'r')
        else:
            h5f = h5py.File('val_data.h5', 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)

