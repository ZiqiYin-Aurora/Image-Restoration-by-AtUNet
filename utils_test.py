'''
2022/4/17
'''
import os
from glob import glob

import cv2
import h5py
import numpy as np
from natsort import natsorted
from patchify import patchify, unpatchify


'''
dir: (the img need to be restored)
--dir:
----old_img.png
----pad_img.png
----restored_img.png
----noisy.h5
----clean.h5
----input_patches:
--------1_1.png
--------1_2.png
--------......
----restored_patches:
--------1_1.png
--------1_2.png
--------......
'''


################################# Image Patchify ####################################

def my_patchify(img_path, gt_path, ps, res_path):
    # print(img_path)
    img = cv2.imread(img_path)  # H,W,C
    gt = cv2.imread(gt_path)
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # print(img.shape)
    assert img.shape == gt.shape

    cv2.imwrite(os.path.join(res_path, 'old_img.png'), img)
    cv2.imwrite(os.path.join(res_path, 'gt_img.png'), gt) ##

    noisy_h5f = h5py.File(os.path.join(res_path, 'noisy.h5'), 'w')
    clean_h5f = h5py.File(os.path.join(res_path, 'groundtruth.h5'), 'w') ##

    h_old, w_old, c = img.shape
    h_pad = (h_old // ps + 1) * ps - h_old
    w_pad = (w_old // ps + 1) * ps - w_old
    # print(h_pad, w_pad)

    img = cv2.copyMakeBorder(img, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=(0, 255, 0))
    gt = cv2.copyMakeBorder(gt, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=(0, 255, 0)) ##

    # print(img.shape)
    # img = img.squeeze().numpy().transpose((1,2,0))
    cv2.imwrite(os.path.join(res_path, 'pad_img.png'), img)
    cv2.imwrite(os.path.join(res_path, 'pad_gt_img.png'), gt) ##

    patches_img = patchify(img, (ps, ps, 3), step=ps)  # patches_img.shape = (14, 18, 1, ps, ps, 3)
    patches_gt_img = patchify(gt, (ps, ps, 3), step=ps)  ##
    # print(patches_img.shape)

    patches_dir = os.path.join(res_path, 'input_patches')
    # print(patches_dir)
    if not os.path.exists(patches_dir):
        os.makedirs(patches_dir)

    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i, j, 0, :, :, :].copy()
            single_patch_gt_img = patches_gt_img[i, j, 0, :, :, :].copy()

            # cv2.rectangle(single_patch_img, (30, 30), (ps-30, ps-30), (0, 255, 0), 3)  # Draw something (for testing).
            if not cv2.imwrite(os.path.join(patches_dir, '{}_{}.png'.format(i + 1, j + 1)), single_patch_img):
                raise Exception("Could not write the image")
            noisy_h5f.create_dataset('{}_{}'.format(i+1, j+1), data=single_patch_img)
            clean_h5f.create_dataset('{}_{}'.format(i+1, j+1), data=single_patch_gt_img)

    # Store an unpatchified reference for testing
    cv2.imwrite(os.path.join(res_path, 'unpatched_ref.jpg'), unpatchify(patches_img, img.shape))
    noisy_h5f.close()


################################# Image Unpatchify ####################################

def my_unpatchify(dir, ps):
    # Allocate sapces for storing the patches
    old_img = cv2.imread(os.path.join(dir, 'old_img.png'))
    h_old, w_old, c = old_img.shape

    img = cv2.imread(os.path.join(dir, 'pad_img.png'))  # Read test.jpg just for getting the shape
    img = np.zeros_like(img)  # Fill with zeros for the example (start from an empty image).

    # Use patchify just for getting the size. shape = (14, 18, 1, ps, ps, 3)
    # We could have also used: patches = np.zeros((14, 18, 1, ps, ps, 3), np.uint8)
    patches = patchify(img, (ps, ps, 3), step=ps)

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch_img = cv2.imread(os.path.join(dir, 'restored_patches', '{}_{}.png'.format(i + 1, j + 1)))  # Read a patch image.
            if single_patch_img is None:
                raise Exception("Could not read the image")
            patches[i, j, 0, :, :, :] = single_patch_img.copy()  # Copy single path image to patches

    reconstructed_image = unpatchify(patches, img.shape)
    cropped = reconstructed_image[0:h_old, 0:w_old, :]
    cv2.imwrite(os.path.join(dir, 'restored_img.png'), cropped)
    # print(cropped.shape)



####### test Main #######
if __name__ == '__main__':
    dir_name = os.path.dirname(os.path.abspath(__file__))
    res = '../dataset/test/denoise/test_img1'

    img_path = '../dataset/SIDD_Small_sRGB_Only/Data/0192_009_IP_00100_00200_3200_N/NOISY_SRGB_010.PNG' #'test-noisy.PNG'
    gt_path = '../dataset/SIDD_Small_sRGB_Only/Data/0192_009_IP_00100_00200_3200_N/GT_SRGB_010.PNG' #'test-noisy.PNG'

    ps = 512

    if not os.path.exists(res):
        os.makedirs(res)

    assert os.path.exists(res)
    assert os.path.exists(img_path)

    my_patchify(img_path, gt_path, ps, res)
    # my_unpatchify(res, ps)