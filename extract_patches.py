'''
Generate patches and save, according to paras given.
Updated 2022/4/5 By Aurora Yin
'''

from glob import glob
from tqdm import tqdm
from natsort import natsorted
import h5py
from joblib import Parallel, delayed
import argparse
from noiseAdd import *

# if __name__ == '__main__':

parser = argparse.ArgumentParser(description='Generate patches from Full Resolution images')
parser.add_argument('--dataset', default='SIDD', type=str, help='SIDD/Rain100L/GoPro or others: denoise_clean_only/derain_clean_only/')
parser.add_argument('--src_dir', default='../dataset/SIDD_Small_sRGB_Only/Data', type=str, help='Directory for full resolution images')
parser.add_argument('--tar_dir', default='../dataset/denoising/sidd/train',type=str, help='Directory for image patches')
parser.add_argument('--ps', default=256, type=int, help='Image Patch Size')
parser.add_argument('--num_patches', default=100, type=int, help='Number of patches per image')
parser.add_argument('--num_cores', default=10, type=int, help='Number of CPU Cores')
parser.add_argument('--noise_level', default=25, type=int, help='If only clean images in dataset, then need set noise level to generate noisy image!')
parser.add_argument('--imgtype', default='*.jpg', type=str, help='Type of image in your dataset, e.g. *.jpg')

args = parser.parse_args()

dataset = args.dataset
src = args.src_dir
tar = args.tar_dir
ps = args.ps
type = args.imgtype
NUM_PATCHES = args.num_patches
NOISE_LEVEL = args.noise_level
NUM_CORES = args.num_cores

noisy_patchDir = os.path.join(tar, 'noisy')
clean_patchDir = os.path.join(tar, 'groundtruth')

if os.path.exists(tar):
    os.system("rm -r {}".format(tar))

os.makedirs(noisy_patchDir)
os.makedirs(clean_patchDir)

noisy_h5f = h5py.File(os.path.join(tar,'noisy.h5'), 'w')
gt_h5f = h5py.File(os.path.join(tar,'groundtruth.h5'), 'w')

if dataset == 'SIDD':   #dataset for denoise
    # get sorted folders
    files = natsorted(glob(os.path.join(src, '*', '*.PNG')))
    noisy_files, clean_files = [], []
    for file_ in files:
        filename = os.path.split(file_)[-1]
        if 'GT' in filename:
            print("ok")
            if '010' in filename:
                print("ok2")
                clean_files.append(file_)
        if 'NOISY' in filename:
            if '010' in filename:
                noisy_files.append(file_)
elif 'Rain' in dataset: # dataset for derain
    files = natsorted(glob(os.path.join(src, '*', '*.png')))
    noisy_files, clean_files = [], []
    for file_ in files:
        filename = os.path.split(file_)[-1]
        if 'x2' in filename:
            noisy_files.append(file_)
        else:
            clean_files.append(file_)
elif 'GoPro' in dataset: # dataset for deblur
    gt_f = natsorted(glob(os.path.join(src, '*', 'sharp', '*.png')))
    blur_f = natsorted(glob(os.path.join(src, '*', 'blur', '*.png')))
    noisy_files, clean_files = [], []
    for f in gt_f:
        fn = os.path.split(f)[-1]
        clean_files.append(f)
    for _f in blur_f:
        fn = os.path.split(_f)[-1]
        noisy_files.append(_f)
elif 'RSCD' in dataset:   # RSCD dataset for deblur
    gt_f = natsorted(glob(os.path.join(src, 'global', '*', '*.png')))
    blur_f = natsorted(glob(os.path.join(src, 'rolling', '*', '*.png')))
    noisy_files, clean_files = [], []
    for f in gt_f:
        fn = os.path.split(f)[-1]
        clean_files.append(f)
    for _f in blur_f:
        fn = os.path.split(_f)[-1]
        noisy_files.append(_f)
elif 'CBSD' in dataset:
    gt_f = natsorted(glob(os.path.join(src, 'original_png', '*.png')))
    noisy_f = natsorted(glob(os.path.join(src, 'noisy25', '*.png')))
    noisy_files, clean_files = [], []
    for f in gt_f:
        fn = os.path.split(f)[-1]
        clean_files.append(f)
    for _f in noisy_f:
        fn = os.path.split(_f)[-1]
        noisy_files.append(_f)
elif 'PolyU' in dataset:
    noisy_files, clean_files = [], []
    filenames = natsorted(glob(os.path.join(src, '*', '*.JPG')))
    print("ok")
    for f in filenames:
        if 'mean.JPG' in f:
            fn = os.path.split(f)[-1]
            clean_files.append(f)
        if 'real.JPG' in f:
            fn = os.path.split(f)[-1]
            noisy_files.append(f)

print("noisy=%d, clean=%d" % (len(noisy_files), len(clean_files)))

def save_files(i):
    if (dataset == 'SIDD') or ('Rain' in dataset) or ('GoPro' in dataset) or ('RSCD' in dataset) or ('CBSD' in dataset) or ('PolyU' in dataset):
        noisy_file, clean_file = noisy_files[i], clean_files[i]
        clean_img = cv2.imread(clean_file)
        noisy_img = cv2.imread(noisy_file)
    else:
        clean_file = clean_files[i]
        clean_img = cv2.imread(clean_file)
        noisy_img = gasuss_noise(clean_file, mean=0, var=NOISE_LEVEL / 255.)

    h = clean_img.shape[0]
    w = clean_img.shape[1]
    for j in range(NUM_PATCHES):
        rr = np.random.randint(0, h - ps)
        cc = np.random.randint(0, w - ps)
        clean_patch = clean_img[rr:rr + ps, cc:cc + ps, :]
        noisy_patch = noisy_img[rr:rr + ps, cc:cc + ps, :]

        cv2.imwrite(os.path.join(clean_patchDir, '{}_{}.png'.format(i+1,j+1)), clean_patch)
        cv2.imwrite(os.path.join(noisy_patchDir, '{}_{}.png'.format(i+1,j+1)), noisy_patch)

Parallel(n_jobs=NUM_CORES)(delayed(save_files)(i) for i in tqdm(range(len(clean_files))))


clean_patches = natsorted(glob(os.path.join(clean_patchDir, '*.png')))
noisy_patches = natsorted(glob(os.path.join(noisy_patchDir, '*.png')))

for c in clean_patches:
    img = cv2.imread(c)
    filename = os.path.split(c)[-1]
    gt_h5f.create_dataset(filename, data=img)

for n in noisy_patches:
    img = cv2.imread(n)
    filename = os.path.split(n)[-1]
    noisy_h5f.create_dataset(filename, data=img)

# noisy_h5f.create_dataset('{}_{}.png'.format(i + 1, j + 1), data=noisy_patch)
gt_h5f.close()
noisy_h5f.close()
