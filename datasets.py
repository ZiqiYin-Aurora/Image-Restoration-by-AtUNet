'''
Different dataloaders for different usages.
Updated 2022/4/5 By Aurora Yin
'''
import cv2
import h5py
import random
from utils import *
from torch.utils.data import Dataset


augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')]

######### Normalization #########
def normalize(x):
    return x/255.

##################################################################################################
class TrainDataset(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(TrainDataset, self).__init__()

        self.gt_dir = os.path.join(rgb_dir, 'groundtruth.h5')
        self.noisy_dir = os.path.join(rgb_dir, 'noisy.h5')

        with h5py.File(os.path.join(rgb_dir, 'groundtruth.h5'), 'r') as gt:
            self.gt_keys = sorted(list(gt.keys()))
        with h5py.File(os.path.join(rgb_dir, 'noisy.h5'), 'r') as input:
            self.noisy_keys = sorted(list(input.keys()))

        self.img_options = img_options
        self.tar_size = len(self.gt_keys) # get the size of target

    def __len__(self):
        return self.tar_size

    def open_hdf5_gt(self):
        self.gt_h5f = h5py.File(name=self.gt_dir, mode='r')

    def open_hdf5_noisy(self):
        self.noisy_h5f = h5py.File(name=self.noisy_dir, mode='r')

    def __getitem__(self, index):
        if not hasattr(self, "gt_h5f"):
            self.open_hdf5_gt()

        if not hasattr(self, "noisy_h5f"):
            self.open_hdf5_noisy()

        tar_index = index % self.tar_size

        key = self.gt_keys[tar_index]
        clean = torch.from_numpy(np.float32(self.gt_h5f[key]).copy())

        _key = self.noisy_keys[tar_index]
        noisy = torch.from_numpy(np.float32(self.noisy_h5f[_key]).copy())
        #
        # print(key == _key)
        # print(key+' '+_key)

        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)

        # Crop Input and Target
        ps = self.img_options['patch_size']
        h = clean.shape[1]
        w = clean.shape[2]

        if h - ps == 0:
            r = 0
            c = 0
        else:
            r = np.random.randint(0, h - ps)
            c = np.random.randint(0, w - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)

        clean = normalize(clean)
        noisy = normalize(noisy)

        return clean, noisy, key, _key

##################################################################################################

##################################################################################################
class ValDataset(Dataset):
    def __init__(self, rgb_dir):
        super(ValDataset, self).__init__()

        self.gt_dir = os.path.join(rgb_dir, 'groundtruth.h5')
        self.noisy_dir = os.path.join(rgb_dir, 'noisy.h5')

        with h5py.File(os.path.join(rgb_dir, 'groundtruth.h5'), 'r') as gt:
            self.gt_keys = sorted(list(gt.keys()))
        with h5py.File(os.path.join(rgb_dir, 'noisy.h5'), 'r') as input:
            self.noisy_keys = sorted(list(input.keys()))

        self.tar_size = len(self.gt_keys)  # get the size of target

    def __len__(self):
        return self.tar_size

    def open_hdf5_gt(self):
        self.gt_h5f = h5py.File(name=self.gt_dir, mode='r')

    def open_hdf5_noisy(self):
        self.noisy_h5f = h5py.File(name=self.noisy_dir, mode='r')

    def __getitem__(self, index):
        if not hasattr(self, "gt_h5f"):
            self.open_hdf5_gt()

        if not hasattr(self, "noisy_h5f"):
            self.open_hdf5_noisy()

        tar_index = index % self.tar_size

        key = self.gt_keys[tar_index]
        clean = torch.from_numpy(np.float32(self.gt_h5f[key]).copy())

        _key = self.noisy_keys[tar_index]
        noisy = torch.from_numpy(np.float32(self.noisy_h5f[_key]).copy())

        # key = self.gt_keys[tar_index]
        # data = np.array(self.gt_h5f[key])
        # clean = torch.from_numpy(data)
        #
        # _key = self.noisy_keys[tar_index]
        # _data = np.array(self.noisy_h5f[_key])
        # noisy = torch.from_numpy(_data)

        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)

        clean = normalize(clean)
        noisy = normalize(noisy)

        return clean, noisy, key, _key


##################################################################################################

##################################################################################################
class TestDataset(Dataset):
    def __init__(self, rgb_dir):
        super(TestDataset, self).__init__()
        self.dir = os.path.join(rgb_dir, 'test_data.h5')
        with h5py.File(os.path.join(rgb_dir, 'test_data.h5'), 'r') as f:
            self.noisy_keys = sorted(list(f.keys()))

        self.tar_size = len(self.noisy_keys)  # get the size of target

    def __len__(self):
        return self.tar_size

    def open_hdf5(self):
        self.noisy_h5f = h5py.File(name=self.dir, mode='r')

    def __getitem__(self, index):
        if not hasattr(self, "noisy_h5f"):
            self.open_hdf5()

        tar_index = index % self.tar_size

        _key = self.noisy_keys[tar_index]
        _data = np.array(self.noisy_h5f[_key])
        noisy = torch.from_numpy(_data)

        noisy = noisy.permute(2, 0, 1)

        return noisy, _key
