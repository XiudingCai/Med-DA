import os.path

import torch

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
import numpy as np
from torchvision import transforms


class UnalignedSlices2NDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'

        # self.A_paths = sorted([os.path.join(self.dir_A, p) for p in os.listdir(self.dir_A)])   # load images from '/path/to/data/trainA'
        # self.B_paths = sorted([os.path.join(self.dir_B, p) for p in os.listdir(self.dir_B)])    # load images from '/path/to/data/trainB'

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.grayscale = True
        self.phase = opt.phase
        self.num_K = opt.num_K

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range

        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        aryA = np.load(os.path.join(*A_path.replace(self.phase + 'A', self.phase + 'A' + '_npy').split('\\')[:-2], A_path.split('\\')[-2] + '.npy'))
        aryB = np.load(os.path.join(*B_path.replace(self.phase + 'B', self.phase + 'B' + '_npy').split('\\')[:-2], B_path.split('\\')[-2] + '.npy'))

        idxA = os.listdir(os.path.join(*A_path.split('\\')[:-1])).index(A_path.split('\\')[-1])
        idxB = os.listdir(os.path.join(*B_path.split('\\')[:-1])).index(B_path.split('\\')[-1])

        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')
        # A_img = Image.open(A_path)
        # B_img = Image.open(B_path)

        # Apply image transformation
        # For FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
#        print('current_epoch', self.current_epoch)
#         is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        # modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        # transform = get_transform(modified_opt, grayscale=self.grayscale)
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize(self.opt.load_size),
            transforms.RandomCrop(self.opt.crop_size),
            transforms.Normalize([0.5, ], [0.5, ])
        ])

        A = transform(aryA)
        B = transform(aryB)

        # totally 1 + K + 1 + K = 2K+2 slices
        K = self.num_K

        A = torch.cat([torch.cat([A[[0]] for _ in range(K+1)], dim=0), A,
                       torch.cat([A[[-1]] for _ in range(K)], dim=0)], dim=0)[idxA: idxA + 2 * K + 2]
        B = torch.cat([torch.cat([B[[0]] for _ in range(K+1)], dim=0), B,
                       torch.cat([B[[-1]] for _ in range(K)], dim=0)], dim=0)[idxB: idxB + 2 * K + 2]
        # print(A.shape)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
