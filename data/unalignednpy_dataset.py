import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
import numpy as np

class UnalignedNPYDataset(BaseDataset):
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
        print(opt.phase)
        if opt.phase == 'train':
            self.dir_A = "/home/cas/home_ez/Datasets/MPSCL/data/data_np/train_ct"
            self.dir_B = "/home/cas/home_ez/Datasets/MPSCL/data/data_np/train_mr"
            self.dir_A_gt = "/home/cas/home_ez/Datasets/MPSCL/data/data_np/gt_train_ct"
            self.dir_B_gt = "/home/cas/home_ez/Datasets/MPSCL/data/data_np/gt_train_mr"
        elif opt.phase == 'test':
            self.dir_A = "/home/cas/home_ez/Datasets/MPSCL/data/data_np/val_ct"
            self.dir_B = "/home/cas/home_ez/Datasets/MPSCL/data/data_np/val_mr"
            self.dir_A_gt = "/home/cas/home_ez/Datasets/MPSCL/data/data_np/gt_val_ct"
            self.dir_B_gt = "/home/cas/home_ez/Datasets/MPSCL/data/data_np/gt_val_mr"

        print(self.dir_A_gt)
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_gt_paths = sorted(make_dataset(self.dir_A_gt, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_gt_paths = sorted(make_dataset(self.dir_B_gt, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

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

        A_gt_path = self.A_gt_paths[index % self.A_size]
        B_gt_path = self.B_gt_paths[index_B]

        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')
        A_img = Image.fromarray((np.load(A_path) + 1) * 127.5)
        B_img = Image.fromarray((np.load(B_path) + 1) * 127.5)

        A_gt_img = Image.fromarray(np.load(A_gt_path), mode='L')
        B_gt_img = Image.fromarray(np.load(B_gt_path), mode='L')

        # Apply image transformation
        # For FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
#        print('current_epoch', self.current_epoch)
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)

        if self.opt.input_nc == 1 and self.opt.output_nc == 1:
            transform = get_transform(modified_opt, grayscale=True)
        else:
            transform = get_transform(modified_opt)

        A = transform(A_img)
        B = transform(B_img)
        # A_gt = transform(A_gt_img)
        # B_gt = transform(B_gt_img)

        return {'A': A, 'B': B, 'A_gt': A_gt, 'B_gt': B_gt, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)