import os.path

import torch

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import random
import util.util as util
from torchvision import transforms
import SimpleITK as sitk
from monai import transforms, data

from glob import glob
from sklearn.model_selection import train_test_split


class Unaligned4MMWHSDataset(BaseDataset):
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

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.A_gt_paths = sorted(
            make_dataset(self.dir_A_gt, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_gt_paths = sorted(
            make_dataset(self.dir_B_gt, opt.max_dataset_size))  # load images from '/path/to/data/trainB'

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.grayscale = True
        self.phase = opt.phase
        # self.num_K = opt.num_K

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
        index_A = index % self.A_size
        A_path = self.A_paths[index_A]  # make sure index is within then range

        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_gt_path = self.A_gt_paths[index_A]
        B_gt_path = self.B_gt_paths[index_B]  # make sure index is within then range

        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)

        transform_A = get_transform(modified_opt, keys=['image_A', 'image_A_gt'], label=True)
        transform_B = get_transform(modified_opt, keys=['image_B', 'image_B_gt'], label=True)

        # print(A_path, R_path)
        dataA = transform_A({'image_A': A_path, 'image_A_gt': A_gt_path})
        dataB = transform_B({'image_B': B_path, 'image_B_gt': B_gt_path})

        return {'A': dataA['image_A'], 'A_gt': dataA['image_A_gt'],
                'B': dataB['image_B'], 'B_gt': dataB['image_B_gt'],
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)


def get_transform(opt, keys=['image'], label=False):
    MIN = 0.
    MAX = 1.
    mode = ['bilinear', 'nearest']
    from monai.data.image_reader import NumpyReader
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=keys, reader=NumpyReader),
            transforms.EnsureChannelFirstd(keys=keys),
            # transforms.Orientationd(keys=keys,
            #                         axcodes="RAS"),
            # transforms.Spacingd(keys=keys,
            #                     pixdim=(opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]),
            #                     mode=("bilinear",)),  # both images
            # transforms.ScaleIntensityRanged(keys=keys,
            #                                 a_min=0,
            #                                 a_max=1000,
            #                                 b_min=-1.,
            #                                 b_max=1.,
            #                                 clip=True),
            # DeNorm(keys=keys[:-1], a_min=MIN, a_max=MAX),
            # ScaleMinMaxNorm(keys=keys[:-1], a_min=MIN, a_max=MAX),
            # transforms.CropForegroundd(keys=keys, source_key="image"),
            # transforms.SpatialPadd(keys=keys, spatial_size=(196, 196, -1), mode='reflect'),
            # CheckDim(keys=keys),
            # transforms.RandSpatialCropd(keys=keys,
            #                             roi_size=(176, 176, 1),
            #                             random_size=False),
            # CheckDim(keys=keys),

            # transforms.Resized(keys=keys, spatial_size=(256, 256, 1),
            #                    mode=mode),
            # CheckDim(keys=keys),

            # transforms.Rotate90d(keys=keys, k=2, spatial_axes=(0, 1)),
            # transforms.Rand3DElasticd(keys=keys, sigma_range=(5, 7), magnitude_range=(50, 150),
            #                           padding_mode='zeros', mode=mode),
            # transforms.RandCropByPosNegLabeld(
            #     keys=keys,
            #     label_key="image",
            #     spatial_size=(opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]),
            #     pos=1,
            #     neg=1,
            #     num_samples=4,
            #     image_key="image",
            #     image_threshold=0,
            # ),
            # transforms.RandFlipd(keys=keys,
            #                      prob=0.2,
            #                      spatial_axis=0),
            # transforms.RandFlipd(keys=keys,
            #                      prob=0.2,
            #                      spatial_axis=1),
            # transforms.RandFlipd(keys=keys,
            #                      prob=0.2,
            #                      spatial_axis=2),
            # transforms.RandRotate90d(
            #     keys=keys,
            #     prob=0.2,
            #     max_k=3,
            # ),
            # transforms.RandScaleIntensityd(keys=keys[:-1], factors=0.1, prob=1.0),
            # transforms.RandShiftIntensityd(keys=keys[:-1], offsets=0.1, prob=1.0),
            transforms.ToTensord(keys=keys),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=keys, reader=NumpyReader),
            transforms.EnsureChannelFirstd(keys=keys),
            # transforms.Orientationd(keys=keys,
            #                         axcodes="RAS"),
            # transforms.Spacingd(keys=keys,
            #                     pixdim=(opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]),
            #                     mode=("bilinear",)),  # both images
            # transforms.ScaleIntensityRanged(keys=keys,
            #                                 a_min=0,
            #                                 a_max=1000,
            #                                 b_min=-1.,
            #                                 b_max=1.,
            #                                 clip=True),
            # DeNorm(keys=keys[:-1], a_min=MIN, a_max=MAX),
            # ScaleMinMaxNorm(keys=keys[:-1], a_min=MIN, a_max=MAX),
            # transforms.CenterSpatialCropd(keys=keys, roi_size=(176, 176, 1), ),
            # transforms.Resized(keys=keys, spatial_size=(256, 256, -1), mode=mode),
            # transforms.Rotate90d(keys=keys, k=2, spatial_axes=(0, 1)),
            transforms.ToTensord(keys=keys),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=keys, reader=NumpyReader),
            transforms.EnsureChannelFirstd(keys=keys),
            # transforms.Orientationd(keys=keys,
            #                         axcodes="RAS"),
            # transforms.Spacingd(keys=keys,
            #                     pixdim=(opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]),
            #                     mode=("bilinear",)),  # both images
            # transforms.ScaleIntensityRanged(keys=keys,
            #                                 a_min=0,
            #                                 a_max=1000,
            #                                 b_min=-1.,
            #                                 b_max=1.,
            #                                 clip=True),
            # DeNorm(keys=keys[:-1], a_min=MIN, a_max=MAX),
            # ScaleMinMaxNorm(keys=keys[:-1], a_min=MIN, a_max=MAX),
            # transforms.CenterSpatialCropd(keys=keys, roi_size=(176, 176, -1), ),
            # transforms.Resized(keys=keys, spatial_size=(256, 256, -1), mode=mode),
            # transforms.Rotate90d(keys=keys, k=2, spatial_axes=(0, 1)),
            transforms.ToTensord(keys=keys),
        ]
    )

    if opt.isTrain and opt.phase == 'train':
        return train_transform
    elif opt.isTrain and opt.phase == 'test':
        return val_transform
    else:
        return test_transform


from monai.transforms.transform import MapTransform
from typing import Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor


class ScaleMinMaxNorm(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            a_min: float = 0,
            a_max: float = 1,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.a_min = a_min
        self.a_max = a_max

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = ((d[key] - d[key].min()) / (d[key].max() - d[key].min())) * (self.a_max - self.a_min) + self.a_min
        return d


class DeNorm(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            a_min: float = 0,
            a_max: float = 1,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.a_min = a_min
        self.a_max = a_max

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = (d[key] + 1) * 127.5
        return d


class CheckDim(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            a_min: float = 0,
            a_max: float = 1,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.a_min = a_min
        self.a_max = a_max

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            print(key, d[key].shape)
        return d
