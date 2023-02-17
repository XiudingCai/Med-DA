import os.path

import torch

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import random
import util.util as util
from torchvision import transforms
import SimpleITK as sitk
from monai import transforms, data


class UnalignedSlices4NoRegDataset(BaseDataset):
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

        assert self.A_size == self.B_size

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
        index_A = index % self.A_size
        A_path = self.A_paths[index_A]  # make sure index is within then range

        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        R_path = self.B_paths[index_A]

        # aryA = np.load(os.path.join(*A_path.replace(self.phase + 'A', self.phase + 'A' + '_npy').split('\\')[:-2], A_path.split('\\')[-2] + '.npy'))
        # aryB = np.load(os.path.join(*B_path.replace(self.phase + 'B', self.phase + 'B' + '_npy').split('\\')[:-2], B_path.split('\\')[-2] + '.npy'))
        #
        # idxA = os.listdir(os.path.join(*A_path.split('\\')[:-1])).index(A_path.split('\\')[-1])
        # idxB = os.listdir(os.path.join(*B_path.split('\\')[:-1])).index(B_path.split('\\')[-1])

        A_img = sitk.ReadImage(A_path)
        B_img = sitk.ReadImage(B_path)
        # R_img = sitk.ReadImage(R_path)

        space_A = A_img.GetSpacing()
        origin_A = A_img.GetOrigin()
        direction_A = A_img.GetDirection()
        space_B = B_img.GetSpacing()
        origin_B = B_img.GetOrigin()
        direction_B = B_img.GetDirection()

        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)

        transform_A = get_transform(modified_opt, keys=['image_A', 'image_R'])
        transform_B = get_transform(modified_opt, keys=['image_B'])

        # print(A_path, R_path)
        dataA = transform_A({'image_A': A_path, 'image_R': R_path})
        dataB = transform_B({'image_B': B_path})

        return {'A': dataA['image_A'], 'B': dataB['image_B'], 'R': dataA['image_R'],
                'A_paths': A_path, 'B_paths': B_path,
                'A_meta': {'spacing': space_A, 'origin': origin_A, 'direction': direction_A},
                'B_meta': {'spacing': space_B, 'origin': origin_B, 'direction': direction_B}}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)


def get_transform(opt, keys=['image']):
    DATASET = os.path.basename(opt.dataroot)
    MIN = -1.
    MAX = 1.
    if 'MR2CT' in DATASET:
        opt.padding_size = (512, 512, 256)
        opt.patch_size = (256, 256, 3)
        opt.resize_size = (256, 256, 3)
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=keys),
                transforms.AddChanneld(keys=keys),
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
                ScaleMinMaxNorm(keys=keys, a_min=MIN, a_max=MAX),
                # transforms.CropForegroundd(keys=keys, source_key="image"),
                transforms.SpatialPadd(keys=keys,
                                       spatial_size=(512, 512, 256)),
                transforms.CenterSpatialCropd(keys=keys,
                                              roi_size=(176, 176, 64),),
                transforms.RandSpatialCropd(keys=keys,
                                            roi_size=(176, 176, 3),
                                            random_size=False),
                transforms.Rotate90d(keys=keys, k=2, spatial_axes=(0, 1)),
                transforms.Resized(keys=keys, spatial_size=(256, 256, 3)),
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
                # transforms.RandScaleIntensityd(keys="image",
                #                                factors=0.1,
                #                                prob=0.1),
                # transforms.RandShiftIntensityd(keys="image",
                #                                offsets=0.1,
                #                                prob=0.1),
                transforms.ToTensord(keys=keys),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=keys),
                transforms.AddChanneld(keys=keys),
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
                ScaleMinMaxNorm(keys=keys, a_min=MIN, a_max=MAX),
                transforms.SpatialPadd(keys=keys,
                                       spatial_size=(512, 512, 256)),
                transforms.CenterSpatialCropd(keys=keys,
                                              roi_size=(176, 176, 64),),
                # transforms.RandSpatialCropd(keys=keys,
                #                             roi_size=(176, 176, 3),
                #                             random_size=False),
                transforms.Rotate90d(keys=keys, k=2, spatial_axes=(0, 1)),
                transforms.Resized(keys=keys, spatial_size=(256, 256, 64)),
                transforms.ToTensord(keys=keys),
            ]
        )
    else:
        raise NotImplementedError

    if opt.isTrain:
        return train_transform
    else:
        return val_transform


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
