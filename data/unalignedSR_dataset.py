import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util


class UnalignedSRDataset(BaseDataset):
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
        self.dir_HR = os.path.join(opt.dataroot, 'trainB-HR')  # create a path '/path/to/data/trainB'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.HR_paths = sorted(make_dataset(self.dir_HR, opt.max_dataset_size))
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.HR_size = len(self.HR_paths)  # get the size of dataset HR

        self.grayscale = True if opt.input_nc == 1 else False

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

        dx = A_path.rfind('_')
        idxpng = A_path[dx+1:]
        idx, png = idxpng.split('.')
        idx = int(idx)
        if 0 < idx <= 66:
            index_H = random.randint(28, 90)
        elif 66 < idx <= 92:
            index_H = random.randint(86, 132)
        elif 92 < idx <= 120:
            index_H = random.randint(130, 209)
        else:
            index_H = random.randint(208, 283)
        HR_path = self.HR_paths[(index_H - 28) % self.HR_size]

        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')
        A_img = Image.open(A_path)
        B_img = Image.open(B_path)
        HR_img = Image.open(HR_path)

        # Apply image transformation
        # For FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
#        print('current_epoch', self.current_epoch)
#         is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        # modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        # transform = get_transform(modified_opt, grayscale=self.grayscale)
        from torchvision import transforms
        transform_LR = transforms.Compose([
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((256, 192)),
            # transforms.RandomCrop(self.opt.crop_size),
            transforms.Normalize([0.5, ], [0.5, ])
        ])
        transform_HR = transforms.Compose([
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((512, 384)),
            # transforms.RandomCrop(self.opt.crop_size),
            transforms.Normalize([0.5, ], [0.5, ])
        ])

        A = transform_LR(A_img)
        B = transform_LR(B_img)
        HR = transform_HR(HR_img)

        return {'A': A, 'B': B, 'HR': HR, 'A_paths': A_path, 'B_paths': B_path, 'HR_paths': HR_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
