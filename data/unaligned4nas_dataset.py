import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util


class Unaligned4NASDataset(BaseDataset):
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
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.Aw_paths = self.A_paths[:self.A_size // 2]
        self.Aa_paths = self.A_paths[self.A_size // 2:]
        self.Bw_paths = self.B_paths[:self.B_size // 2]
        self.Ba_paths = self.B_paths[self.B_size // 2:]

        self.Aw_size = len(self.Aw_paths)  # get the size of dataset Aw
        self.Aa_size = len(self.Aa_paths)  # get the size of dataset Aa
        self.Bw_size = len(self.Bw_paths)  # get the size of dataset Bw
        self.Ba_size = len(self.Ba_paths)  # get the size of dataset Ba

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
        Aw_path = self.Aw_paths[index % self.Aw_size]  # make sure index is within then range
        Aa_path = self.Aa_paths[index % self.Aa_size]  # make sure index is within then range

        if self.opt.serial_batches:   # make sure index is within then range
            index_Bw = index % self.Bw_size
            index_Ba = index % self.Ba_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_Bw = random.randint(0, self.Bw_size - 1)
            index_Ba = random.randint(0, self.Ba_size - 1)
        Bw_path = self.B_paths[index_Bw]
        Ba_path = self.B_paths[index_Ba]
        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')
        Aw_img = Image.open(Aw_path)
        Aa_img = Image.open(Aa_path)
        Bw_img = Image.open(Bw_path)
        Ba_img = Image.open(Ba_path)

        # Apply image transformation
        # For FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
#        print('current_epoch', self.current_epoch)
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt, grayscale=self.grayscale)

        Aw = transform(Aw_img)
        Aa = transform(Aa_img)
        Bw = transform(Bw_img)
        Ba = transform(Ba_img)

        return {'A': Aw, 'B': Bw, 'A_paths': Aw_path, 'B_paths': Bw_path,
                'Aa': Aa, 'Ba': Ba, 'Aa_paths': Aa_path, 'Ba_paths': Ba_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.Aw_size, self.Bw_size)
