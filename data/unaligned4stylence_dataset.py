import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util


class Unaligned4StyleNCEDataset(BaseDataset):
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

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

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

        prefix, idxpng = A_path.split('_')
        idx, png = idxpng.split('.')

        A_pre_path = f"{prefix}_{eval(idx) - 2}.{png}"
        A_post_path = f"{prefix}_{eval(idx) + 2}.{png}"

        if not os.path.exists(A_pre_path):
            A_pre_path = A_path
        if not os.path.exists(A_post_path):
            A_post_path = A_path

        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        prefix, idxpng = B_path.split('_')
        idx, png = idxpng.split('.')

        B_pre_path = f"{prefix}_{eval(idx) - 2}.{png}"
        B_post_path = f"{prefix}_{eval(idx) + 2}.{png}"

        if not os.path.exists(B_pre_path):
            B_pre_path = B_path
        if not os.path.exists(B_post_path):
            B_post_path = B_path

        R_path = A_path.replace('trainA', 'trainB')
        R_pre_path = A_pre_path.replace('trainA', 'trainB')
        R_post_path = A_post_path.replace('trainA', 'trainB')

        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')
        A_img = Image.open(A_path)
        B_img = Image.open(B_path)
        S_img = Image.open(B_path.replace('trainB', 'trainA'))
        R_img = Image.open(R_path)
        A_pre_img = Image.open(A_pre_path)
        B_pre_img = Image.open(B_pre_path)
        S_pre_img = Image.open(B_pre_path.replace('trainB', 'trainA'))
        R_pre_img = Image.open(R_pre_path)
        A_post_img = Image.open(A_pre_path)
        B_post_img = Image.open(B_post_path)
        S_post_img = Image.open(B_post_path.replace('trainB', 'trainA'))
        R_post_img = Image.open(R_post_path)

        # Apply image transformation
        # For FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
        #        print('current_epoch', self.current_epoch)
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt, grayscale=self.grayscale)
        A = transform(A_img)
        B = transform(B_img)
        R = transform(R_img)
        S = transform(S_img)
        A_pre = transform(A_pre_img)
        B_pre = transform(B_pre_img)
        S_pre = transform(S_pre_img)
        R_pre = transform(R_pre_img)
        A_post = transform(A_post_img)
        B_post = transform(B_post_img)
        S_post = transform(S_post_img)
        R_post = transform(R_post_img)

        return {'A': A, 'B': B, 'R': R, 'S': S,
                'A_pre': A_pre, 'B_pre': B_pre, 'R_pre': R_pre, 'S_pre': S_pre,
                'A_post': A_post, 'B_post': B_post, 'R_post': R_post, 'S_post': S_post,
                'A_paths': A_path, 'B_paths': B_path, 'R_paths': R_path,
                'A_pre_paths': A_pre_path, 'B_pre_paths': A_pre_path, 'R_pre_paths': R_pre_path,
                'A_post_paths': A_post_path, 'B_post_paths': A_post_path, 'R_post_paths': R_post_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
