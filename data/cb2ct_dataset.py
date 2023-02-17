import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
import pydicom
import numpy as np


class CB2CTDataset(BaseDataset):
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
        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')
        A_img = Image.open(A_path)
        B_img = Image.open(B_path)

        # Apply image transformation
        # For FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
#        print('current_epoch', self.current_epoch)
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt, grayscale=self.grayscale)
        A = transform(A_img)
        B = transform(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    def get_array_from_dcm(self, dcm_path, min_max=True):
        dcm = pydicom.read_file(dcm_path)  # 读取 dicom 文件
        pixel_array, dcm.Rows, dcm.Columns = self.get_pixeldata(dcm)  # 得到 dicom文件的 CT 值
        if min_max:
            pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
        return pixel_array.copy()

    @staticmethod
    def get_pixeldata(dicom_dataset):
        """If NumPy is available, return an ndarray of the Pixel Data.
        Raises
        ------
        TypeError
            If there is no Pixel Data or not a supported data type.
        ImportError
            If NumPy isn't found
        NotImplementedError
            if the transfer syntax is not supported
        AttributeError
            if the decoded amount of data does not match the expected amount
        Returns
        -------
        numpy.ndarray
           The contents of the Pixel Data element (7FE0,0010) as an ndarray.
        """
        if 'PixelData' not in dicom_dataset:
            raise TypeError("No pixel data found in this dataset.")

        # Make NumPy format code, e.g. "uint16", "int32" etc
        # from two pieces of info:
        # dicom_dataset.PixelRepresentation -- 0 for unsigned, 1 for signed;
        # dicom_dataset.BitsAllocated -- 8, 16, or 32
        if dicom_dataset.BitsAllocated == 1:
            # single bits are used for representation of binary data
            format_str = 'uint8'
        elif dicom_dataset.PixelRepresentation == 0:
            format_str = 'uint{}'.format(dicom_dataset.BitsAllocated)
        elif dicom_dataset.PixelRepresentation == 1:
            format_str = 'int{}'.format(dicom_dataset.BitsAllocated)
        else:
            format_str = 'bad_pixel_representation'
        try:
            numpy_dtype = np.dtype(format_str)
        except TypeError:
            msg = ("Data type not understood by NumPy: "
                   "format='{}', PixelRepresentation={}, "
                   "BitsAllocated={}".format(
                format_str,
                dicom_dataset.PixelRepresentation,
                dicom_dataset.BitsAllocated))
            raise TypeError(msg)

        pixel_bytearray = dicom_dataset.PixelData

        if dicom_dataset.BitsAllocated == 1:
            # if single bits are used for binary representation, a uint8 array
            # has to be converted to a binary-valued array (that is 8 times bigger)
            try:
                pixel_array = np.unpackbits(
                    np.frombuffer(pixel_bytearray, dtype='uint8'))
            except NotImplementedError:
                # PyPy2 does not implement numpy.unpackbits
                raise NotImplementedError(
                    'Cannot handle BitsAllocated == 1 on this platform')
        else:
            pixel_array = np.frombuffer(pixel_bytearray, dtype=numpy_dtype)
        length_of_pixel_array = pixel_array.nbytes
        expected_length = dicom_dataset.Rows * dicom_dataset.Columns
        if ('NumberOfFrames' in dicom_dataset and
                dicom_dataset.NumberOfFrames > 1):
            expected_length *= dicom_dataset.NumberOfFrames
        if ('SamplesPerPixel' in dicom_dataset and
                dicom_dataset.SamplesPerPixel > 1):
            expected_length *= dicom_dataset.SamplesPerPixel
        if dicom_dataset.BitsAllocated > 8:
            expected_length *= (dicom_dataset.BitsAllocated // 8)
        padded_length = expected_length
        if expected_length & 1:
            padded_length += 1
        if length_of_pixel_array != padded_length:
            raise AttributeError(
                "Amount of pixel data %d does not "
                "match the expected data %d" %
                (length_of_pixel_array, padded_length))
        if expected_length != padded_length:
            pixel_array = pixel_array[:expected_length]
        if dicom_dataset.Modality.lower().find('ct') >= 0:  # CT图像需要得到其CT值图像
            pixel_array = pixel_array * dicom_dataset.RescaleSlope + dicom_dataset.RescaleIntercept  # 获得图像的CT值
        pixel_array = pixel_array.reshape(dicom_dataset.Rows, dicom_dataset.Columns * dicom_dataset.SamplesPerPixel)
        return pixel_array, dicom_dataset.Rows, dicom_dataset.Columns

