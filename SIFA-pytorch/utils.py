##load data
from torch.utils.data import Dataset
import numpy as np
import os
import torch
import yaml
import random
import configparser
import torchvision.transforms as transforms
import nibabel
from scipy import ndimage
from PIL import Image
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# config setting
def is_int(val_str):
    start_digit = 0
    if (val_str[0] == '-'):
        start_digit = 1
    flag = True
    for i in range(start_digit, len(val_str)):
        if (str(val_str[i]) < '0' or str(val_str[i]) > '9'):
            flag = False
            break
    return flag


def is_float(val_str):
    flag = False
    if ('.' in val_str and len(val_str.split('.')) == 2 and not ('./' in val_str)):
        if (is_int(val_str.split('.')[0]) and is_int(val_str.split('.')[1])):
            flag = True
        else:
            flag = False
    elif ('e' in val_str and val_str[0] != 'e' and len(val_str.split('e')) == 2):
        if (is_int(val_str.split('e')[0]) and is_int(val_str.split('e')[1])):
            flag = True
        else:
            flag = False
    else:
        flag = False
    return flag


def is_bool(var_str):
    if (var_str.lower() == 'true' or var_str.lower() == 'false'):
        return True
    else:
        return False


def parse_bool(var_str):
    if (var_str.lower() == 'true'):
        return True
    else:
        return False


def is_list(val_str):
    if (val_str[0] == '[' and val_str[-1] == ']'):
        return True
    else:
        return False


def parse_list(val_str):
    sub_str = val_str[1:-1]
    splits = sub_str.split(',')
    output = []
    for item in splits:
        item = item.strip()
        if (is_int(item)):
            output.append(int(item))
        elif (is_float(item)):
            output.append(float(item))
        elif (is_bool(item)):
            output.append(parse_bool(item))
        elif (item.lower() == 'none'):
            output.append(None)
        else:
            output.append(item)
    return output


def parse_value_from_string(val_str):
    #     val_str = val_str.encode('ascii','ignore')
    if (is_int(val_str)):
        val = int(val_str)
    elif (is_float(val_str)):
        val = float(val_str)
    elif (is_list(val_str)):
        val = parse_list(val_str)
    elif (is_bool(val_str)):
        val = parse_bool(val_str)
    elif (val_str.lower() == 'none'):
        val = None
    else:
        val = val_str
    return val


def parse_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    output = {}
    for section in config.sections():
        output[section] = {}
        for key in config[section]:
            val_str = str(config[section][key])
            if (len(val_str) > 0):
                val = parse_value_from_string(val_str)
                output[section][key] = val
            else:
                val = None
            print(section, key, val_str, val)
    return output


def load_npz(path):
    img = np.load(path)['arr_0']
    gt = np.load(path)['arr_1']
    return img, gt


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def set_random(seed_id=1234):
    np.random.seed(seed_id)
    torch.manual_seed(seed_id)  # for cpu
    torch.cuda.manual_seed_all(seed_id)  # for GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# config setting
def is_int(val_str):
    start_digit = 0
    if (val_str[0] == '-'):
        start_digit = 1
    flag = True
    for i in range(start_digit, len(val_str)):
        if (str(val_str[i]) < '0' or str(val_str[i]) > '9'):
            flag = False
            break
    return flag


def is_float(val_str):
    flag = False
    if ('.' in val_str and len(val_str.split('.')) == 2 and not ('./' in val_str)):
        if (is_int(val_str.split('.')[0]) and is_int(val_str.split('.')[1])):
            flag = True
        else:
            flag = False
    elif ('e' in val_str and val_str[0] != 'e' and len(val_str.split('e')) == 2):
        if (is_int(val_str.split('e')[0]) and is_int(val_str.split('e')[1])):
            flag = True
        else:
            flag = False
    else:
        flag = False
    return flag


def is_bool(var_str):
    if (var_str.lower() == 'true' or var_str.lower() == 'false'):
        return True
    else:
        return False


def parse_bool(var_str):
    if (var_str.lower() == 'true'):
        return True
    else:
        return False


def is_list(val_str):
    if (val_str[0] == '[' and val_str[-1] == ']'):
        return True
    else:
        return False


def parse_list(val_str):
    sub_str = val_str[1:-1]
    splits = sub_str.split(',')
    output = []
    for item in splits:
        item = item.strip()
        if (is_int(item)):
            output.append(int(item))
        elif (is_float(item)):
            output.append(float(item))
        elif (is_bool(item)):
            output.append(parse_bool(item))
        elif (item.lower() == 'none'):
            output.append(None)
        else:
            output.append(item)
    return output


def parse_value_from_string(val_str):
    #     val_str = val_str.encode('ascii','ignore')
    if (is_int(val_str)):
        val = int(val_str)
    elif (is_float(val_str)):
        val = float(val_str)
    elif (is_list(val_str)):
        val = parse_list(val_str)
    elif (is_bool(val_str)):
        val = parse_bool(val_str)
    elif (val_str.lower() == 'none'):
        val = None
    else:
        val = val_str
    return val


def parse_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    output = {}
    for section in config.sections():
        output[section] = {}
        for key in config[section]:
            val_str = str(config[section][key])
            if (len(val_str) > 0):
                val = parse_value_from_string(val_str)
                output[section][key] = val
            else:
                val = None
            print(section, key, val_str, val)
    return output


class UnpairedDataset(Dataset):
    # get unpaired dataset, such as MR-CT dataset
    def __init__(self, A_path, B_path, preprocess, phase):
        self.dir_A = os.path.join(A_path, phase)
        self.dir_Atarget = os.path.join(A_path, phase + 'target')
        self.dir_B = os.path.join(B_path, phase)
        self.dir_Btarget = os.path.join(B_path, phase + 'target')
        self.paths_A = os.listdir(self.dir_A)
        self.boundingbox = []
        self.index_A = [0]
        for dir in self.paths_A:
            Atarget_path = os.path.join(self.dir_Atarget, dir)
            Atarget_img = load_nifty_volume_as_array2(Atarget_path)
            # Atarget_img[Atarget_img == 63] = 255
            # Atarget_img[Atarget_img<255] = 0
            boundingbox = get_ND_bounding_box(Atarget_img, 30)
            self.boundingbox.append(boundingbox)
            idex = boundingbox[1][0] - boundingbox[0][0] + self.index_A[-1]
            print(boundingbox[1][0], boundingbox[0][0], self.index_A[-1], "247")
            # idex = boundingbox[1]-boundingbox[0]+self.index_A[-1]
            self.index_A.append(idex)
        self.paths_B = os.listdir(self.dir_B)
        self.boundingbox_B = []
        self.index_B = [0]
        for dir in self.paths_B:
            Btarget_img = os.path.join(self.dir_Btarget, dir)
            Btarget_img = load_nifty_volume_as_array2(Btarget_img)
            # Btarget_img[Btarget_img == 63] = 255                         
            # Btarget_img[Btarget_img<255] = 0
            boundingbox = get_ND_bounding_box(Btarget_img, 60)
            self.boundingbox_B.append(boundingbox)
            self.index_B.append(
                boundingbox[1][0] - boundingbox[0][0] + self.index_B[-1])  # index[-1] biaoshi zui hou yi ge
        '''listA = os.listdir(A_path)
        listB = os.listdir(B_path)
        self.listA = [os.path.join(A_path,k) for k in listA]
        self.listB = [os.path.join(B_path,k) for k in listB]
        self.Asize = len(self.listA)
        self.Bsize = len(self.listB)'''
        self.Asize = self.index_A[-1]  # A de zong qie pian ceng shu
        self.Bsize = self.index_B[-1]  # B de zong qie pian ceng shu
        self.dataset_size = max(self.Asize, self.Bsize)
        self.preprocess = preprocess  # "resize and crop"
        self.crop_size = 256
        self.load_size = 288
        self.phase = phase

    def __getitem__(self, index):
        index_A = index % self.Asize  # Asize qie pian dd zongshu
        id_A = int(
            np.where((np.array(self.index_A)) <= index_A)[0].max())  # fan hui gai index_A qiepian shuliang de index
        if (id_A == len(self.paths_A)): id_A = id_A - 1
        A_path = os.path.join(self.dir_A, self.paths_A[id_A])
        slice_A = index_A - self.index_A[id_A]

        volume_A = load_nifty_volume_as_array2(A_path)
        volume_A = crop_ND_volume_with_bounding_box(volume_A, self.boundingbox[id_A][0], self.boundingbox[id_A][1])
        print(self.boundingbox[id_A][0], self.boundingbox[id_A][1])
        A = volume_A[slice_A]
        print(A.shape)
        index_B = index % self.Bsize
        id_B = int(np.where((np.array(self.index_B)) <= index_B)[0].max())
        if (id_B == len(self.paths_B)): id_B = id_B - 1
        B_path = os.path.join(self.dir_B, self.paths_B[id_B])
        slice_B = index_B - self.index_B[id_B]
        volume_B = load_nifty_volume_as_array2(B_path)
        volume_B = crop_ND_volume_with_bounding_box(volume_B, self.boundingbox_B[id_B][0], self.boundingbox_B[id_B][1])
        B = volume_B[slice_B]
        Atarget_path = os.path.join(self.dir_Atarget, self.paths_A[id_A])
        volume_Atarget = load_nifty_volume_as_array2(Atarget_path)
        volume_Atarget = crop_ND_volume_with_bounding_box(volume_Atarget, self.boundingbox[id_A][0],
                                                          self.boundingbox[id_A][1])
        A_gt = volume_Atarget[slice_A]
        Btarget_path = os.path.join(self.dir_Btarget, self.paths_B[id_B])
        volume_Btarget = load_nifty_volume_as_array2(Btarget_path)
        volume_Btarget = crop_ND_volume_with_bounding_box(volume_Btarget, self.boundingbox_B[id_B][0],
                                                          self.boundingbox_B[id_B][1])
        B_gt = volume_Btarget[slice_B]
        '''if self.Asize == self.dataset_size:
            A,A_gt = load_npz(self.listA[index])
            B,B_gt = load_npz(self.listB[random.randint(0, self.Bsize - 1)])
        else :
            B,B_gt = load_npz(self.listB[index])
            A,A_gt = load_npz(self.listA[random.randint(0, self.Asize - 1)])


        A = torch.from_numpy(A.copy()).unsqueeze(0).float()
        A_gt = torch.from_numpy(A_gt.copy()).unsqueeze(0).float()
        B = torch.from_numpy(B.copy()).unsqueeze(0).float()
        B_gt = torch.from_numpy(B_gt.copy()).unsqueeze(0).float()'''
        if self.phase == 'val' or self.phase == 'test':
            A, B, A_gt, B_gt = get_val_transform2(self.preprocess, A, B, A_gt, B_gt, self.load_size, self.crop_size)

        else:
            A, B, A_gt, B_gt = get_transform2(self.preprocess, A, B, A_gt, B_gt, self.load_size, self.crop_size)
        return A, A_gt, B, B_gt

    def __len__(self):
        return self.dataset_size


class SingleDataset(Dataset):
    def __init__(self, test_path):
        self.root = os.path.join(test_path, 'test')
        self.test_Atarget = os.path.join(test_path, 'testtarget')
        self.paths_A = os.listdir(self.root)
        self.boundingbox = []
        self.index_A = [0]
        for dir in self.paths_A:
            Atarget_path = os.path.join(self.test_Atarget, dir)
            Atarget_img = load_nifty_volume_as_array2(Atarget_path)
            boundingbox = get_ND_bounding_box(Atarget_img, 60)
            self.boundingbox.append(boundingbox)
            idex = boundingbox[1][0] - boundingbox[0][0] + self.index_A[-1]
            # idex = boundingbox[1]-boundingbox[0]+self.index_A[-1]
            self.index_A.append(idex)
        self.crop_size = 256
        self.load_size = 288
        self.test = self.index_A[-1]

    def __getitem__(self, index):
        index_A = index % self.test
        id_A = int(np.where((np.array(self.index_A)) <= index_A)[0].max())
        if (id_A == len(self.paths_A)): id_A = id_A - 1
        A_path = os.path.join(self.root, self.paths_A[id_A])
        slice_A = index_A - self.index_A[id_A]
        volume_A = load_nifty_volume_as_array2(A_path)
        volume_A = crop_ND_volume_with_bounding_box(volume_A, self.boundingbox[id_A][0], self.boundingbox[id_A][1])
        A = volume_A[slice_A]
        Atarget_path = os.path.join(self.test_Atarget, self.paths_A[id_A])
        volume_Atarget = load_nifty_volume_as_array2(Atarget_path)
        volume_Atarget = crop_ND_volume_with_bounding_box(volume_Atarget, self.boundingbox[id_A][0],
                                                          self.boundingbox[id_A][1])
        A_gt = volume_Atarget[slice_A]
        img, gt = get_test_transform2(A, A_gt, self.load_size, self.crop_size)
        # img = torch.from_numpy(img.copy()).unsqueeze(0).float()
        # gt = torch.from_numpy(gt.copy()).unsqueeze(0).float()
        return img, gt

    def __len__(self):
        return self.test


def cut_off_values_upper_lower_percentile(image, mask=None, percentile_lower=0.2, percentile_upper=99.8):
    '''if mask is None:
        mask = image != image[0,0,0]'''
    cut_off_lower = np.percentile(image.ravel(), percentile_lower)
    cut_off_upper = np.percentile(image.ravel(), percentile_upper)
    res = np.copy(image)
    res[(res < cut_off_lower)] = cut_off_lower
    res[(res > cut_off_upper)] = cut_off_upper
    return res


def get_params(preprocess, load_size, crop_size, size):
    w, h = size
    new_h = h
    new_w = w
    if preprocess == 'resize_and_crop':
        new_h = new_w = load_size
    elif preprocess == 'scale_width_and_crop':
        new_w = load_size
        new_h = load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def load_nifty_volume_as_array2(filename, with_header=False, is_label=False):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    The output array shape is like [Depth, Height, Width]
    inputs:
        filename: the input file name, should be *.nii or *.nii.gz
        with_header: return affine and hearder infomation
    outputs:
        data: a numpy data array
    """
    img = nibabel.load(filename)
    data = img.get_data()
    data = np.transpose(data, [2, 1, 0])
    if (with_header):
        return data, img.affine, img.header
    else:
        return data


def get_transform2(preprocess, A, B, A_gt, B_gt, load_size, crop_size, grayscale=False, method=Image.BICUBIC,
                   convert=True):
    B[B > 240] = 240
    B[B < -160] = -160

    # image[image>500] = -160
    # image[image>240] = 240
    # image[image<-160] = -160
    # mask = np.flip(mask,0)
    if 'random' in preprocess:
        r = random.uniform(0.8, 1.2)
    else:
        r = 1
    if 'resize' in preprocess:
        B = resize_3D_volume_to_given_shape(B, (load_size, load_size), 3)
        A = resize_3D_volume_to_given_shape(A, (load_size, load_size), 3)
        B_gt = resize_3D_volume_to_given_shape(B_gt, (load_size, load_size), 0)
        A_gt = resize_3D_volume_to_given_shape(A_gt, (load_size, load_size), 0)
    if 'flip' in preprocess:
        if random.random() < 0.5:
            A = np.flip(A, 0)
            B = np.flip(B, 0)
            A_gt = np.flip(A_gt, 0)
            B_gt = np.flip(B_gt, 0)
    B = (B - B.min()) / (B.max() - B.min())
    A = (A - A.min()) / (A.max() - A.min())
    if (B_gt.max() - B_gt.min() > 0):
        B_gt = (B_gt - B_gt.min()) / (B_gt.max() - B_gt.min())
    if (A_gt.max() - A_gt.min() > 0):
        A_gt = (A_gt - A_gt.min()) / (A_gt.max() - A_gt.min())

    A = A.astype(np.float32)

    B = B.astype(np.float32)

    A_gt = A_gt.astype(np.float32)
    B_gt = B_gt.astype(np.float32)
    if 'center_crop' in preprocess:
        A, A_gt = center_crop(crop_size, A, A_gt)
        B, B_gt = center_crop(crop_size, B, B_gt)
    if 'rand_crop' in preprocess:
        A, A_gt = rand_crop(crop_size, A, A_gt)
        B, B_gt = rand_crop(crop_size, B, B_gt)
    '''plt.imshow(A_gt)
    plt.show()'''
    if convert:
        image_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        A = image_transform(A)
        B = image_transform(B)
        mask_transform = transforms.Compose([transforms.ToTensor()])
        A_gt = mask_transform(A_gt)
        B_gt = mask_transform(B_gt)
    return A, B, A_gt, B_gt


def get_transform(preprocess, load_size, crop_size, no_flip, params=None, grayscale=True, method=Image.BICUBIC,
                  convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))  # 转换成灰度图
    if 'resize' in preprocess:
        osize = [load_size, load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, load_size, method)))

    if 'crop' in preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], crop_size)))

    if preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        '''if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]'''
    return transforms.Compose(transform_list)


def resize_3D_volume_to_given_shape(volume, out_shape, order=3):
    """
    resize an nd volume to a given shape
    inputs:
        volume: the input nd volume, an nd array
        out_shape: the desired output shape, a list
        order: the order of interpolation
    outputs:
        out_volume: the reized nd volume with given shape
    """
    shape0 = volume.shape
    assert (len(shape0) == len(out_shape))
    scale = [(out_shape[i] + 0.0) / shape0[i] for i in range(len(shape0))]
    out_volume = ndimage.interpolation.zoom(volume, scale, order=order)
    return out_volume


def get_val_transform2(preprocess, A, B, A_gt, B_gt, load_size, crop_size, grayscale=False, method=Image.BICUBIC,
                       convert=True):
    w, h = B.shape
    B[B > 240] = 240
    B[B < -160] = -160
    A_gt[A_gt == 63] = 255
    A_gt[A_gt != 255] = 0

    if 'resize' in preprocess:
        B = resize_3D_volume_to_given_shape(B, (load_size, load_size), 3)
        A = resize_3D_volume_to_given_shape(A, (load_size, load_size), 3)
        B_gt = resize_3D_volume_to_given_shape(B_gt, (load_size, load_size), 0)
        A_gt = resize_3D_volume_to_given_shape(A_gt, (load_size, load_size), 0)

    B = (B - B.min()) / (B.max() - B.min())
    A = (A - A.min()) / (A.max() - A.min())
    if (B_gt.max() - B_gt.min() > 0):
        B_gt = (B_gt - B_gt.min()) / (B_gt.max() - B_gt.min())
    if (A_gt.max() - A_gt.min() > 0):
        A_gt = (A_gt - A_gt.min()) / (A_gt.max() - A_gt.min())

    A = A.astype(np.float32)

    B = B.astype(np.float32)

    A_gt = A_gt.astype(np.float32)
    B_gt = B_gt.astype(np.float32)

    if 'crop' in preprocess:
        A, A_gt = center_crop(crop_size, A, A_gt)
        B, B_gt = center_crop(crop_size, B, B_gt)

    if convert:
        image_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        A = image_transform(A)
        B = image_transform(B)
        mask_transform = transforms.Compose([transforms.ToTensor()])
        A_gt = mask_transform(A_gt)
        B_gt = mask_transform(B_gt)
    return A, B, A_gt, B_gt


def get_test_transform2(A, A_gt, load_size, crop_size, grayscale=False, method=Image.BICUBIC, convert=True):
    w, h = A.shape
    A[A > 240] = 240
    A[A < -160] = -160
    A = resize_3D_volume_to_given_shape(A, (load_size, load_size), 3)
    A_gt = resize_3D_volume_to_given_shape(A_gt, (load_size, load_size), 0)
    A = (A - A.min()) / (A.max() - A.min())
    if (A_gt.max() - A_gt.min() > 0):
        A_gt = (A_gt - A_gt.min()) / (A_gt.max() - A_gt.min())

    A = A.astype(np.float32)

    A_gt = A_gt.astype(np.float32)

    A, A_gt = center_crop(crop_size, A, A_gt)

    image_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    A = image_transform(A)
    mask_transform = transforms.Compose([transforms.ToTensor()])
    A_gt = mask_transform(A_gt)
    return A, A_gt


def get_val_transform(preprocess, load_size, crop_size, params=None, grayscale=True, method=Image.BICUBIC,
                      convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))  # 转换成灰度图

    '''if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))'''

    osize = [load_size, load_size]
    transform_list.append(transforms.Resize(osize, method))
    if 'crop' in preprocess:
        transform_list.append(transforms.CenterCrop(crop_size))

    if convert:
        transform_list += [transforms.ToTensor()]
        '''if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]'''
    return transforms.Compose(transform_list)


def crop_ND_volume_with_bounding_box(volume, min_idx, max_idx):
    """
    crop/extract a subregion form an nd image.
    """
    dim = len(volume.shape)
    assert (dim >= 2 and dim <= 5)
    '''if(len([min_idx]) == 1):
        output = volume[np.ix_(range(min_idx, max_idx + 1))]'''
    if (dim == 2):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1))]
    elif (dim == 3):
        output = volume[np.ix_(range(min_idx[0], max_idx[0]),
                               range(min_idx[1], max_idx[1]),
                               range(min_idx[2], max_idx[2]))]
    elif (dim == 4):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1))]
    elif (dim == 5):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1),
                               range(min_idx[4], max_idx[4] + 1))]
    else:
        raise ValueError("the dimension number shoud be 2 to 5")
    return output


def get_ND_bounding_box(label, margin):
    """
    get the bounding box of the non-zero region of an ND volume
    """
    input_shape = label.shape
    if (type(margin) is int):
        margin = [margin] * len(input_shape)
    assert (len(input_shape) == len(margin))
    indxes = np.nonzero(label)
    # print(indxes)
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):  # de dao gan zang shape
        idx_min.append(indxes[i].min())
        idx_max.append(indxes[i].max())

    for i in range(1, len(input_shape)):  # xiang wai kuo zhang margin pixels, ru guo chao guo boudary, jiuyong boundary
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
    r = 0  # shang xia cai jian 1/3
    a = idx_min[0]
    b = idx_max[0]
    if (margin == 30):
        idx_min[0] = int((b - a) * r + a)
        idx_max[0] = int(b - (b - a) * r)
    else:
        idx_min[0] = int((b - a) * r + a)
    return idx_min, idx_max


def get_train_target_transform(preprocess, load_size, crop_size, no_flip, params=None, grayscale=True,
                               method=Image.NEAREST, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))  # 转换成灰度图
    if 'resize' in preprocess:
        osize = [load_size, load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, load_size, method)))

    if 'crop' in preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], crop_size)))

    if preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
    return transforms.Compose(transform_list)


def get_target_transform(preprocess, load_size, crop_size, grayscale=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))  # 转换成灰度图
    osize = [load_size, load_size]
    transform_list.append(transforms.Resize(osize, interpolation=Image.NEAREST))
    # transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=Image.NEAREST)))
    if 'crop' in preprocess:
        transform_list.append(transforms.CenterCrop(crop_size))
    transform_list += [transforms.ToTensor()]
    return transforms.Compose(transform_list)


def get_t_transform(opt, grayscale=False):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))  # 转换成灰度图
    transform_list += [transforms.ToTensor()]
    return transforms.Compose(transform_list)


# 将图片调整为四的倍数
def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


# 按照比例变成目标宽度
def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


# 剪裁
def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def center_crop(crop_size, image, label):
    w, h = image.shape
    if h < crop_size or w < crop_size:
        padh = crop_size - h if h < crop_size else 0
        padw = crop_size - w if w < crop_size else 0
        image = np.pad(image, ((padw // 2, padw - padw // 2), (padh // 2, padh - padh // 2)), 'constant',
                       constant_values=0)
        label = np.pad(label, ((padw // 2, padw - padw // 2), (padh // 2, padh - padh // 2)), 'constant',
                       constant_values=0)
        x1 = int(round((w + padw - crop_size) / 2.))
        y1 = int(round((h + padh - crop_size) / 2.))
        image = image[x1:x1 + crop_size, y1:y1 + crop_size]
        label = label[x1:x1 + crop_size, y1:y1 + crop_size]
    else:
        x1 = int(round((w - crop_size) / 2.))
        y1 = int(round((h - crop_size) / 2.))
        image = image[x1:x1 + crop_size, y1:y1 + crop_size]
        label = label[x1:x1 + crop_size, y1:y1 + crop_size]

    return image, label


def rand_crop(crop_size, image, label):
    h, w = image.shape

    new_h, new_w = label.shape
    x = random.randint(0, new_w - crop_size)
    y = random.randint(0, new_h - crop_size)

    image = image[y:y + crop_size, x:x + crop_size]
    label = label[y:y + crop_size, x:x + crop_size]

    return image, label


# 翻转
def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


# 输出提示信息
def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True


class UnpairedDataset2D(Dataset):
    # get unpaired dataset, such as MR-CT dataset
    def __init__(self, A_path, B_path, preprocess, phase):
        self.dir_A = os.path.join(A_path, phase)
        self.dir_Atarget = os.path.join(A_path, phase + 'target')
        self.dir_B = os.path.join(B_path, phase)
        self.dir_Btarget = os.path.join(B_path, phase + 'target')
        self.paths_A = sorted(make_dataset(self.dir_A))
        self.paths_B = sorted(make_dataset(self.dir_B))
        self.Atarget_paths = sorted(make_dataset(self.dir_Atarget))
        self.Btarget_paths = sorted(make_dataset(self.dir_Btarget))
        self.Asize = len(self.paths_A)
        self.Bsize = len(self.paths_B)
        self.dataset_size = max(self.Asize, self.Bsize)
        self.preprocess = preprocess
        self.crop_size = 256
        self.load_size = 288
        self.phase = phase

    def __getitem__(self, index):
        A_path = self.paths_A[index % self.Asize]
        index_B = random.randint(0, self.Bsize - 1)
        B_path = self.paths_B[index_B]
        A = Image.open(A_path).convert('L')
        B = Image.open(B_path).convert('L')
        Atarget_path = self.Atarget_paths[index % self.Asize]
        A_gt = Image.open(Atarget_path).convert('L')
        Btarget_path = self.Btarget_paths[index_B]
        B_gt = Image.open(Btarget_path).convert('L')
        if self.phase == 'val' or self.phase == 'test':
            self.transform_A = get_val_transform(self.preprocess, self.load_size, self.crop_size)
            self.transform_B = get_val_transform(self.preprocess, self.load_size, self.crop_size)
            self.transform_targetA = get_target_transform(self.preprocess, self.load_size, self.crop_size)
            self.transform_targetB = get_target_transform(self.preprocess, self.load_size, self.crop_size)
        else:
            transform_params = get_params(self.preprocess, self.load_size, self.crop_size, A.size)
            transform_param = get_params(self.preprocess, self.load_size, self.crop_size, B.size)
            self.transform_A = get_transform(self.preprocess, self.load_size, self.crop_size, False, transform_params)
            self.transform_B = get_transform(self.preprocess, self.load_size, self.crop_size, False, transform_param)
            self.transform_targetA = get_train_target_transform(self.preprocess, self.load_size, self.crop_size, False,
                                                                transform_params)
            self.transform_targetB = get_train_target_transform(self.preprocess, self.load_size, self.crop_size, False,
                                                                transform_param)
        A = self.transform_A(A)
        B = self.transform_B(B)
        A_gt = self.transform_targetA(A_gt)
        B_gt = self.transform_targetB(B_gt)
        # plt.imshow(B[0])
        # plt.show()
        # plt.imshow(B_gt[0])
        # plt.show()
        return A, A_gt, B, B_gt

    def __len__(self):
        return self.dataset_size


# 判断文件是否为规定图片
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)  # any()判断对象是否为空


# 读取图片地址
def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
