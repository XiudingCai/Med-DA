from torch.utils.data import Dataset
import numpy as np
import os
import torch
import random
import configparser
from PIL import Image
import math
import os.path as osp
from torch.nn import init
from scipy import ndimage
import SimpleITK as sitk
import torch.nn.functional as F
import torch.nn as nn


def winadj_mri(array, data_type="mri"):
    if data_type == "mri":
        array[array != 63] = 0
        array[array == 63] = 255
    # D,H,W = array.shape
    # zoom = [1,384/H,384/W]
    # img=ndimage.zoom(img,zoom,order=3)
    # lab=ndimage.zoom(lab,zoom,order=0)
    # v0 = np.percentile(array, 1)
    # v1 = np.percentile(array, 99)
    # array[array < v0] = v0    
    # array[array > v1] = v1  
    # v0 = array.min() 
    # v1 = array.max() 
    # array[array < v0] = v0
    # array[array > v1] = v1
    # array = (array - v0) / (v1 - v0) * 2.0 - 1.0

    return array


dir = '/media/disk8t_/yqk/wenjian1/MRI/testtarget'
names = os.listdir(dir)
save_dir_ = "/media/disk8t_/yqk/wenjian/MRI/testtarget"
for name in names:
    nii_dir = os.path.join(dir, name)
    # print(nii_dir)
    img_obj = sitk.ReadImage(nii_dir)
    A = sitk.GetArrayFromImage(img_obj) / 1
    A = winadj_mri(A, data_type='mri')
    A = sitk.GetImageFromArray(A)
    # print(type(A))
    save_dir = os.path.join(save_dir_, name)
    print(save_dir)
    sitk.WriteImage(A, save_dir)
