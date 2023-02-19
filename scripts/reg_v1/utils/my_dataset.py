import os
import numpy as np
import SimpleITK as sitk
import torch
# import my_config as cfg
# import utils.img_tools as img_tools
from torch.utils.data import Dataset


def get_img(img_path):
    # print(img_path)
    data = sitk.ReadImage(img_path)
    # 获取体素间距
    spacing = np.array(data.GetSpacing())
    img_data = sitk.GetArrayFromImage(data)
    direction = data.GetDirection()
    origin = data.GetOrigin()
    # 获取原图大小
    size = np.array(data.GetSize())
    # 重采样到指定的大小
    # img_data, new_spacing = img_tools.image_resample(img_data, size, cfg.IMAGE_SHAPE, spacing, direction, origin)
    new_spacing = spacing
    img_data = img_data.transpose([1, 2, 0])
    # 归一化
    img_data, img_max, img_min = img_tools.normalization(img_data)
    # img_max = np.max(img_data)
    # img_min = np.min(img_data)
    # print(size)
    img_data = img_data[np.newaxis, :, :, :].astype('float32')
    # print(img_max, img_min, img_data.shape)
    img_data = torch.from_numpy(img_data)

    return img_data, (img_max, img_min, size, spacing, direction, origin, new_spacing)


class Dalian_Dataset(Dataset):
    def __init__(self, img_path, fixed_modal='D', moving_modal='A'):
        # 先按照病例找到所有人的图像，然后求解所有可能性
        self.images = []
        for item in os.listdir(img_path):
            self.images.append(
                (
                    os.path.join(img_path, item, '%s_%s.nii.gz' % (item, fixed_modal)),
                    os.path.join(img_path, item, '%s_%s.nii.gz' % (item, moving_modal)),
                )
            )

    def __getitem__(self, index):
        fi_path, mi_path = self.images[index]

        fi, _ = get_img(os.path.join(fi_path))
        mi, _ = get_img(os.path.join(mi_path))
        return fi, mi

    def __len__(self):
        return len(self.images)


class Dalian_Dataset_test(Dataset):
    def __init__(self, img_path, fixed_modal='D', moving_modal='A'):
        # 先按照病例找到所有人的图像，然后求解所有可能性
        self.images = []
        for item in os.listdir(img_path):
            self.images.append(
                (
                    os.path.join(img_path, item, '%s_%s.nii.gz' % (item, fixed_modal)),
                    os.path.join(img_path, item, '%s_%s.nii.gz' % (item, moving_modal)),
                )
            )

    def __getitem__(self, index):
        fi_path, mi_path = self.images[index]

        fi, _ = get_img(os.path.join(fi_path))
        mi, info = get_img(os.path.join(mi_path))
        return fi, mi, info, mi_path

    def __len__(self):
        return len(self.images)
