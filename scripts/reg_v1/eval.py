"""
    评估配准后的结果
"""
import os

import torch
import numpy as np
import SimpleITK as sitk
from utils import resample
from reg_eval import Evaluation


def normalization(img):
    """
        归一化
    :param img: numpy 数组
    :return: 归一化后img, max(img), min(img)
    """
    img_max = np.max(img)
    img_min = np.min(img)
    img_data = (img - img_min) / (img_max - img_min + 1e-5)
    return img_data, img_max, img_min


def get_img(path, resize=None):
    img = sitk.ReadImage(path)
    if resize:
        img = resample(img, new_size=resize)
    img_data = sitk.GetArrayFromImage(img).astype('float32')  # z, x, y
    img_data = img_data.transpose(1, 2, 0)
    img_data, _, _ = normalization(img_data)
    img_data = img_data[np.newaxis, np.newaxis, ...]
    return img_data, img


if __name__ == '__main__':
    base_paths = [
        # r'G:\Dataset\hx\reg_seg\20221022_same_spacing',
        # r'G:\Dataset\hx\reg_seg\20221022_affine',
        # r'G:\Dataset\hx\reg_seg\20221022_affine_crop',
        r'G:\Dataset\hx\reg_seg\20221022_affine_crop_TVMSQC',
    ]
    for base_path in base_paths:

        os.makedirs('./result/', exist_ok=True)
        output_path = './result/%s.csv' % os.path.basename(base_path)

        metrics = ['ncc', 'mi', 'ssim3d', 'mse']
        device = 'cpu'
        resize = (512, 512, 64)

        evaluater = Evaluation(metrics=metrics)
        result = np.zeros(len(metrics))
        for name in os.listdir(os.path.join(base_path, 'CT', 'images')):
            print('calculating ', name)
            fixed_path = os.path.join(base_path, 'CT', 'images', name)
            moving_path = os.path.join(base_path, 'MRI', 'images', name)

            fixed_img, _ = get_img(fixed_path, resize=resize)
            moving_img, _ = get_img(moving_path, resize=resize)

            fixed_img_t, moving_img_t = torch.from_numpy(fixed_img), torch.from_numpy(moving_img)
            result += evaluater.eval(fixed_img_t, moving_img_t).astype('float32')

        result /= len(os.listdir(os.path.join(base_path, 'CT', 'images')))
        print('\t'.join(['%s: %.4f' % (item, result[i]) for i, item in enumerate(metrics)]))

        # TO file

        output_format = '%s' + ',%.4f' * len(metrics)

        with open(output_path, 'w', encoding='utf-8') as f:
            file_lines = [','.join(['type'] + metrics) + '\n']
            # for i, line in enumerate(result):
            file_lines.append(output_format % tuple(['MRI2CT'] + result.tolist()) + '\n')
            f.writelines(file_lines)
            print('Result file saved to :', output_path)
