import os
import shutil

import ants
import numpy as np
from tqdm import tqdm
from glob import glob
import time

root = "/home/cas/home_ez/Datasets/CT2MR_Reg"
root_save = "/home/cas/home_ez/Datasets/CT2MR_Reg/original_TRSAA"

# FIX_paths = glob(os.path.join(root, 'original_cat', 'CT', 'images', '*'))
# MOV_paths = glob(os.path.join(root, 'original_cat', 'MR', 'images', '*'))
FIX_paths = glob(os.path.join(root, 'original', 'CT', 'bones', '*'))
MOV_paths = glob(os.path.join(root, 'original', 'MR', 'bones', '*'))

MR_paths = glob(os.path.join(root, 'original', 'MR', 'images', '*'))
MRb_paths = glob(os.path.join(root, 'original', 'MR', 'bones', '*'))
MRt_paths = glob(os.path.join(root, 'original', 'MR', 'tumor', '*'))

transform = 'TRSAA'

for idx, (fix_path, mov_path) in tqdm(enumerate(zip(FIX_paths, MOV_paths))):
    print(fix_path)
    fix_img = ants.image_read(fix_path)

    mov_img = ants.image_read(mov_path)

    if transform == 'TRSAA':
        out = ants.registration(fixed=fix_img, moving=mov_img,
                                type_of_transform='TRSAA', reg_iterations=(40, 20, 0))
    else:
        out = ants.registration(fixed=fix_img, moving=mov_img, type_of_transform='TRSAA')

    # reg_img = out['warpedmovout']  # 获取配准结果

    def apply_t(x, label=False):
        img_to_reg = ants.image_read(x)

        if label:
            moving = ants.apply_transforms(fixed=fix_img, moving=img_to_reg,
                                           transformlist=out['fwdtransforms'], interpolator='nearestNeighbor')
        else:
            moving = ants.apply_transforms(fixed=fix_img, moving=img_to_reg,
                                           transformlist=out['fwdtransforms'])

        save_path = x.replace('original', 'original_TRSAA')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ants.image_write(moving, save_path)

    apply_t(MR_paths[idx])
    apply_t(MRb_paths[idx])
    apply_t(MRt_paths[idx], label=True)


shutil.copytree(os.path.join(root, 'original', 'CT'),
                os.path.join(root_save, 'CT'))
