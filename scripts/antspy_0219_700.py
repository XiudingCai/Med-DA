import os
import shutil

import ants
import numpy as np
from tqdm import tqdm
from glob import glob
import time

root = "/home/cas/home_ez/Datasets/CT2MR_Reg"
root_save = "/home/cas/home_ez/Datasets/CT2MR_Reg/original_700"

# FIX_paths = glob(os.path.join(root, 'original_cat', 'CT', 'images', '*'))
# MOV_paths = glob(os.path.join(root, 'original_cat', 'MR', 'images', '*'))
FIX_paths = glob(os.path.join(root, 'original', 'CT', 'bones', '*0001643700.nii.gz'))
MOV_paths = glob(os.path.join(root, 'original', 'MR', 'bones', '*0001643700.nii.gz'))

MR_paths = glob(os.path.join(root, 'original', 'MR', 'images', '*0001643700.nii.gz'))
MRb_paths = glob(os.path.join(root, 'original', 'MR', 'bones', '*0001643700.nii.gz'))
MRt_paths = glob(os.path.join(root, 'original', 'MR', 'tumor', '*0001643700.nii.gz'))

transforms = ['TRSAA', 'Translation', 'Rigid', 'Similarity', 'QuickRigid', 'DenseRigid', 'BOLDRigid',
              'Affine', 'AffineFast', 'BOLDAffine']

for transform in transforms:
    idx = 0
    fix_path, mov_path = FIX_paths[0], MOV_paths[0]
    fix_img = ants.image_read(fix_path)
    print(fix_path, transform)

    mov_img = ants.image_read(mov_path)

    if transform == 'TRSAA':
        out = ants.registration(fixed=fix_img, moving=mov_img,
                                type_of_transform='TRSAA', reg_iterations=(40, 20, 0))
    else:
        out = ants.registration(fixed=fix_img, moving=mov_img, type_of_transform=transform)


    # reg_img = out['warpedmovout']  # 获取配准结果

    def apply_t(x, label=False):
        img_to_reg = ants.image_read(x)

        if label:
            moving = ants.apply_transforms(fixed=fix_img, moving=img_to_reg,
                                           transformlist=out['fwdtransforms'], interpolator='nearestNeighbor')
        else:
            moving = ants.apply_transforms(fixed=fix_img, moving=img_to_reg,
                                           transformlist=out['fwdtransforms'])

        save_path = f"/home/cas/home_ez/Datasets/CT2MR_Reg/original_700/MR_0001643700_{transform}.nii.gz"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ants.image_write(moving, save_path)

    apply_t(MR_paths[idx])
    # apply_t(MRb_paths[idx])
    # apply_t(MRt_paths[idx], label=True)

shutil.copy(os.path.join(root, 'original', 'CT', 'images', '0001643700.nii.gz'),
            (os.path.join(root, 'original_700', 'CT_0001643700.nii.gz')))
