import shutil
import os
from glob import glob

path_a = "/home/yht/Casit/ez/GenRSeg/results/MR2CT_affine_crop_png_cutndv2_ep5k/test_latest/images/real_A_png"
path_b = "/home/yht/Casit/ez/GenRSeg/results/MR2CT_affine_crop_png_cutndv2_ep5k/test_latest/images/real_B_png"

path_ref = "/home/yht/Casit/Datasets/ez/datasets/MR2CT_Reg/MR2CT_affine_crop"
path_save = "/home/yht/Casit/Datasets/ez/datasets/MR2CT_Reg_PNG"

os.makedirs(os.path.join(path_save, 'trainA'), exist_ok=True)
os.makedirs(os.path.join(path_save, 'trainB'), exist_ok=True)
os.makedirs(os.path.join(path_save, 'testA'), exist_ok=True)
os.makedirs(os.path.join(path_save, 'testB'), exist_ok=True)

for name in os.listdir(path_a):
    nam = name.split('_')[0]
    if nam in str(os.listdir(os.path.join(path_ref, 'trainA'))):
        shutil.copy(os.path.join(path_a, name), os.path.join(path_save, 'trainA', name))
    else:
        shutil.copy(os.path.join(path_a, name), os.path.join(path_save, 'testA', name))

for name in os.listdir(path_b):
    nam = name.split('_')[0]
    if nam in str(os.listdir(os.path.join(path_ref, 'trainA'))):
        shutil.copy(os.path.join(path_b, name), os.path.join(path_save, 'trainB', name))
    else:
        shutil.copy(os.path.join(path_b, name), os.path.join(path_save, 'testB', name))

# for path in glob(os.path.join(path_ref, 'trainA')):

