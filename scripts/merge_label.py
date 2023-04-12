from glob import glob
import os
import shutil
import SimpleITK as sitk
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# from random import random
import random
from random import shuffle


random.seed(4092)


def make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


make_dir('trainA')
make_dir('trainB')
make_dir('testA')
make_dir('testB')

make_dir('trainA_gt')
make_dir('trainB_gt')
make_dir('testA_gt')
make_dir('testB_gt')

test_list_now = os.listdir('./CT/labels')
test_list = os.listdir('/home/cas/home_ez/Datasets/HX_CT2MR/2.16/1-sure')
test_list = [x.replace('-1', '') for x in test_list]
test_list = list(set(test_list_now) & set(test_list))
shuffle(test_list)
test_list = test_list[:12]

ct_path = "./CT/images/*.nii.gz"
mr_path = "./MR/images/*.nii.gz"

for path in tqdm(glob(ct_path)):
    name = os.path.basename(path)
    if name not in test_list:
        out_path = os.path.join('trainA', name)
        shutil.copy(path, out_path)

        if os.path.exists(path.replace('images', 'labels')):
            out_path = os.path.join('trainA_gt', name)
            shutil.copy(path.replace('images', 'labels'), out_path)
    else:
        out_path = os.path.join('testA', name)
        shutil.copy(path, out_path)

        out_path = os.path.join('testA_gt', name)
        shutil.copy(path.replace('images', 'labels'), out_path)

for path in tqdm(glob(mr_path)):
    name = os.path.basename(path)
    if name not in test_list:
        out_path = os.path.join('trainB', name)
        shutil.copy(path, out_path)

        out_path = os.path.join('trainB_gt', name)
        shutil.copy(path.replace('images', 'labels'), out_path)
    else:
        out_path = os.path.join('testB', name)
        shutil.copy(path, out_path)

        out_path = os.path.join('testB_gt', name)
        shutil.copy(path.replace('images', 'labels'), out_path)
