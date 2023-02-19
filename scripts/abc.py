from glob import glob
import os
import shutil
import SimpleITK as sitk
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from random import random
from random import shuffle


# random.seed(42)


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

test_list = os.listdir('./CT/tumor')
shuffle(test_list)
test_list = test_list[:9]

ct_path = "./CT/images/*.nii.gz"
mr_path = "./MRI/images/*.nii.gz"

for path in tqdm(glob(ct_path)):
    name = os.path.basename(path)
    if name not in test_list:
        out_path = os.path.join('trainA', name)
        shutil.copy(path, out_path)

    # out_path = os.path.join('trainA_gt', name)
    # shutil.copy(path.replace('images', 'tumor'), out_path)
    else:
        out_path = os.path.join('testA', name)
        shutil.copy(path, out_path)

        out_path = os.path.join('testA_gt', name)
        shutil.copy(path.replace('images', 'tumor'), out_path)

for path in tqdm(glob(mr_path)):
    name = os.path.basename(path)
    if name not in test_list:
        out_path = os.path.join('trainB', name)
        shutil.copy(path, out_path)

        out_path = os.path.join('trainB_gt', name)
        shutil.copy(path.replace('images', 'tumor'), out_path)
    else:
        out_path = os.path.join('testB', name)
        shutil.copy(path, out_path)

        out_path = os.path.join('testB_gt', name)
        shutil.copy(path.replace('images', 'tumor'), out_path)
