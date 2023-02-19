from glob import glob
import os
import shutil
import SimpleITK as sitk
from sklearn.model_selection import train_test_split

def make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    
ct_path = "./CT/images/*"
mr_path = "./MRI/images/*"

train_A, test_A = train_test_split(glob(ct_path), test_size=0.2, random_state=1)
train_B, test_B = train_test_split(glob(mr_path), test_size=0.2, random_state=1)

make_dir('trainA')
make_dir('trainB')
make_dir('testA')
make_dir('testB')

for path in train_A:
    name = os.path.basename(path)
    out_path = os.path.join('trainA', name)
    shutil.copy(path, out_path)

for path in train_B:
    name = os.path.basename(path)
    out_path = os.path.join('trainB', name)
    shutil.copy(path, out_path)
    
for path in test_A:
    name = os.path.basename(path)
    out_path = os.path.join('testA', name)
    shutil.copy(path, out_path)
    
for path in test_B:
    name = os.path.basename(path)
    out_path = os.path.join('testB', name)
    shutil.copy(path, out_path)
    
x, y, z = [], [], []
for path_A, path_B in zip(train_A, train_B):
    img_A = sitk.ReadImage(path_A)
    # print(img_A.GetSize())
    
    img_B = sitk.ReadImage(path_B)
    # print(img_B.GetSize())
    
    assert img_A.GetSize() == img_B.GetSize()
    
    x.append(img_A.GetSize()[0])
    y.append(img_A.GetSize()[1])
    z.append(img_A.GetSize()[2])
    
print('min(x), min(y), min(z)', min(x), min(y), min(z))