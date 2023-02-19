import os
from utils import resample
import SimpleITK as sitk

base_dir = r'G:\Dataset\hx\reg_seg\20221212_nct&others_spacing112\original_same_spacing\MRI\tumor'
base_save_path = r'G:\Dataset\hx\reg_seg\20221212_nct&others_spacing112\original_same_spacing\MRI\tumor112'

split_symbol = '\\'
for root, dir_, files in os.walk(base_dir):
    if len(files) > 0:
        for file in files:
            save_path = os.path.join(base_save_path, split_symbol.join(root.split(split_symbol)[-2:]))
            if os.path.exists(os.path.join(save_path, file)):
                continue
            print(save_path, file)

            new_img = resample(sitk.ReadImage(os.path.join(root, file)), new_spacing=(1, 1, 2))
            if 'image' not in file:
                new_img = resample(sitk.ReadImage(os.path.join(root, file)), new_spacing=(1, 1, 2), nearest=True)
            os.makedirs(save_path, exist_ok=True)
            sitk.WriteImage(new_img, os.path.join(save_path, file))

