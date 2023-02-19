"""

"""
import os
from shutil import copy
import SimpleITK as sitk
from utils.crop import Crop


def cropping(path, save_path, params):
    min_z, len_z, min_x, len_x, min_y, len_y = params
    img = sitk.ReadImage(path)
    img_data = sitk.GetArrayFromImage(img)
    cropped_img = img_data[min_z:min_z + len_z, min_x:min_x + len_x, min_y:min_y + len_y]
    crop_img = sitk.GetImageFromArray(cropped_img)
    crop_img.SetOrigin(img.GetOrigin())
    crop_img.SetSpacing(img.GetSpacing())
    crop_img.SetDirection(img.GetDirection())

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sitk.WriteImage(crop_img, save_path)
    print('saving', save_path)


base_dir = r'G:\Dataset\hx\reg_seg\20221212_nct&others_spacing112\original_affine'
save_dir = r'G:\Dataset\hx\reg_seg\20221212_nct&others_spacing112\original_affine_crop'
for name in os.listdir(os.path.join(base_dir, 'CT', 'bones')):
    base_data = sitk.GetArrayFromImage(
        sitk.ReadImage(os.path.join(base_dir, 'MRI', 'images', name)))

    crop_params = Crop.crop(base_data)

    cropping(os.path.join(base_dir, 'CT', 'images', name),
             os.path.join(save_dir, 'CT', 'images', name), crop_params)

    cropping(os.path.join(base_dir, 'CT', 'bones', name),
             os.path.join(save_dir, 'CT', 'bones', name), crop_params)

    cropping(os.path.join(base_dir, 'MRI', 'images', name),
             os.path.join(save_dir, 'MRI', 'images', name), crop_params)

    cropping(os.path.join(base_dir, 'MRI', 'bones', name),
             os.path.join(save_dir, 'MRI', 'bones', name), crop_params)
