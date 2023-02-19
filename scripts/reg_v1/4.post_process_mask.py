import os
import SimpleITK as sitk

base_path = r'G:\Dataset\hx\reg_seg\202211123\original_rigid_crop_syn\MRI\tumor_labels'
save_path = r'G:\Dataset\hx\reg_seg\202211123\original_rigid_crop_syn\MRI\tumor_labels_post'
os.makedirs(save_path, exist_ok=True)
for file in os.listdir(base_path):
    img = sitk.ReadImage(os.path.join(base_path, file))
    data = sitk.GetArrayFromImage(img)
    data[data > 0.5] = 1
    data[data < 0.5] = 0
    new_img = sitk.GetImageFromArray(data)
    new_img.SetOrigin(img.GetOrigin())
    new_img.SetSpacing(img.GetSpacing())
    new_img.SetDirection(img.GetDirection())

    sitk.WriteImage(new_img, os.path.join(save_path, file))
    print(file)
