"""

"""
import os
from shutil import copy

import nibabel as nib

base_dir = r'G:\Dataset\hx\reg_seg\202211123\o_dataset'
save_dir = r'G:\Dataset\hx\reg_seg\202211123\original'

os.makedirs(os.path.join(save_dir, 'CT', 'images'), exist_ok=True)
os.makedirs(os.path.join(save_dir, 'CT', 'gtv_labels'), exist_ok=True)
os.makedirs(os.path.join(save_dir, 'CT', 'bone_labels'), exist_ok=True)
os.makedirs(os.path.join(save_dir, 'MRI', 'images'), exist_ok=True)
os.makedirs(os.path.join(save_dir, 'MRI', 'bone_labels'), exist_ok=True)
os.makedirs(os.path.join(save_dir, 'MRI', 'tumor_labels'), exist_ok=True)

# forbidding_ids = ['0017603443', '0020139089', '0017301962']
forbidding_ids = []
result = []
for name in os.listdir(os.path.join(base_dir, 'CT', 'gtv_labels')):
    id_, _ = name.split('_')
    if id_ not in forbidding_ids:

        copy(os.path.join(base_dir, 'CT', 'gtv_labels', name), os.path.join(save_dir, 'CT', 'gtv_labels', name))

        copy(os.path.join(base_dir, 'CT', 'images', name.replace('gtv', 'images')),
             os.path.join(save_dir, 'CT', 'images', name.replace('gtv', 'image')))

        copy(os.path.join(base_dir, 'CT', 'bone_labels', name.replace('gtv', 'Bones')),
             os.path.join(save_dir, 'CT', 'bone_labels', name.replace('gtv', 'bone')))

        copy(os.path.join(base_dir, 'MRI', 'bone_labels', name.replace('gtv', 'bone')),
             os.path.join(save_dir, 'MRI', 'bone_labels', name.replace('gtv', 'bone')))

        copy(os.path.join(base_dir, 'MRI', 'images', name.replace('gtv', 'image')),
             os.path.join(save_dir, 'MRI', 'images', name.replace('gtv', 'image')))

        copy(os.path.join(base_dir, 'MRI', 'tumor_labels', name.replace('gtv', 'tumor')),
             os.path.join(save_dir, 'MRI', 'tumor_labels', name.replace('gtv', 'tumor')))

        result += [
            os.path.join(save_dir, 'CT', 'gtv_labels', name),
            os.path.join(save_dir, 'CT', 'images', name.replace('gtv', 'image')),
            os.path.join(save_dir, 'CT', 'bone_labels', name.replace('gtv', 'bone')),

            os.path.join(save_dir, 'MRI', 'bone_labels', name.replace('gtv', 'bone')),
            os.path.join(save_dir, 'MRI', 'images', name.replace('gtv', 'image')),
            os.path.join(save_dir, 'MRI', 'tumor_labels', name.replace('gtv', 'tumor'))
        ]

for item in result:
    img = nib.load(item)
    qform = img.get_qform()
    img.set_qform(qform)
    sfrom = img.get_sform()
    img.set_sform(sfrom)
    nib.save(img, item)
    print(item)
