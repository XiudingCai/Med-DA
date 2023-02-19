import os
base_dir = r'G:\Dataset\hx\reg_seg\202211123\original_same_spacing'

for root, dir_, files in os.walk(base_dir):
    # if len(files) > 0:
    for file in files:
        id_, date = file.split('_')
        os.rename(os.path.join(root, file), os.path.join(root, '%s.nii.gz' % id_))
        print(os.path.join(root, file))
