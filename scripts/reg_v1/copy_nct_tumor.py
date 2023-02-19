# import os
# import shutil
#
# tumor_files = os.listdir(r'G:\Dataset\hx\reg_seg\20221212_nct&others_spacing112\original_same_spacing\MRI\tumor')
#
# for item in os.listdir(r'G:\Dataset\hx\reg_seg\20221212_nct&others_spacing112\original_same_spacing\MRI\images'):
#     if item not in tumor_files:
#         shutil.copy(
#             os.path.join(r'G:\project\reg_seg\code\preprocess\nct_dataset\MRI\tumor', item),
#             os.path.join(r'G:\Dataset\hx\reg_seg\20221212_nct&others_spacing112\original_same_spacing\MRI\tumor', item)
#         )
#         print(item)
import os

tumor_path =r'G:\Dataset\hx\reg_seg\20221212_nct&others_spacing112\original_same_spacing\MRI\tumor'
mri_path = r'G:\Dataset\hx\reg_seg\20221212_nct&others_spacing112\original_same_spacing\MRI\images'
for item in os.listdir(tumor_path):
    if item not in os.listdir(mri_path):
        os.remove(os.path.join(tumor_path, item))
