import os
from shutil import copy
import SimpleITK as sitk
from glob import glob
import pandas as pd
import numpy as np
import pandas_profiling as pp

ct_list = glob(f"/home/cas/home_ez/Datasets/CT2MR_Reg/original/CT/images/*")

columns = []
data = []

for idx, path in enumerate(ct_list):
    if idx == 0:
        # 动态创建df
        img = sitk.ReadImage(path)

        img_size = img.GetSize()
        for i in range(len(img_size)):
            columns.append(f"size_{i}")

        img_origin = img.GetOrigin()
        for i in range(len(img_origin)):
            columns.append(f"origin_{i}")

        img_spacing = img.GetSpacing()
        for i in range(len(img_spacing)):
            columns.append(f"spacing_{i}")

        # img_direction = img.GetDirection()
        columns.append('direction')
        columns.append('data_min')
        columns.append('data_max')
        columns.append('data_mean')
        columns.append('data_median')
        columns.append('pid')

        # df = pd.DataFrame(columns=columns)
    # else:

    img = sitk.ReadImage(path)
    row = []

    img_size = img.GetSize()
    for x in img_size:
        row.append(x)
    if row[-2] == 296:
        print(path)
    img_origin = img.GetOrigin()
    for x in img_origin:
        row.append(x)

    img_spacing = img.GetSpacing()
    for x in img_spacing:
        row.append(x)

    img_direction = img.GetDirection()
    row.append(str(img_direction[0]))

    img_data = sitk.GetArrayFromImage(img)
    data_min, data_max, data_mean, data_median = np.min(img_data), np.max(img_data), np.mean(img_data), np.median(
        img_data)
    row += [data_min, data_max, data_mean, data_median, path]

    data.append(row)

df = pd.DataFrame(columns=columns, data=data)

######################################################
# 一键分析
######################################################
# 进行分析
print(columns)
print(np.sum(df['size_1'] == 496))
