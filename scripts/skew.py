import numpy as np
import SimpleITK as sitk


def skew_and_crop_into_a_cube(in_data, axis=0, sampling_start=0.3, sampling_end=0.7, ratio=1., verbose=True):
    """
    in_data: input data, numpy array
    axis: the axis of summation
    """
    out_data = []
    if not isinstance(in_data, list):
        in_data = [in_data]
    # crop black
    z, y, x = np.where(in_data[0] > 0)
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)
    min_z, max_z = min(z), max(z)
    for i in range(len(in_data)):
        data = in_data[i][min_z: max_z + 1, min_y: max_y + 1, min_x: max_x + 1]

        alphabet = "abcdefghijklmnopqrstxyz"
        exp_swap = f"{alphabet[:len(data.shape)]}->{alphabet[axis] + alphabet[:axis] + alphabet[axis + 1: len(data.shape)]}"
        exp_rec = f"{alphabet[:len(data.shape)]}->{alphabet[1:axis + 1] + alphabet[0] + alphabet[axis + 1: len(data.shape)]}"
        data_swap = np.einsum(exp_swap, data)

        exp_sum = f"{alphabet[:len(data.shape)]}->{alphabet[0]}"
        data_vec = np.einsum(exp_sum, data_swap)

        # selection by min-max
        diff = data_vec[1:] - data_vec[:-1]
        start = int(len(diff) * sampling_start)
        end = int(len(diff) * sampling_end)

        idx_max = np.where(diff > diff[start: end].max() * ratio)
        idx_min = np.where(diff < diff[start: end].min() * ratio)

        # selection by mean
        idx_mean = np.where(data_vec[:-1] < data_vec[start: end].mean() * 0.5)
        idx_all = np.sort(np.array(list(set(np.concatenate([idx_max, idx_min, idx_mean], axis=1)[0].tolist()))))
        # print(idx_all)

        if idx_all.shape[0] != 0:
            if idx_all[0] == 0:
                for i in range(idx_all.shape[0]):
                    if i != idx_all[i]:
                        start = i
                        break
            else:
                start = None
            if idx_all[-1] == data_swap.shape[0] - 1 - 1:
                for i in range(1, idx_all.shape[0] + 1):
                    # print(i)
                    if idx_all[-i] != idx_all[-i - 1] + 1:
                        end = -i
                        break
            else:
                end = None
            if verbose:
                print(start, end, idx_all)

            data_crop = data_swap[start: end]
            # print(data_crop.shape)
            data = np.einsum(exp_rec, data_crop)
        else:
            data = data
        out_data.append(data)
        # print(in_data.shape, data_out.shape)
    return out_data


mr_path = "/home/cas/home_ez/Datasets/CT2MR_Reg/original_700/MR_0001643700.nii.gz"
save_path = "/home/cas/home_ez/Datasets/CT2MR_Reg/original_700/MR_0001643700_skew.nii.gz"

nii = sitk.ReadImage(mr_path)
nii_data = sitk.GetArrayFromImage(nii)
out_data = skew_and_crop_into_a_cube(nii_data, axis=0, sampling_start=0.3, sampling_end=0.7, ratio=1)[0]

nii_new = sitk.GetImageFromArray(out_data)
nii_new.SetOrigin(nii.GetOrigin())
nii_new.SetSpacing(nii.GetSpacing())
nii_new.SetDirection(nii.GetDirection())

import os

os.makedirs(os.path.dirname(save_path), exist_ok=True)
sitk.WriteImage(nii_new, save_path)
print('saving', save_path)
