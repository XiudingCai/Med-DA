import numpy as np
import SimpleITK as sitk
from SimpleITK import ResampleImageFilter


def resample(sitk_img, new_spacing=None, new_size=None, z_spacing=None, nearest=False):
    """
        重采样接口
        model表示使用spacing作为标准还是img size作为标准
    :param sitk_img: simpleitk类的图像变量
    :param new_spacing: 新的体素大小，默认None
    :param new_size: 新的图像大小，默认None
    :param z_spacing: 新的z轴体素大小，x\y 保持一致，默认None
    :return: resampled simple itk image
    """

    # 获取原图信息
    size = np.array(sitk_img.GetSize())
    spacing = np.array(sitk_img.GetSpacing())
    new_size_refine = None
    new_spacing_refine = None
    if new_spacing:
        try:
            len_spacing = len(new_spacing)
            if len_spacing < 3:
                raise Exception('错误，输入体素必须为三个维度的空间体素大小')
            # 进行空间重采样的参数设置
            new_spacing_refine = [float(item) for item in new_spacing]
            # 根据当前图像指定的体素大小计算出图像大小
            new_size_refine = size * spacing / new_spacing_refine
            new_size_refine = [int(item) for item in new_size_refine]
        except Exception:
            raise Exception('错误，输入体素必须为三个维度的空间体素大小')
    elif z_spacing:
        # 进行空间重采样的参数设置
        new_spacing_refine = [spacing[0], spacing[1], z_spacing]
        # 根据当前图像指定的体素大小计算出图像大小
        new_size_refine = size * spacing / new_spacing_refine
        new_size_refine = [int(item) for item in new_size_refine]

    elif new_size:
        try:
            len_size = len(new_size)
            if len_size < 3:
                raise Exception('错误，输入图像大小必须为三个维度的空间体素大小')

            # 进行大小重采样的参数设置
            new_spacing_refine = size * spacing / new_size  # 根据当前指定新图像的大小计算出新的体素大小
            # new_spacing_refine = np.array([float(s) for s in new_spacing_refine]).tolist()
            new_size_refine = [int(item) for item in new_size]

        except Exception:
            raise Exception('错误，输入图像大小必须为三个维度的空间体素大小')
    else:
        raise Exception('必须指定重采样后的体素大小或图像大小，参考样例new_spacing=[10, 10, 10]或new_size=[512, 512, 128]')

    resampler: ResampleImageFilter = sitk.ResampleImageFilter()
    if nearest:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetSize(new_size_refine)
    resampler.SetOutputSpacing(new_spacing_refine)

    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    # 暂时使用线性插值算法
    resampler.SetInterpolator(sitk.sitkLinear)

    new_img = resampler.Execute(sitk_img)
    return new_img
