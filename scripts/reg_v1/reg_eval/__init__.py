from torch.nn import MSELoss
from .mutual_info import MutualInformation
from .ncc import NCC
from .ssim import SSIM
from .ssim import SSIM3D

from numpy import array
from monai.metrics import (
    compute_meandice,
    compute_average_surface_distance,
    compute_hausdorff_distance,
    compute_percent_hausdorff_distance
)


class Evaluation:
    def __init__(self, metrics=('ncc', 'mi', 'ssim3d')):
        self.name2metric = {
            'ncc': NCC(),
            'ssim': SSIM(),
            'ssim3d': SSIM3D(),
            'mi': MutualInformation(),
            'mse': MSELoss(),
        }
        self.metrics = metrics

    def eval(self, mi, fi):
        result = []
        for item in self.metrics:
            result.append(
                self.name2metric[item](mi, fi).cpu().numpy()
            )
        result = array(result)
        return result


class SegEvaluation:
    def __init__(self, metrics=('ncc', 'mi', 'ssim3d')):
        self.name2metric = {
            'dice': compute_meandice,
            'surface distance': compute_average_surface_distance,
            'hd95': compute_hausdorff_distance,
        }
        self.metrics = metrics

    def eval(self, mi, fi):
        result = []
        for item in self.metrics:
            if item == 'hd95':
                result.append(
                    self.name2metric[item](mi, fi, include_background=False, percentile=95).cpu().squeeze().numpy()
                )
            else:
                result.append(
                    self.name2metric[item](mi, fi, include_background=False).cpu().squeeze().numpy()
                )
        result = array(result)
        return result
