import torch
from torch import nn
import math
import numpy as np
import os, sys
from torchvision import models
import torch.nn.functional as F


# MIND

class MIND(torch.nn.Module):

    def __init__(self, non_local_region_size=9, patch_size=7, neighbor_size=3, gaussian_patch_sigma=3.0):
        super(MIND, self).__init__()
        self.nl_size = non_local_region_size
        self.p_size = patch_size
        self.n_size = neighbor_size
        self.sigma2 = gaussian_patch_sigma * gaussian_patch_sigma

        # calc shifted images in non local region
        self.image_shifter = torch.nn.Conv2d(in_channels=1, out_channels=self.nl_size * self.nl_size,
                                             kernel_size=(self.nl_size, self.nl_size),
                                             stride=1, padding=((self.nl_size - 1) // 2, (self.nl_size - 1) // 2),
                                             dilation=1, groups=1, bias=False, padding_mode='zeros')

        for i in range(self.nl_size * self.nl_size):
            t = torch.zeros((1, self.nl_size, self.nl_size))
            t[0, i % self.nl_size, i // self.nl_size] = 1
            self.image_shifter.weight.data[i] = t

        # patch summation
        self.summation_patcher = torch.nn.Conv2d(in_channels=self.nl_size * self.nl_size,
                                                 out_channels=self.nl_size * self.nl_size,
                                                 kernel_size=(self.p_size, self.p_size),
                                                 stride=1, padding=((self.p_size - 1) // 2, (self.p_size - 1) // 2),
                                                 dilation=1, groups=self.nl_size * self.nl_size, bias=False,
                                                 padding_mode='zeros')

        for i in range(self.nl_size * self.nl_size):
            # gaussian kernel
            t = torch.zeros((1, self.p_size, self.p_size))
            cx = (self.p_size - 1) // 2
            cy = (self.p_size - 1) // 2
            for j in range(self.p_size * self.p_size):
                x = j % self.p_size
                y = j // self.p_size
                d2 = torch.norm(torch.tensor([x - cx, y - cy]).float(), 2)
                t[0, x, y] = math.exp(-d2 / self.sigma2)

            self.summation_patcher.weight.data[i] = t

        # neighbor images
        self.neighbors = torch.nn.Conv2d(in_channels=1, out_channels=self.n_size * self.n_size,
                                         kernel_size=(self.n_size, self.n_size),
                                         stride=1, padding=((self.n_size - 1) // 2, (self.n_size - 1) // 2),
                                         dilation=1, groups=1, bias=False, padding_mode='zeros')

        for i in range(self.n_size * self.n_size):
            t = torch.zeros((1, self.n_size, self.n_size))
            t[0, i % self.n_size, i // self.n_size] = 1
            self.neighbors.weight.data[i] = t

        # neighbor patcher
        self.neighbor_summation_patcher = torch.nn.Conv2d(in_channels=self.n_size * self.n_size,
                                                          out_channels=self.n_size * self.n_size,
                                                          kernel_size=(self.p_size, self.p_size),
                                                          stride=1,
                                                          padding=((self.p_size - 1) // 2, (self.p_size - 1) // 2),
                                                          dilation=1, groups=self.n_size * self.n_size, bias=False,
                                                          padding_mode='zeros')

        for i in range(self.n_size * self.n_size):
            t = torch.ones((1, self.p_size, self.p_size))
            self.neighbor_summation_patcher.weight.data[i] = t

    def forward(self, orig):
        assert (len(orig.shape) == 4)
        assert (orig.shape[1] == 1)

        # get original image channel stack
        orig_stack = torch.stack([orig.squeeze(dim=1) for i in range(self.nl_size * self.nl_size)], dim=1)

        # get shifted images
        shifted = self.image_shifter(orig)

        # get image diff
        diff_images = shifted - orig_stack

        # diff's L2 norm
        Dx_alpha = self.summation_patcher(torch.pow(diff_images, 2.0))

        # calc neighbor's variance
        neighbor_images = self.neighbor_summation_patcher(self.neighbors(orig))
        Vx = neighbor_images.var(dim=1).unsqueeze(dim=1)

        # output mind
        nume = torch.exp(-Dx_alpha / (Vx + 1e-8))
        denomi = nume.sum(dim=1).unsqueeze(dim=1)
        mind = nume / denomi
        return mind


class MINDLoss(torch.nn.Module):

    def __init__(self, non_local_region_size=9, patch_size=7, neighbor_size=3, gaussian_patch_sigma=3.0):
        super(MINDLoss, self).__init__()
        self.pwc = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
        self.nl_size = non_local_region_size
        self.MIND = MIND(non_local_region_size=non_local_region_size,
                         patch_size=patch_size,
                         neighbor_size=neighbor_size,
                         gaussian_patch_sigma=gaussian_patch_sigma)

    def forward(self, input, target):
        assert input.shape == target.shape
        if input.shape[1] > 1:
            input = self.pwc(input)
            target = self.pwc(target)
        in_mind = self.MIND(input)
        tar_mind = self.MIND(target)
        mind_diff = in_mind - tar_mind
        l1 = torch.norm(mind_diff, 1)
        return l1 / (input.shape[2] * input.shape[3] * self.nl_size * self.nl_size)


# MI Loss

class MILoss(nn.Module):
    '''
    This class is a pytorch implementation of the mutual information (MI) calculation between two images.
    This is an approximation, as the images' histograms rely on differentiable approximations of rectangular windows.
            I(X, Y) = H(X) + H(Y) - H(X, Y) = \sum(\sum(p(X, Y) * log(p(Y, Y)/(p(X) * p(Y)))))
    where H(X) = -\sum(p(x) * log(p(x))) is the entropy
    '''

    def __init__(self, bins=50, min=0, max=1, sigma=10, reduction='sum'):
        super(MILoss, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.reduction = reduction

        # 2D joint histogram
        self.hist2d = SoftHistogram2D(bins, min, max, sigma)

        # Epsilon - to avoid log(0)
        self.eps = torch.tensor(0.00000001, dtype=torch.float32, requires_grad=False)

    def forward(self, im1, im2):
        '''
        Forward implementation of a differentiable MI estimator for batched images
        :param im1: N x ... tensor, where N is the batch size
                    ... dimensions can take any form, i.e. 2D images or 3D volumes.
        :param im2: N x ... tensor, where N is the batch size
        :return: N x 1 vector - the approximate MI values between the batched im1 and im2
        '''

        # Check for valid inputs
        assert im1.size() == im2.size(), "(MI_pytorch) Inputs should have the same dimensions."

        batch_size = im1.size()[0]

        # Flatten tensors
        #         im1_flat = im1.view(im1.size()[0], -1)
        #         im2_flat = im2.view(im2.size()[0], -1)
        im1_flat = im1.reshape(im1.size()[0], -1)
        im2_flat = im2.reshape(im2.size()[0], -1)

        # Calculate joint histogram
        hgram = self.hist2d(im1_flat, im2_flat)

        # Convert to a joint distribution
        # Pxy = torch.distributions.Categorical(probs=hgram).probs
        Pxy = torch.div(hgram, torch.sum(hgram.view(hgram.size()[0], -1)))

        # Calculate the marginal distributions
        Py = torch.sum(Pxy, dim=1).unsqueeze(1)
        Px = torch.sum(Pxy, dim=2).unsqueeze(1)

        # Use the KL divergence distance to calculate the MI
        Px_Py = torch.matmul(Px.permute((0, 2, 1)), Py)

        # Reshape to batch_size X all_the_rest
        Pxy = Pxy.reshape(batch_size, -1)
        Px_Py = Px_Py.reshape(batch_size, -1)

        # Calculate mutual information - this is an approximation due to the histogram calculation and eps,
        # but it can handle batches
        if batch_size == 1:
            # No need for eps approximation in the case of a single batch
            nzs = Pxy > 0  # Calculate based on the non-zero values only
            mut_info = torch.matmul(Pxy[nzs], torch.log(Pxy[nzs]) - torch.log(Px_Py[nzs]))  # MI calculation
        else:
            # For arbitrary batch size > 1
            mut_info = torch.sum(Pxy * (torch.log(Pxy + self.eps) - torch.log(Px_Py + self.eps)), dim=1)

        # Reduction
        if self.reduction == 'sum':
            mut_info = torch.sum(mut_info)
        elif self.reduction == 'batchmean':
            mut_info = torch.sum(mut_info)
            mut_info = mut_info / float(batch_size)

        return mut_info


# Note: This is an extension to the 2D case of the previous code snippet
class SoftHistogram2D(nn.Module):
    '''
    Differentiable 1D histogram calculation (supported via pytorch's autograd)
    inupt:
          x, y  - N x D array, where N is the batch size and D is the length of each data series
                 (i.e. vectorized image or vectorized 3D volume)
          bins  - Number of bins for the histogram
          min   - Scalar min value to be included in the histogram
          max   - Scalar max value to be included in the histogram
          sigma - Scalar smoothing factor fir the bin approximation via sigmoid functions.
                  Larger values correspond to sharper edges, and thus yield a more accurate approximation
    output:
          N x bins array, where each row is a histogram
    '''

    def __init__(self, bins=50, min=0, max=1, sigma=10):
        super(SoftHistogram2D, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)  # Bin centers
        self.centers = nn.Parameter(self.centers, requires_grad=False)  # Wrap for allow for cuda support

    def forward(self, x, y):
        assert x.size() == y.size(), "(SoftHistogram2D) x and y sizes do not match"

        # Replicate x and for each row remove center
        x = torch.unsqueeze(x, 1) - torch.unsqueeze(self.centers, 1)
        y = torch.unsqueeze(y, 1) - torch.unsqueeze(self.centers, 1)

        # Bin approximation using a sigmoid function (can be sigma_x and sigma_y respectively - same for delta)
        x = torch.sigmoid(self.sigma * (x + self.delta / 2)) - torch.sigmoid(self.sigma * (x - self.delta / 2))
        y = torch.sigmoid(self.sigma * (y + self.delta / 2)) - torch.sigmoid(self.sigma * (y - self.delta / 2))

        # Batched matrix multiplication - this way we sum jointly
        z = torch.matmul(x, y.permute((0, 2, 1)))
        return z


# DISTS Loss

class DISTSLoss(torch.nn.Module):
    def __init__(self, load_weights=True):
        super(DISTSLoss, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0, 4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

        self.chns = [3, 64, 128, 256, 512, 512]
        self.register_parameter("alpha", nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.register_parameter("beta", nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.alpha.data.normal_(0.1, 0.01)
        self.beta.data.normal_(0.1, 0.01)
        if load_weights:
            weights = torch.load("util/weights.pt")
            self.alpha.data = weights['alpha']
            self.beta.data = weights['beta']

    def forward_once(self, x):
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def forward(self, x, y, require_grad=False, batch_average=False):
        if require_grad:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y)
        dist1 = 0
        dist2 = 0
        c1 = 1e-6
        c2 = 1e-6
        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha / w_sum, self.chns, dim=1)
        beta = torch.split(self.beta / w_sum, self.chns, dim=1)
        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2, 3], keepdim=True)
            y_mean = feats1[k].mean([2, 3], keepdim=True)
            S1 = (2 * x_mean * y_mean + c1) / (x_mean ** 2 + y_mean ** 2 + c1)
            dist1 = dist1 + (alpha[k] * S1).sum(1, keepdim=True)

            x_var = ((feats0[k] - x_mean) ** 2).mean([2, 3], keepdim=True)
            y_var = ((feats1[k] - y_mean) ** 2).mean([2, 3], keepdim=True)
            xy_cov = (feats0[k] * feats1[k]).mean([2, 3], keepdim=True) - x_mean * y_mean
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta[k] * S2).sum(1, keepdim=True)

        score = 1 - (dist1 + dist2).squeeze()
        if batch_average:
            return score.mean()
        else:
            return score


class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer('filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input):
        input = input ** 2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out + 1e-12).sqrt()


if __name__ == "__main__":
    mind = MINDLoss()
    orig = torch.ones(4, 1, 128, 128)
    print(mind(orig, orig).shape)
