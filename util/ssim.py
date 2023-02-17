import warnings

import torch
import torch.nn.functional as F

# https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py

def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim(X, Y, data_range, win, size_average=True, K=(0.01, 0.03)):

    r""" Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
    Returns:
        torch.Tensor: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim(
    X,
    Y,
    data_range=255,
    size_average=True,
    win_size=11,
    win_sigma=1.5,
    win=None,
    K=(0.01, 0.03),
    nonnegative_ssim=False,
):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu
    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if not X.type() == Y.type():
        raise ValueError("Input images should have the same dtype.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def ms_ssim(
    X, Y, data_range=255, size_average=True, win_size=11, win_sigma=1.5, win=None, weights=None, K=(0.01, 0.03)
):

    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if not X.type() == Y.type():
        raise ValueError("Input images should have the same dtype.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (
        2 ** 4
    ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    # weights = torch.FloatTensor(weights, device=X.device, dtype=X.dtype)
    weights = torch.FloatTensor(weights).to(X.device)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, K=K)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


class SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range=255,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        channel=3,
        spatial_dims=2,
        K=(0.01, 0.03),
        nonnegative_ssim=False,
    ):
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        )


class MS_SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range=255,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        channel=3,
        spatial_dims=2,
        weights=None,
        K=(0.01, 0.03),
    ):
        r""" class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """

        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def forward(self, X, Y):
        return ms_ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            weights=self.weights,
            K=self.K,
        )


##
import torch.nn.functional as F
# import pyrtools as pt
import numpy as np
import torch


class IW_SSIM():
    def __init__(self, iw_flag=True, Nsc=5, blSzX=3, blSzY=3, parent=True,
                 sigma_nsq=0.4, use_cuda=False, use_double=False):
        # MS-SSIM parameters
        self.K = [0.01, 0.03]
        self.L = 255
        self.weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        self.winsize = 11
        self.sigma = 1.5

        # IW-SSIM parameters
        self.iw_flag = iw_flag
        self.Nsc = Nsc  # scales
        self.blSzX = blSzX  # Neighbor size
        self.blSzY = blSzY
        self.parent = parent
        self.sigma_nsq = sigma_nsq

        self.bound = np.ceil((self.winsize - 1) / 2)
        self.bound1 = self.bound - np.floor((self.blSzX - 1) / 2)
        self.use_cuda = use_cuda
        self.use_double = use_double

        self.samplet = torch.tensor([1.0])
        if self.use_cuda:
            self.samplet = self.samplet.cuda()
        if self.use_double:
            self.samplet = self.samplet.double()
        self.samplen = np.array([1.0])
        if not self.use_double:
            self.samplen = self.samplen.astype('float32')

    def fspecial(self, fltr, ws, **kwargs):
        if fltr == 'uniform':
            return np.ones((ws, ws)) / ws ** 2

        elif fltr == 'gaussian':
            x, y = np.mgrid[-ws // 2 + 1:ws // 2 + 1, -ws // 2 + 1:ws // 2 + 1]
            g = np.exp(-((x ** 2 + y ** 2) / (2.0 * kwargs['sigma'] ** 2)))
            g[g < np.finfo(g.dtype).eps * g.max()] = 0
            assert g.shape == (ws, ws)
            den = g.sum()
            if den != 0:
                g /= den
            return g

        return None

    def get_pyrd(self, imgo, imgd):
        imgopr = {}
        imgdpr = {}
        lpo = pt.pyramids.LaplacianPyramid(imgo, height=5)
        lpd = pt.pyramids.LaplacianPyramid(imgd, height=5)
        for scale in range(1, self.Nsc + 1):
            imgopr[scale] = torch.from_numpy(lpo.pyr_coeffs[(scale - 1, 0)]).unsqueeze(0).unsqueeze(0).type(
                self.samplet.type())
            imgdpr[scale] = torch.from_numpy(lpd.pyr_coeffs[(scale - 1, 0)]).unsqueeze(0).unsqueeze(0).type(
                self.samplet.type())

        return imgopr, imgdpr

    def scale_qualty_maps(self, imgopr, imgdpr):

        ms_win = self.fspecial('gaussian', ws=self.winsize, sigma=self.sigma)
        ms_win = torch.from_numpy(ms_win).unsqueeze(0).unsqueeze(0).type(self.samplet.type())
        C1 = (self.K[0] * self.L) ** 2
        C2 = (self.K[1] * self.L) ** 2
        cs_map = {}
        for i in range(1, self.Nsc + 1):
            imgo = imgopr[i]
            imgd = imgdpr[i]
            mu1 = F.conv2d(imgo, ms_win)
            mu2 = F.conv2d(imgd, ms_win)
            sigma12 = F.conv2d(imgo * imgd, ms_win) - mu1 * mu2
            sigma1_sq = F.conv2d(imgo ** 2, ms_win) - mu1 * mu1
            sigma2_sq = F.conv2d(imgd ** 2, ms_win) - mu2 * mu2
            sigma1_sq = torch.max(torch.zeros(sigma1_sq.shape).type(self.samplet.type()), sigma1_sq)
            sigma2_sq = torch.max(torch.zeros(sigma2_sq.shape).type(self.samplet.type()), sigma2_sq)
            cs_map[i] = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
            if i == self.Nsc:
                l_map = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)

        return l_map, cs_map

    def roll(self, x, shift, dim):
        if dim == 0:
            return torch.cat((x[-shift:, :], x[:-shift, :]), dim)
        else:
            return torch.cat((x[:, -shift:], x[:, :-shift]), dim)

    def imenlarge2(self, im):
        _, _, M, N = im.shape
        t1 = F.upsample(im, size=(int(4 * M - 3), int(4 * N - 3)), mode='bilinear')
        t2 = torch.zeros([1, 1, 4 * M - 1, 4 * N - 1]).type(self.samplet.type())
        t2[:, :, 1: -1, 1:-1] = t1
        t2[:, :, 0, :] = 2 * t2[:, :, 1, :] - t2[:, :, 2, :]
        t2[:, :, -1, :] = 2 * t2[:, :, -2, :] - t2[:, :, -3, :]
        t2[:, :, :, 0] = 2 * t2[:, :, :, 1] - t2[:, :, :, 2]
        t2[:, :, :, -1] = 2 * t2[:, :, :, -2] - t2[:, :, :, -3]
        imu = t2[:, :, ::2, ::2]

        return imu

    def info_content_weight_map(self, imgopr, imgdpr):

        tol = 1e-15
        iw_map = {}
        for scale in range(1, self.Nsc):

            imgo = imgopr[scale]
            imgd = imgdpr[scale]
            win = np.ones([self.blSzX, self.blSzY])
            win = win / np.sum(win)
            win = torch.from_numpy(win).unsqueeze(0).unsqueeze(0).type(self.samplet.type())
            padding = int((self.blSzX - 1) / 2)

            # Prepare for estimating IW-SSIM parameters
            mean_x = F.conv2d(imgo, win, padding=padding)
            mean_y = F.conv2d(imgd, win, padding=padding)
            cov_xy = F.conv2d(imgo * imgd, win, padding=padding) - mean_x * mean_y
            ss_x = F.conv2d(imgo ** 2, win, padding=padding) - mean_x ** 2
            ss_y = F.conv2d(imgd ** 2, win, padding=padding) - mean_y ** 2

            ss_x[ss_x < 0] = 0
            ss_y[ss_y < 0] = 0

            # Estimate gain factor and error
            g = cov_xy / (ss_x + tol)
            vv = (ss_y - g * cov_xy)
            g[ss_x < tol] = 0
            vv[ss_x < tol] = ss_y[ss_x < tol]
            ss_x[ss_x < tol] = 0
            g[ss_y < tol] = 0
            vv[ss_y < tol] = 0

            # Prepare parent band
            aux = imgo
            _, _, Nsy, Nsx = aux.shape
            prnt = (self.parent and scale < self.Nsc - 1)
            BL = torch.zeros([1, 1, aux.shape[2], aux.shape[3], 1 + prnt])
            if self.use_cuda:
                BL = BL.cuda()
            if self.use_double:
                BL = BL.double()

            BL[:, :, :, :, 0] = aux
            if prnt:
                auxp = imgopr[scale + 1]
                auxp = self.imenlarge2(auxp)
                BL[:, :, :, :, 1] = auxp[:, :, 0:Nsy, 0:Nsx]
            imgo = BL
            _, _, nv, nh, nb = imgo.shape

            block = torch.tensor([win.shape[2], win.shape[3]])
            if self.use_cuda:
                block = block.cuda()

            # Group neighboring pixels
            nblv = nv - block[0] + 1
            nblh = nh - block[1] + 1
            nexp = nblv * nblh
            N = torch.prod(block) + prnt
            Ly = int((block[0] - 1) / 2)
            Lx = int((block[1] - 1) / 2)
            Y = torch.zeros([nexp, N]).type(self.samplet.type())

            n = -1
            for ny in range(-Ly, Ly + 1):
                for nx in range(-Lx, Lx + 1):
                    n = n + 1
                    temp = imgo[0, 0, :, :, 0]
                    foo1 = self.roll(temp, ny, 0)
                    foo = self.roll(foo1, nx, 1)
                    foo = foo[Ly: Ly + nblv, Lx: Lx + nblh]
                    Y[:, n] = foo.flatten()
            if prnt:
                n = n + 1
                temp = imgo[0, 0, :, :, 1]
                foo = temp
                foo = foo[Ly: Ly + nblv, Lx: Lx + nblh]
                Y[:, n] = foo.flatten()

            C_u = torch.mm(torch.transpose(Y, 0, 1), Y) / nexp.type(self.samplet.type())
            eig_values, H = torch.eig(C_u, eigenvectors=True)
            eig_values = eig_values.type(self.samplet.type())
            H = H.type(self.samplet.type())
            if self.use_double:
                L = torch.diag(eig_values[:, 0] * (eig_values[:, 0] > 0).double()) * torch.sum(eig_values) / (
                            (torch.sum(eig_values[:, 0] * (eig_values[:, 0] > 0).double())) + (
                                torch.sum(eig_values[:, 0] * (eig_values[:, 0] > 0).double()) == 0))
            else:
                L = torch.diag(eig_values[:, 0] * (eig_values[:, 0] > 0).float()) * torch.sum(eig_values) / (
                            (torch.sum(eig_values[:, 0] * (eig_values[:, 0] > 0).float())) + (
                                torch.sum(eig_values[:, 0] * (eig_values[:, 0] > 0).float()) == 0))
            C_u = torch.mm(torch.mm(H, L), torch.transpose(H, 0, 1))
            C_u_inv = torch.inverse(C_u)
            ss = (torch.mm(Y, C_u_inv)) * Y / N.type(self.samplet.type())
            ss = torch.sum(ss, 1)
            ss = ss.view(nblv, nblh)
            ss = ss.unsqueeze(0).unsqueeze(0)
            g = g[:, :, Ly: Ly + nblv, Lx: Lx + nblh]
            vv = vv[:, :, Ly: Ly + nblv, Lx: Lx + nblh]

            # Calculate mutual information
            infow = torch.zeros(g.shape).type(self.samplet.type())
            for j in range(len(eig_values)):
                infow = infow + torch.log2(
                    1 + ((vv + (1 + g * g) * self.sigma_nsq) * ss * eig_values[j, 0] + self.sigma_nsq * vv) / (
                                self.sigma_nsq * self.sigma_nsq))
            infow[infow < tol] = 0
            iw_map[scale] = infow

        return iw_map

    def test(self, imgo, imgd):

        imgo = imgo.astype(self.samplen.dtype)
        imgd = imgd.astype(self.samplen.dtype)
        imgopr, imgdpr = self.get_pyrd(imgo, imgd)
        l_map, cs_map = self.scale_qualty_maps(imgopr, imgdpr)
        if self.iw_flag:
            iw_map = self.info_content_weight_map(imgopr, imgdpr)

        wmcs = []
        for s in range(1, self.Nsc + 1):
            cs = cs_map[s]
            if s == self.Nsc:
                cs = cs_map[s] * l_map

            if self.iw_flag:
                if s < self.Nsc:
                    iw = iw_map[s]
                    if self.bound1 != 0:
                        iw = iw[:, :, int(self.bound1): -int(self.bound1), int(self.bound1): -int(self.bound1)]
                    else:
                        iw = iw[:, :, int(self.bound1):, int(self.bound1):]
                else:
                    iw = torch.ones(cs.shape).type(self.samplet.type())
                wmcs.append(torch.sum(cs * iw) / torch.sum(iw))
            else:
                wmcs.append(torch.mean(cs))

        wmcs = torch.tensor(wmcs).type(self.samplet.type())
        self.weight = torch.tensor(self.weight).type(self.samplet.type())
        score = torch.prod((torch.abs(wmcs)) ** (self.weight))

        return score


if __name__ == '__main__':
    iw_ssim = IW_SSIM(iw_flag=True, Nsc=5, blSzX=3, blSzY=3, parent=True, sigma_nsq=0.4, use_cuda=False, use_double=False)
    x = torch.randn(1, 1, 224, 224)
    y = torch.randn(1, 1, 224, 224)
    res = iw_ssim.test(x, y)
