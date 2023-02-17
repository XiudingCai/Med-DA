import functools

import numpy as np
import torch
from torch import nn
import tqdm
from torch.autograd import Variable

import data

# N = 88
N = 179
epsilon = 1e-8
device = 'cuda:0'


def azimuthalAverage_np(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).

    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]  # location of changed radius
    nr = rind[1:] - rind[:-1]  # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof

def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).

    """
    # Calculate the indices from the image
    # y, x = np.indices(image.shape)
    y, x = torch.meshgrid(torch.arange(image.size(0)), torch.arange(image.size(1)))

    if not center:
        center = torch.tensor([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])

    r = torch.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = torch.argsort(r.flatten())
    r_sorted = r.flatten()[ind]
    i_sorted = image.flatten()[ind]

    # Get the integer part of the radii (bin size = 1)
    # r_int = r_sorted.astype(int)
    r_int = r_sorted.type(torch.int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = torch.where(deltar)[0]  # location of changed radius
    nr = rind[1:] - rind[:-1]  # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = torch.cumsum(i_sorted, dim=0, dtype=torch.float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof


def cal_loss_np(x_fake, x_real):
    # fake image 1d power spectrum
    psd1D_img = np.zeros([x_fake.shape[0], N])
    for t in range(x_fake.shape[0]):
        img_gray = x_fake[t, 0, :, :].cpu().detach().numpy()
        fft = np.fft.fft2(img_gray)
        fshift = np.fft.fftshift(fft)
        fshift += epsilon
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        psd1D = azimuthalAverage_np(magnitude_spectrum)
        psd1D = (psd1D - np.min(psd1D)) / (np.max(psd1D) - np.min(psd1D))
        psd1D_img[t, :] = psd1D

    psd1D_img = torch.from_numpy(psd1D_img).float()
    psd1D_img = Variable(psd1D_img, requires_grad=True).to(device)

    # real image 1d power spectrum
    psd1D_rec = np.zeros([x_real.shape[0], N])
    for t in range(x_real.shape[0]):
        img_gray = x_real[t, 0, :, :].cpu().detach().numpy()
        fft = np.fft.fft2(img_gray)
        fshift = np.fft.fftshift(fft)
        fshift += epsilon
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        psd1D = azimuthalAverage_np(magnitude_spectrum)
        psd1D = (psd1D - np.min(psd1D)) / (np.max(psd1D) - np.min(psd1D))
        psd1D_rec[t, :] = psd1D

    psd1D_rec = torch.from_numpy(psd1D_rec).float()
    psd1D_rec = Variable(psd1D_rec, requires_grad=True).to(device)

    loss_freq = torch.nn.BCELoss()(psd1D_rec, psd1D_img.detach())

    return loss_freq


def cal_loss_torch(x_fake, x_real):
    # fake image 1d power spectrum
    psd1D_img = torch.zeros([x_fake.shape[0], N])
    for t in range(x_fake.shape[0]):
        img_gray = x_fake[t, 0, :, :].detach()
        fft = torch.fft.fft2(img_gray)
        fshift = torch.fft.fftshift(fft)
        fshift += epsilon
        magnitude_spectrum = 20 * torch.log(torch.abs(fshift))
        psd1D = azimuthalAverage(magnitude_spectrum)
        psd1D = (psd1D - torch.min(psd1D)) / (torch.max(psd1D) - torch.min(psd1D))
        psd1D_img[t, :] = psd1D

    psd1D_img = Variable(psd1D_img, requires_grad=True).to(device)

    # real image 1d power spectrum
    psd1D_rec = torch.zeros([x_real.shape[0], N])
    for t in range(x_real.shape[0]):
        img_gray = x_real[t, 0, :, :].detach()
        fft = torch.fft.fft2(img_gray)
        fshift = torch.fft.fftshift(fft)
        fshift += epsilon
        magnitude_spectrum = 20 * torch.log(torch.abs(fshift))
        psd1D = azimuthalAverage(magnitude_spectrum)
        psd1D = (psd1D - torch.min(psd1D)) / (torch.max(psd1D) - torch.min(psd1D))
        psd1D_rec[t, :] = psd1D

    psd1D_rec = Variable(psd1D_rec, requires_grad=True).to(device)

    loss_freq = torch.nn.BCELoss()(psd1D_rec, psd1D_img.detach())

    return loss_freq


if __name__ == '__main__':
    x_fake = torch.randn(2, 1, 256, 256)
    x_real = torch.randn(2, 1, 256, 256)

    loss_np = cal_loss_np(x_fake, x_real)
    loss_torch = cal_loss_torch(x_fake, x_real)
    print(loss_np, '\n', loss_torch)
