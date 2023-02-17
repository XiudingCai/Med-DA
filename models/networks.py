import random

import einops
import torch
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
from .stylegan_networks import StyleGAN2Discriminator, StyleGAN2Generator
from models.blocks.LPTN import *
from .blocks.shuffle_blocks import ShuffleBlock, ShuffleBlockX
from .blocks.mb_blocks import MBInvertedConvLayer
from .blocks.convnext_blocks import Block as NextBlock
from .blocks.hrt_blocks import HRTransBlock, GeneralTransformerBlock


###############################################################################
# Helper Functions
###############################################################################


def get_filter(filt_size=3):
    if (filt_size == 1):
        a = np.array([1., ])
    elif (filt_size == 2):
        a = np.array([1., 1.])
    elif (filt_size == 3):
        a = np.array([1., 2., 1.])
    elif (filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif (filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif (filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif (filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt


class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)),
                          int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if (self.filt_size == 1):
            if (self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class Upsample2(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        self.factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=self.factor, mode=self.mode)


class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride ** 2)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size,
                                     groups=inp.shape[1])[:, :, 1:, 1:]
        if (self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, -1, window_size, window_size)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = windows.shape[0]
    x = windows.view(B, -1, H // window_size, W // window_size, window_size, window_size)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, -1, H, W)
    return x


def get_pad_layer(pad_type):
    if (pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif (pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif (pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        # if not amp:
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal',
             init_gain=0.02, no_antialias=False, no_antialias_up=False, gpu_ids=[], opt=None):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                              no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=9, opt=opt)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                              no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=6, opt=opt)
    elif netG == 'resnet_4blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                              no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=4, opt=opt)
    elif netG == 'next':
        net = NextGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                            no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=4, opt=opt)
    elif netG == 'inred':
        net = RednetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                              no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=4, opt=opt)
    elif netG == 'swin':
        net = SwinTGenerator(img_size=256, patch_size=4, in_chans=input_nc, num_classes=1000,
                             embed_dim=96, depths=[2, 2, 2], num_heads=[3, 6, 12],
                             window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                             norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                             use_checkpoint=False,
                             input_nc=input_nc, output_nc=output_nc, ngf=ngf,
                             use_dropout=use_dropout,
                             no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=9, opt=opt)
    elif netG == 'former':
        from models.CycleGANFormer import TGenerator
        net = TGenerator().cuda()
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'stylegan2':
        net = StyleGAN2Generator(input_nc, output_nc, ngf, use_dropout=use_dropout, opt=opt)
    elif netG == 'smallstylegan2':
        net = StyleGAN2Generator(input_nc, output_nc, ngf, use_dropout=use_dropout, n_blocks=2, opt=opt)
    elif netG == 'resnet_cat':
        n_blocks = 8
        net = G_Resnet(input_nc, output_nc, opt.nz, num_downs=2, n_res=n_blocks - 4, ngf=ngf, norm='inst',
                       nl_layer='relu')
    elif netG == 'lptn':
        net = LPTN(output_nc=output_nc)
    elif netG == 'mposv1':
        from .mpos_space import MPOSGeneratorV1
        net = MPOSGeneratorV1(input_nc, output_nc)
    elif netG == 'resvit':
        from .ResVit import residual_transformers

        vit_name = "Res-ViT-B_16"
        net = residual_transformers.ResViT(residual_transformers.CONFIGS[vit_name],
                                           input_dim=input_nc, img_size=opt.crop_size, output_dim=output_nc, vis=False,
                                           norm_layer=norm_layer, padding_type='replicate')
    elif netG == 'spos':
        net = ResnetGeneratorV3(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=3, opt=opt)
    elif netG == 'hourglass':
        net = ResnetGeneratorV4(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=3, opt=opt)
    elif netG == 'hg_hrt':
        net = HGHRTGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=3, opt=opt)
    elif netG == 'hrkormer':
        net = HRKormerGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=3, opt=opt)
    elif netG == 'hrkormerv2':
        net = HRKormerGeneratorV2(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                  no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=3, opt=opt)
    elif netG == 'hrkormerv3':
        net = HRKormerGeneratorV3(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                  no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=3, opt=opt)
    elif netG == 'hrkormerv4':
        net = HRKormerGeneratorV4(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                  no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=3, opt=opt)
    elif netG == 'hrkormerv5':
        net = HRKormerGeneratorV(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                 no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=3, opt=opt)

    elif netG == 'VIP':
        net = ResnetGeneratorV(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                               no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=3, opt=opt)
    elif netG == 'sr':
        net = ResnetGeneratorSR(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=3, opt=opt)
    elif netG == 'frequency':
        net = ResnetGeneratorV5(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=3, opt=opt)
    elif netG == 'hrt':
        net = ResnetGeneratorV2(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=3, opt=opt)
    elif netG == 'hrt_2blocks':
        net = ResnetGeneratorV2(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=2, opt=opt)
    elif netG == 'hrt_nb':
        net = ResnetGeneratorV2(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                no_antialias=no_antialias, no_antialias_up=no_antialias_up,
                                n_blocks=opt.n_blocks, opt=opt)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids, initialize_weights=('stylegan2' not in netG))


def define_F(input_nc, netF, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, no_antialias=False,
             gpu_ids=[], opt=None):
    if netF == 'global_pool':
        net = PoolingF()
    elif netF == 'reshape':
        net = ReshapeF()
    elif netF == 'mapping':
        net = MappingF(input_nc, gpu_ids=gpu_ids)
    elif netF == 'sample':
        net = PatchSampleF(use_mlp=False, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    elif netF == 'mlp_sample':
        net = PatchSampleF(use_mlp=True, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    elif netF == 'cam_sample':
        net = PatchCamSampleF(use_mlp=False, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    elif netF == 'mlp_cam_sample':
        net = PatchCamSampleF(use_mlp=True, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    elif netF == 'strided_conv':
        net = StridedConvF(init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('projection model name [%s] is not recognized' % netF)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, no_antialias=False,
             gpu_ids=[], opt=None):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leaky RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, no_antialias=no_antialias, )
    elif netD == 'hg':  # default PatchGAN classifier
        net = NLayerHDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, no_antialias=no_antialias, )
    elif netD == 'basic3d':  # default PatchGAN classifier
        net = NLayerDiscriminator3D(input_nc, ndf, n_layers=3, norm_layer=norm_layer, no_antialias=no_antialias, )
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, no_antialias=no_antialias, )
    elif netD == 'n_layers_ms':  # more options
        net = NLayerMSDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, no_antialias=no_antialias, )
    elif netD == 'n_layers_contrast':  # more options
        net = NLayerContrastiveDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer,
                                             no_antialias=no_antialias, )
    elif netD == 'n_layers_cd':  # more options
        net = NLayerCDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, no_antialias=no_antialias, )
    elif netD == 'pixel':  # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif 'stylegan2' in netD:
        net = StyleGAN2Discriminator(input_nc, ndf, n_layers_D, no_antialias=no_antialias, opt=opt)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids,
                    initialize_weights=('stylegan2' not in netD))


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'l1':
            self.loss = nn.L1Loss()
        elif gan_mode == 'smoothl1':
            self.loss = nn.SmoothL1Loss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'nonsaturating']:
            self.loss = None
        elif gan_mode == "hinge":
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        bs = prediction.size(0)
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'nonsaturating':
            if target_is_real:
                loss = F.softplus(-prediction).view(bs, -1).mean(dim=1)
            else:
                loss = F.softplus(prediction).view(bs, -1).mean(dim=1)
        elif self.gan_mode == 'hinge':
            if target_is_real:
                minvalue = torch.min(prediction - 1, torch.zeros(prediction.shape).to(prediction.device))
                loss = -torch.mean(minvalue)
            else:
                minvalue = torch.min(-prediction - 1, torch.zeros(prediction.shape).to(prediction.device))
                loss = -torch.mean(minvalue)
        elif self.gan_mode in ['l1', 'smoothl1']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class PoolingF(nn.Module):
    def __init__(self):
        super(PoolingF, self).__init__()
        model = [nn.AdaptiveMaxPool2d(1)]
        self.model = nn.Sequential(*model)
        self.l2norm = Normalize(2)

    def forward(self, x):
        return self.l2norm(self.model(x))


class ReshapeF(nn.Module):
    def __init__(self):
        super(ReshapeF, self).__init__()
        model = [nn.AdaptiveAvgPool2d(4)]
        self.model = nn.Sequential(*model)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.model(x)
        x_reshape = x.permute(0, 2, 3, 1).flatten(0, 2)
        return self.l2norm(x_reshape)


class MappingF(nn.Module):
    def __init__(self, in_layer=4, gpu_ids=[], nc=256, patch_num=256, dim=64, init_type='normal', init_gain=0.02):
        # hard-coded code.
        super().__init__()
        self.init_type = init_type
        self.nc = nc
        self.dim = dim
        self.in_layer = in_layer
        self.patch_num = patch_num
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids
        avg = nn.AdaptiveAvgPool2d(1)
        conv = nn.Conv2d(in_layer, dim, 3, stride=2)
        self.model = nn.Sequential(
            *[conv, nn.ReLU(), avg, nn.Flatten(), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)])
        init_net(self.model, self.init_type, self.init_gain, self.gpu_ids)
        self.l2norm = Normalize(2)

    def forward(self, x):
        print(x.shape)
        x = x.view(1, -1, self.patch_num, self.nc)
        x = self.model(x)
        x_norm = self.l2norm(x)
        print(x_norm.shape)
        return x_norm


class StridedConvF(nn.Module):
    def __init__(self, init_type='normal', init_gain=0.02, gpu_ids=[]):
        super().__init__()
        # self.conv1 = nn.Conv2d(256, 128, 3, stride=2)
        # self.conv2 = nn.Conv2d(128, 64, 3, stride=1)
        self.l2_norm = Normalize(2)
        self.mlps = {}
        self.moving_averages = {}
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, x):
        C, H = x.shape[1], x.shape[2]
        n_down = int(np.rint(np.log2(H / 32)))
        mlp = []
        for i in range(n_down):
            mlp.append(nn.Conv2d(C, max(C // 2, 64), 3, stride=2))
            mlp.append(nn.ReLU())
            C = max(C // 2, 64)
        mlp.append(nn.Conv2d(C, 64, 3))
        mlp = nn.Sequential(*mlp)
        init_net(mlp, self.init_type, self.init_gain, self.gpu_ids)
        return mlp

    def update_moving_average(self, key, x):
        if key not in self.moving_averages:
            self.moving_averages[key] = x.detach()

        self.moving_averages[key] = self.moving_averages[key] * 0.999 + x.detach() * 0.001

    def forward(self, x, use_instance_norm=False):
        C, H = x.shape[1], x.shape[2]
        key = '%d_%d' % (C, H)
        if key not in self.mlps:
            self.mlps[key] = self.create_mlp(x)
            self.add_module("child_%s" % key, self.mlps[key])
        mlp = self.mlps[key]
        x = mlp(x)
        self.update_moving_average(key, x)
        x = x - self.moving_averages[key]
        if use_instance_norm:
            x = F.instance_norm(x)
        return self.l2_norm(x)


class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            # print(feat.shape, feat_reshape.shape)
            # torch.Size([2, 1, 262, 262])   torch.Size([2, 68644, 1])
            # torch.Size([2, 128, 256, 256]) torch.Size([2, 65536, 128])
            # torch.Size([2, 256, 128, 128]) torch.Size([2, 16384, 256])
            # torch.Size([2, 256, 64, 64])   torch.Size([2, 4096, 256])
            # torch.Size([2, 256, 64, 64])   torch.Size([2, 4096, 256])
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            # print(x_sample.shape)
            # torch.Size([512, 1])
            # torch.Size([512, 128])
            # torch.Size([512, 256])
            # torch.Size([512, 256])
            # torch.Size([512, 256])
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


class PatchCamSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchCamSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None, cams=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)

            if num_patches > 0:
                feat_b_list = []
                id_b_list = []
                if patch_ids is not None:
                    for b, patch_id in enumerate(patch_ids[feat_id]):
                        feat_b_list.append(feat_reshape[b, patch_id, :])
                    feat_reshape = torch.cat(feat_b_list, dim=0).unsqueeze(0)
                else:
                    num_points = int(min(num_patches, H * W))
                    oversample_ratio = 10
                    for b in range(cams[feat_id].shape[0]):
                        cam_b = cams[feat_id][b]

                        points_sampled = torch.randperm(H * W)[:num_points * oversample_ratio]

                        values, indices = torch.sort(cam_b.flatten()[points_sampled], descending=False)

                        good_idx = points_sampled[indices[:num_points // 2]]
                        points_random = torch.randperm(H * W)[-num_points // 2:]

                        patch_id = torch.cat([good_idx, points_random], dim=0)
                        feat_b_list.append(feat_reshape[b, patch_id, :])
                        id_b_list.append(patch_id)
                    feat_reshape = torch.cat(feat_b_list, dim=0).unsqueeze(0)

                x_sample = feat_reshape.flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                id_b_list = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(id_b_list)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


class G_Resnet(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, n_res, ngf=64,
                 norm=None, nl_layer=None):
        super(G_Resnet, self).__init__()
        n_downsample = num_downs
        pad_type = 'reflect'
        self.enc_content = ContentEncoder(n_downsample, n_res, input_nc, ngf, norm, nl_layer, pad_type=pad_type)
        if nz == 0:
            self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, output_nc, norm=norm, activ=nl_layer,
                               pad_type=pad_type, nz=nz)
        else:
            self.dec = Decoder_all(n_downsample, n_res, self.enc_content.output_dim, output_nc, norm=norm,
                                   activ=nl_layer, pad_type=pad_type, nz=nz)

    def decode(self, content, style=None):
        return self.dec(content, style)

    def forward(self, image, style=None, nce_layers=[], encode_only=False):
        content, feats = self.enc_content(image, nce_layers=nce_layers, encode_only=encode_only)
        if encode_only:
            return feats
        else:
            images_recon = self.decode(content, style)
            if len(nce_layers) > 0:
                return images_recon, feats
            else:
                return images_recon


##################################################################################
# Encoder and Decoders
##################################################################################


class E_adaIN(nn.Module):
    def __init__(self, input_nc, output_nc=1, nef=64, n_layers=4,
                 norm=None, nl_layer=None, vae=False):
        # style encoder
        super(E_adaIN, self).__init__()
        self.enc_style = StyleEncoder(n_layers, input_nc, nef, output_nc, norm='none', activ='relu', vae=vae)

    def forward(self, image):
        style = self.enc_style(image)
        return style


class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, vae=False):
        super(StyleEncoder, self).__init__()
        self.vae = vae
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type='reflect')]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type='reflect')]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type='reflect')]
        self.model += [nn.AdaptiveAvgPool2d(1)]  # global average pooling
        if self.vae:
            self.fc_mean = nn.Linear(dim, style_dim)  # , 1, 1, 0)
            self.fc_var = nn.Linear(dim, style_dim)  # , 1, 1, 0)
        else:
            self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]

        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        if self.vae:
            output = self.model(x)
            output = output.view(x.size(0), -1)
            output_mean = self.fc_mean(output)
            output_var = self.fc_var(output)
            return output_mean, output_var
        else:
            return self.model(x).view(x.size(0), -1)


class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type='zero'):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type='reflect')]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type='reflect')]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x, nce_layers=[], encode_only=False):
        if len(nce_layers) > 0:
            feat = x
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in nce_layers:
                    feats.append(feat)
                if layer_id == nce_layers[-1] and encode_only:
                    return None, feats
            return feat, feats
        else:
            return self.model(x), None


class Decoder_all(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, norm='batch', activ='relu', pad_type='zero', nz=0):
        super(Decoder_all, self).__init__()
        # AdaIN residual blocks
        self.resnet_block = ResBlocks(n_res, dim, norm, activ, pad_type=pad_type, nz=nz)
        self.n_blocks = 0
        # upsampling blocks
        for i in range(n_upsample):
            block = [Upsample2(scale_factor=2),
                     Conv2dBlock(dim + nz, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type='reflect')]
            setattr(self, 'block_{:d}'.format(self.n_blocks), nn.Sequential(*block))
            self.n_blocks += 1
            dim //= 2
        # use reflection padding in the last conv layer
        setattr(self, 'block_{:d}'.format(self.n_blocks),
                Conv2dBlock(dim + nz, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type='reflect'))
        self.n_blocks += 1

    def forward(self, x, y=None):
        if y is not None:
            output = self.resnet_block(cat_feature(x, y))
            for n in range(self.n_blocks):
                block = getattr(self, 'block_{:d}'.format(n))
                if n > 0:
                    output = block(cat_feature(output, y))
                else:
                    output = block(output)
            return output


class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, norm='batch', activ='relu', pad_type='zero', nz=0):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, norm, activ, pad_type=pad_type, nz=nz)]
        # upsampling blocks
        for i in range(n_upsample):
            if i == 0:
                input_dim = dim + nz
            else:
                input_dim = dim
            self.model += [Upsample2(scale_factor=2),
                           Conv2dBlock(input_dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type='reflect')]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type='reflect')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x, y=None):
        if y is not None:
            return self.model(cat_feature(x, y))
        else:
            return self.model(x)


##################################################################################
# Sequential Models
##################################################################################


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='inst', activation='relu', pad_type='zero', nz=0):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, nz=nz)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


##################################################################################
# Basic Blocks
##################################################################################
def cat_feature(x, y):
    y_expand = y.view(y.size(0), y.size(1), 1, 1).expand(
        y.size(0), y.size(1), x.size(2), x.size(3))
    x_cat = torch.cat([x, y_expand], 1)
    return x_cat


class ResBlock(nn.Module):
    def __init__(self, dim, norm='inst', activation='relu', pad_type='zero', nz=0):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim + nz, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim + nz, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'batch':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'inst':
            self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=False)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'inst':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


##################################################################################
# Normalization layers
##################################################################################


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class MPResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(MPResnetGenerator, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        model_down = []
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if (no_antialias):
                model_down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                               norm_layer(ngf * mult * 2),
                               nn.ReLU(True)]
            else:
                model_down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                               norm_layer(ngf * mult * 2),
                               nn.ReLU(True),
                               Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, layers=[], encode_only=False):
        """
        input: input
        layers: nec_layers, e.g., [0, 4, 8, 12, 16]
        encode_only: when encode_only is True, for nce loss
        """
        # -1 means the last layer
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            fake = self.model(input)
            return fake


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if (no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, layers=[], encode_only=False):
        """
        input: input
        layers: nec_layers, e.g., [0, 4, 8, 12, 16]
        encode_only: when encode_only is True, for nce loss
        """
        # -1 means the last layer
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            # print(input.shape)

            isTrain = len(input.shape) == 5
            if isTrain:
                input = input.squeeze(1).permute(0, 3, 2, 1)
                input = torch.nn.functional.interpolate(input, size=(256, 256), mode='bilinear')
            # print(input.shape)
            fake = self.model(input)
            # print(fake.shape)
            if isTrain:
                # fake = torch.nn.functional.interpolate(fake, size=(176, 176), mode='bilinear')
                fake = fake.permute(0, 3, 2, 1).unsqueeze(1)
            return fake


class NextGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(NextGenerator, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if (no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [NextBlock(ngf * mult)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, layers=[], encode_only=False):
        """
        input: input
        layers: nec_layers, e.g., [0, 4, 8, 12, 16]
        encode_only: when encode_only is True, for nce loss
        """
        # -1 means the last layer
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            # print(input.shape)

            isTrain = len(input.shape) == 5
            if isTrain:
                input = input.squeeze(1).permute(0, 3, 2, 1)
                input = torch.nn.functional.interpolate(input, size=(256, 256), mode='bilinear')
            # print(input.shape)
            fake = self.model(input)
            # print(fake.shape)
            if isTrain:
                fake = torch.nn.functional.interpolate(fake, size=(176, 176), mode='bilinear')
                fake = fake.permute(0, 3, 2, 1).unsqueeze(1)
            return fake


class ResnetGeneratorV2(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGeneratorV2, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if (no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling

        flag = 'hrt'

        for i in range(n_blocks):  # add ResNet blocks

            # model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
            #                       use_bias=use_bias)]
            if flag == 'hrt':
                num_branches = 1
                block = GeneralTransformerBlock
                num_blocks = [1]
                num_inchannels = [ngf * mult]
                num_channels = [ngf * mult]
                num_heads = [2]
                num_window_sizes = [7]
                num_mlp_ratios = [4]
                reset_multi_scale_output = False
                drop_path = [0., 0.]
                skip_connec = self.opt.skip_connec
                model += [HRTransBlock(num_branches,
                                       block,  # TRANSFORMER_BLOCK
                                       num_blocks,  # 2
                                       num_inchannels,  # 256
                                       num_channels,  # 256
                                       num_heads,  # 2
                                       num_window_sizes,  # 7
                                       num_mlp_ratios,  # 4
                                       reset_multi_scale_output,
                                       drop_path=drop_path,
                                       skip_connec=skip_connec)]
            elif flag == 'swinir':
                from .blocks.swinir import RSTBlock
                model += [RSTBlock(dim=256, input_resolution=(64, 64), depth=6, num_heads=4, window_size=8,
                                   mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                                   drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                                   img_size=64, patch_size=4, resi_connection='1conv')]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        print(self.model)

    def forward(self, input, layers=[], encode_only=False):
        """
        input: input
        layers: nec_layers, e.g., [0, 4, 8, 12, 16]
        encode_only: when encode_only is True, for nce loss
        """
        # -1 means the last layer
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            fake = self.model(input)
            return fake


class ResnetGeneratorV3(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGeneratorV3, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model_in = [nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if (no_antialias):
                model_in += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                             norm_layer(ngf * mult * 2),
                             nn.ReLU(True)]
            else:
                model_in += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                             norm_layer(ngf * mult * 2),
                             nn.ReLU(True),
                             Downsample(ngf * mult * 2)]

        model = []
        mult = 2 ** n_downsampling

        self.n_blocks = self.opt.n_blocks

        for i in range(self.n_blocks):  # add ResNet blocks

            # model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
            #                       use_bias=use_bias)]
            num_branches = 1
            block = GeneralTransformerBlock
            num_blocks = [1]
            num_inchannels = [ngf * mult]
            num_channels = [ngf * mult]
            num_heads = [2]
            num_window_sizes = [7]
            num_mlp_ratios = [4]
            reset_multi_scale_output = False
            drop_path = [0., 0.]

            model += [nn.ModuleList([
                HRTransBlock(num_branches,
                             block,  # TRANSFORMER_BLOCK
                             num_blocks,  # 2
                             num_inchannels,  # 256
                             num_channels,  # 256
                             num_heads,  # 2
                             num_window_sizes,  # 7
                             num_mlp_ratios,  # 4
                             reset_multi_scale_output,
                             drop_path=drop_path),
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias),
                MBInvertedConvLayer(in_channels=ngf * mult, out_channels=ngf * mult, kernel_size=3, stride=1,
                                    expand_ratio=6, mid_channels=None),
                # nn.Sequential(*[ShuffleBlock(in_channels=ngf * mult // 2, out_channels=ngf * mult, kernel=7, stride=1),
                #                 ShuffleBlockX(in_channels=ngf * mult // 2, out_channels=ngf * mult, stride=1),
                #                 ShuffleBlockX(in_channels=ngf * mult // 2, out_channels=ngf * mult, stride=1),
                #                 ShuffleBlock(in_channels=ngf * mult // 2, out_channels=ngf * mult, kernel=7, stride=1)])
                # nn.Sequential(*[])
            ])]
        model_out = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model_out += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                                 kernel_size=3, stride=2,
                                                 padding=1, output_padding=1,
                                                 bias=use_bias),
                              norm_layer(int(ngf * mult / 2)),
                              nn.ReLU(True)]
            else:
                model_out += [Upsample(ngf * mult),
                              nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                        kernel_size=3, stride=1,
                                        padding=1,  # output_padding=1,
                                        bias=use_bias),
                              norm_layer(int(ngf * mult / 2)),
                              nn.ReLU(True)]
        model_out += [nn.ReflectionPad2d(3)]
        model_out += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_out += [nn.Tanh()]

        self.model_in = nn.Sequential(*model_in)
        self.model = nn.Sequential(*model)
        self.model_out = nn.Sequential(*model_out)

    def window_partition(self, x, window_size: int):
        """
        Args:
            x: (B, H, W, C)
            window_size (int): window size
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, C, H, W = x.shape
        x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def window_reverse(self, windows, window_size: int, H: int, W: int):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): Window size
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, input, layers=[], encode_only=False, choice=None):
        """
        input: input
        layers: nec_layers, e.g., [0, 4, 8, 12, 16]
        encode_only: when encode_only is True, for nce loss
        """
        # -1 means the last layer
        if -1 in layers:
            layers.append(len(self.model_in) + len(self.model) + len(self.model_out))
        if len(layers) > 0:
            feat = input
            feats = []
            layer_id = 0
            for layer in self.model_in:
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers
                layer_id += 1

            for i, j in enumerate(choice):
                feat = self.model[i][j](feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers
                layer_id += 1

            for layer in self.model_out:
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers
                layer_id += 1

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            x = self.model_in(input)

            # x = self.model(x)
            for i, j in enumerate(choice):
                x = self.model[i][j](x)

            fake = self.model_out(x)
            return fake


class ResnetGeneratorV4(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGeneratorV4, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        idx_down = 0
        n_downsampling = 1
        # 1: 256 (64) -> 128 (128)
        for _ in range(n_downsampling):  # add downsampling layers
            mult = 2 ** idx_down
            idx_down += 1
            if (no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        # self.n_blocks = self.opt.n_blocks

        # 2: 128 (128) -> 128 (128)
        n_blocks = 1
        for i in range(n_blocks):  # add ResNet blocks
            mult = 2 ** idx_down
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        n_downsampling = 1
        # 3: 128 (128) -> 64 (256)
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** idx_down
            idx_down += 1
            if (no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        # 4: 64 (256) -> 64 (256)
        n_blocks = 2
        for i in range(n_blocks):  # add ResNet blocks
            mult = 2 ** idx_down
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        n_upsampling = 1
        idx_up = idx_down
        # 5: 64 (256) -> 128 (128)
        for i in range(n_upsampling):  # add upsampling layers
            mult = 2 ** idx_up
            idx_up -= 1
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]

        n_blocks = 1
        # 6: 128 (128) -> 128 (128)
        for i in range(n_blocks):  # add ResNet blocks
            mult = 2 ** idx_up
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        n_downsampling = 1
        # 7: 128 (128) -> 64 (256)
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** (idx_down - 1)
            if (no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        # 8: 64 (256) -> 64 (256)
        n_blocks = 2
        for i in range(n_blocks):  # add ResNet blocks
            mult = 2 ** idx_down
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        # 9: 64 (256) -> 128 (128)
        n_upsampling = 1
        for i in range(n_upsampling):  # add upsampling layers
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]

        # 10: 128 (128) -> 128 (128)
        n_blocks = 1
        for i in range(n_blocks):  # add ResNet blocks
            mult = 2 ** (idx_down - 1)

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        # 11: 128 (128) -> 256 (64)
        for i in range(n_upsampling):  # add upsampling layers
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        # print(self.model)

    def forward(self, input, layers=[], encode_only=False):
        """
        input: input
        layers: nec_layers, e.g., [0, 4, 8, 12, 16]
        encode_only: when encode_only is True, for nce loss
        """
        # -1 means the last layer
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            fake = self.model(input)
            return fake


class ResnetGeneratorSR(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGeneratorSR, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        idx_down = 0
        n_downsampling = 1
        # 1: 256 (64) -> 128 (128)
        for _ in range(n_downsampling):  # add downsampling layers
            mult = 2 ** idx_down
            idx_down += 1
            if (no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        # self.n_blocks = self.opt.n_blocks

        # 2: 128 (128) -> 128 (128)
        n_blocks = 1
        for i in range(n_blocks):  # add ResNet blocks
            mult = 2 ** idx_down
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        n_downsampling = 1
        # 3: 128 (128) -> 64 (256)
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** idx_down
            idx_down += 1
            if (no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        # 4: 64 (256) -> 64 (256)
        n_blocks = 2
        for i in range(n_blocks):  # add ResNet blocks
            mult = 2 ** idx_down
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        n_upsampling = 1
        idx_up = idx_down
        # 5: 64 (256) -> 128 (128)
        for i in range(n_upsampling):  # add upsampling layers
            mult = 2 ** idx_up
            idx_up -= 1
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]

        n_blocks = 1
        # 6: 128 (128) -> 128 (128)
        for i in range(n_blocks):  # add ResNet blocks
            mult = 2 ** idx_up
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        n_downsampling = 1
        # 7: 128 (128) -> 64 (256)
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** (idx_down - 1)
            if (no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        # 8: 64 (256) -> 64 (256)
        n_blocks = 2
        for i in range(n_blocks):  # add ResNet blocks
            mult = 2 ** idx_down
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        # 9: 64 (256) -> 128 (128)
        n_upsampling = 1
        for i in range(n_upsampling):  # add upsampling layers
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]

        # 10: 128 (128) -> 128 (128)
        n_blocks = 1
        for i in range(n_blocks):  # add ResNet blocks
            mult = 2 ** (idx_down - 1)

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        # 11: 128 (128) -> 256 (64)
        for i in range(n_upsampling):  # add upsampling layers
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        # print(self.model)

    def forward(self, input, layers=[], encode_only=False):
        """
        input: input
        layers: nec_layers, e.g., [0, 4, 8, 12, 16]
        encode_only: when encode_only is True, for nce loss
        """
        # -1 means the last layer
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            fake = self.model(input)
            return fake


from .blocks.Swin import PatchEmbed, BasicLayer, PixelShuffle, PixelUnshuffle
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops.layers.torch import Rearrange


class SwinTGenerator(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False,
                 input_nc=3, output_nc=3, ngf=64, use_dropout=False, n_blocks=6,
                 padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None,
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
        #     norm_layer=norm_layer if self.patch_norm else None)
        #
        # self.patch_embed = nn.Sequential(
        #     nn.Conv2d(in_chans, embed_dim, kernel_size=7, stride=4, padding=2),
        #     Rearrange('b c h w -> b (h w) c'),
        #     nn.LayerNorm(embed_dim)
        # )

        patch_embed = [nn.ReflectionPad2d(3),
                       nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=False),
                       nn.InstanceNorm2d(ngf),
                       nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            patch_embed += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.InstanceNorm2d(ngf * mult * 2),
                            nn.ReLU(True),
                            Downsample(ngf * mult * 2), ]
        patch_embed += [
            nn.Conv2d(ngf * mult * 2, self.embed_dim, kernel_size=1, stride=1, padding=0, bias=False),
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(self.embed_dim)]
        self.patch_embed = nn.Sequential(*patch_embed)

        num_patches = 64 * 64
        patches_resolution = [64, 64]
        # num_patches = self.patch_embed.num_patches
        # patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PixelUnshuffle if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # build layers
        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers - 1, -1, -1):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PixelShuffle if (i_layer > 0) else None,
                               use_checkpoint=use_checkpoint)
            self.layers_up.append(layer)

        self.norm = norm_layer(self.embed_dim)

        model_out = [Rearrange('B (H W) C -> B C H W', H=64),
                     nn.Conv2d(96, 256, 3, 1, 1)]
        for i in range(2):  # add upsampling layers
            mult = 2 ** (2 - i)
            if no_antialias_up:
                model_out += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                                 kernel_size=3, stride=2,
                                                 padding=1, output_padding=1,
                                                 bias=False),
                              nn.InstanceNorm2d(int(ngf * mult / 2)),
                              nn.ReLU(True)]
            else:
                model_out += [Upsample(ngf * mult),
                              nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                        kernel_size=3, stride=1,
                                        padding=1,  # output_padding=1,
                                        bias=False),
                              nn.InstanceNorm2d(int(64 * mult / 2)),
                              nn.ReLU(True)]
        model_out += [nn.ReflectionPad2d(3)]
        model_out += [nn.Conv2d(64, output_nc, kernel_size=7, padding=0)]
        model_out += [nn.Tanh()]
        self.model_out = nn.Sequential(*model_out)

        self.apply(self._init_weights)

    def forward_features(self, x):
        # print('input', x.shape)
        x = self.patch_embed(x)
        # print('patch_embed', x.shape)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        # print('absolute_pos_embed', x.shape)
        for layer in self.layers:
            x = layer(x)
            # print('down_layer', x.shape)
        # print('after encoding', x.shape)

        for layer in self.layers_up:
            x = layer(x)
            # print(layer)
            # print('layer_up', x.shape)

        x = self.norm(x)  # B L C

        return x

    def forward_features_encode_only(self, x, layers=[], encode_only=False):
        feats = []
        feats.append(x)

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        B, L, C = x.shape
        H = int(L ** 0.5)
        feats.append(einops.rearrange(x, 'B (H W) C -> B C H W', H=H))

        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
            B, L, C = x.shape
            H = int(L ** 0.5)
            feats.append(einops.rearrange(x, 'B (H W) C -> B C H W', H=H))

        return feats
        # for layer in self.layers_up:
        #     x = layer(x)
        #
        # x = self.norm(x)  # B L C
        #
        # return x

    def forward(self, x, layers=[], encode_only=False):
        if encode_only:
            return self.forward_features_encode_only(x, layers=layers, encode_only=encode_only)
        else:
            x = self.forward_features(x)
            x = self.model_out(x)
            return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


class ResnetGeneratorV(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGeneratorV, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        idx_down = 0
        n_downsampling = 1
        # 1: 256 (64) -> 128^2 (128)
        for _ in range(n_downsampling):  # add downsampling layers
            mult = 2 ** idx_down
            idx_down += 1
            if (no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        # self.n_blocks = self.opt.n_blocks

        # 2: 128 (128) -> 128 (128)
        n_blocks = 6
        for i in range(n_blocks):  # add ResNet blocks
            model += [Pathways()]

        # 11: 128 (128) -> 256 (64)
        n_upsampling = 1
        mult = 2
        for i in range(n_upsampling):  # add upsampling layers
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        # print(self.model)

    def forward(self, input, layers=[], encode_only=False):
        """
        input: input
        layers: nec_layers, e.g., [0, 4, 8, 12, 16]
        encode_only: when encode_only is True, for nce loss
        """
        # -1 means the last layer
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            # for m in self.model:
            #     input = m(input)
            #     print(input.shape)
            # return input
            fake = self.model(input)
            return fake


class HRKormerGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(HRKormerGenerator, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 18, 36, 72, 144
        # hr_nc = [32, 64, 128]
        hr_nc = [48, 80, 128]
        z_dim = hr_nc[0] + hr_nc[1]

        # 256 (1) -> 256 (64)
        self.stage_0 = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                                     norm_layer(ngf),
                                     nn.ReLU(True))
        # 256 (64) -> 128 (128)
        self.diverge_0 = Diverge(input_nc=ngf, output_nc=(0, 0, z_dim))

        # 128 (128) -> 128 (128)
        self.stage_1 = nn.Sequential(
            ResnetBlock(z_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias))

        self.diverge_1 = Diverge(input_nc=z_dim, output_nc=(0, hr_nc[0], hr_nc[1]))

        num_branches = [2, 3, 2]
        num_blocks = [[2, 2], [3, 3, 3], [4, 4]]
        num_inchannels = [hr_nc[0], hr_nc[1], hr_nc[2]]
        num_channels = [hr_nc[0], hr_nc[1], hr_nc[2]]
        num_heads = [[1, 2], [1, 2, 4], [1, 2]]
        num_window_sizes = [[7, 7], [7, 7, 7], [7, 7]]
        num_mlp_ratios = [[4, 4], [4, 4, 4], [4, 4]]
        reset_multi_scale_output = [True, True, True]
        drop_path = [0., 0., 0., 0.0]
        skip_connec = [False, False, False]

        self.stage_2 = nn.Sequential(HRTransBlock(num_branches=num_branches[0],
                                                  blocks=GeneralTransformerBlock,
                                                  num_blocks=num_blocks[0],
                                                  num_inchannels=num_inchannels[:num_branches[0]],
                                                  num_channels=num_channels[:num_branches[0]],
                                                  num_heads=num_heads[0],
                                                  num_window_sizes=num_window_sizes[0],
                                                  num_mlp_ratios=num_mlp_ratios[0],
                                                  multi_scale_output=reset_multi_scale_output[0],
                                                  drop_path=drop_path,
                                                  skip_connec=skip_connec[0]))

        self.diverge_21 = Diverge(input_nc=hr_nc[0], output_nc=(0, int(hr_nc[0] * 5 / 8), int(hr_nc[1] * 3 / 8)))
        self.diverge_22 = Diverge(input_nc=hr_nc[1], output_nc=(int(hr_nc[0] * 3 / 8), int(hr_nc[1] * 5 / 8), hr_nc[2]))

        self.stage_3 = nn.Sequential(HRTransBlock(num_branches=num_branches[1],
                                                  blocks=GeneralTransformerBlock,
                                                  num_blocks=num_blocks[1],
                                                  num_inchannels=num_inchannels[:num_branches[1]],
                                                  num_channels=num_channels[:num_branches[1]],
                                                  num_heads=num_heads[1],
                                                  num_window_sizes=num_window_sizes[1],
                                                  num_mlp_ratios=num_mlp_ratios[1],
                                                  multi_scale_output=reset_multi_scale_output[1],
                                                  drop_path=drop_path,
                                                  skip_connec=skip_connec[1]))

        self.diverge_31 = Diverge(input_nc=hr_nc[0], output_nc=(0, int(hr_nc[0] * 5 / 8), int(hr_nc[1] / 4)))
        self.diverge_32 = Diverge(input_nc=hr_nc[1], output_nc=(int(hr_nc[0] * 3 / 8), int(hr_nc[1] / 2), 0))
        self.diverge_33 = Diverge(input_nc=hr_nc[2], output_nc=(int(hr_nc[1] / 4), 0, 0))

        self.stage_4 = nn.Sequential(HRTransBlock(num_branches=num_branches[2],
                                                  blocks=GeneralTransformerBlock,
                                                  num_blocks=num_blocks[2],
                                                  num_inchannels=num_inchannels[:num_branches[2]],
                                                  num_channels=num_channels[:num_branches[2]],
                                                  num_heads=num_heads[2],
                                                  num_window_sizes=num_window_sizes[2],
                                                  num_mlp_ratios=num_mlp_ratios[2],
                                                  multi_scale_output=reset_multi_scale_output[2],
                                                  drop_path=drop_path,
                                                  skip_connec=skip_connec[2]))

        self.diverge_41 = Diverge(input_nc=hr_nc[0], output_nc=(0, hr_nc[0], 0))
        self.diverge_42 = Diverge(input_nc=hr_nc[1], output_nc=(hr_nc[1], 0, 0))

        self.stage_5 = nn.Sequential(
            ResnetBlock(z_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias))
        self.diverge_5 = Diverge(input_nc=z_dim, output_nc=(ngf, 0, 0))

        self.stage_6 = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                     nn.Tanh())

    def forward(self, input, layers=[], encode_only=False):
        if encode_only:
            feats = []

            x = self.stage_0(input)
            _, _, x = self.diverge_0(x)

            x = self.stage_1(x)
            _, x1, x2 = self.diverge_1(x)

            feats += [x1, x2]

            x1, x2 = self.stage_2([x1, x2])
            _, x11, x12 = self.diverge_21(x1)
            x21, x22, x23 = self.diverge_22(x2)

            x1 = torch.cat([x11, x21], dim=1)
            x2 = torch.cat([x12, x22], dim=1)
            x3 = x23
            feats += [x1, x2]

            x1, x2, x3 = self.stage_3([x1, x2, x3])
            _, x11, x12 = self.diverge_31(x1)
            x21, x22, _ = self.diverge_32(x2)
            x32, _, _ = self.diverge_33(x3)

            x1 = torch.cat([x11, x21], dim=1)
            x2 = torch.cat([x12, x22, x32], dim=1)
            feats += [x1, x2]

            # x1, x2 = self.stage_4([x1, x2])
            # _, x11, _ = self.diverge_41(x1)
            # x21, _, _ = self.diverge_42(x2)
            #
            # x1 = torch.cat([x11, x21], dim=1)
            # x = self.stage_5(x1)
            # # 128 -> 256
            # x1, _, _ = self.diverge_5(x)
            # feat = self.stage_6(x1)

            return feats
        else:
            """Standard forward"""
            x = self.stage_0(input)
            _, _, x = self.diverge_0(x)

            x = self.stage_1(x)
            _, x1, x2 = self.diverge_1(x)

            x1, x2 = self.stage_2([x1, x2])
            _, x11, x12 = self.diverge_21(x1)
            x21, x22, x23 = self.diverge_22(x2)

            x1 = torch.cat([x11, x21], dim=1)
            x2 = torch.cat([x12, x22], dim=1)
            x3 = x23
            x1, x2, x3 = self.stage_3([x1, x2, x3])
            _, x11, x12 = self.diverge_31(x1)
            x21, x22, _ = self.diverge_32(x2)
            x32, _, _ = self.diverge_33(x3)

            x1 = torch.cat([x11, x21], dim=1)
            x2 = torch.cat([x12, x22, x32], dim=1)
            x1, x2 = self.stage_4([x1, x2])
            _, x11, _ = self.diverge_41(x1)
            x21, _, _ = self.diverge_42(x2)

            x1 = torch.cat([x11, x21], dim=1)
            x = self.stage_5(x1)
            x1, _, _ = self.diverge_5(x)
            fake = self.stage_6(x1)

            return fake

            # fake = self.model(input)
            # return fake


class HRKormerGeneratorV2(nn.Module):
    """
    two pathway without fusion.
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(HRKormerGeneratorV2, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 18, 36, 72, 144
        # hr_nc = [32, 64, 128]
        hr_nc = [64, 128]
        z_dim = hr_nc[0] + hr_nc[1]

        # 256 (1) -> 256 (64)
        self.stage_0 = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                                     norm_layer(ngf),
                                     nn.ReLU(True))
        # 256 (64) -> 128 (128)
        self.diverge_0 = Diverge(input_nc=ngf, output_nc=(0, 0, z_dim))

        # 128 (128) -> 128 (128)
        self.stage_1 = nn.Sequential(
            ResnetBlock(z_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias))

        self.diverge_1 = Diverge(input_nc=z_dim, output_nc=(0, hr_nc[0], hr_nc[1]))

        self.stage_2 = ParallelBranches(num_blocks=1, inputs_nc=(hr_nc[0], hr_nc[1]),
                                        blocks_type=('ResnetBlock', 'HRTBlock'))
        self.stage_3 = ParallelBranches(num_blocks=4, inputs_nc=(hr_nc[0], hr_nc[1]),
                                        blocks_type=('ResnetBlock', 'HRTBlock'))
        self.stage_4 = ParallelBranches(num_blocks=4, inputs_nc=(hr_nc[0], hr_nc[1]),
                                        blocks_type=('ResnetBlock', 'HRTBlock'))

        self.diverge_41 = Diverge(input_nc=hr_nc[0], output_nc=(0, hr_nc[0], 0))
        self.diverge_42 = Diverge(input_nc=hr_nc[1], output_nc=(hr_nc[1], 0, 0))

        self.stage_5 = nn.Sequential(
            ResnetBlock(z_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias))
        self.diverge_5 = Diverge(input_nc=z_dim, output_nc=(ngf, 0, 0))

        self.stage_6 = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                     nn.Tanh())

    def forward(self, input, layers=[], encode_only=False):
        if encode_only:
            feats = [input]
            x = self.stage_0(input)
            _, _, x = self.diverge_0(x)

            x = self.stage_1(x)
            feats += [x]

            _, x1, x2 = self.diverge_1(x)

            x1, x2 = self.stage_2([x1, x2])
            feats += [x1, x2]
            x1, x2 = self.stage_3([x1, x2])
            feats += [x1, x2]
            # x1, x2 = self.stage_4([x1, x2])

            # x1, x2 = self.stage_4([x1, x2])
            # _, x11, _ = self.diverge_41(x1)
            # x21, _, _ = self.diverge_42(x2)
            #
            # x1 = torch.cat([x11, x21], dim=1)
            # x = self.stage_5(x1)
            # # 128 -> 256
            # x1, _, _ = self.diverge_5(x)
            # feat = self.stage_6(x1)

            return feats
        else:
            """Standard forward"""
            x = self.stage_0(input)
            _, _, x = self.diverge_0(x)

            x = self.stage_1(x)
            _, x1, x2 = self.diverge_1(x)

            x1, x2 = self.stage_2([x1, x2])
            x1, x2 = self.stage_3([x1, x2])
            x1, x2 = self.stage_4([x1, x2])

            _, x11, _ = self.diverge_41(x1)
            x21, _, _ = self.diverge_42(x2)

            x1 = torch.cat([x11, x21], dim=1)
            x = self.stage_5(x1)
            x1, _, _ = self.diverge_5(x)
            fake = self.stage_6(x1)

            return fake

            # fake = self.model(input)
            # return fake


class HRKormerGeneratorV3(nn.Module):
    """
    two pathway without fusion.
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(HRKormerGeneratorV3, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 18, 36, 72, 144
        # hr_nc = [32, 64, 128]
        hr_nc = [64, 128]
        z_dim = hr_nc[0] + hr_nc[1]

        # 256 (1) -> 256 (64)
        self.stage_0 = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                                     norm_layer(ngf),
                                     nn.ReLU(True))
        # 256 (64) -> 128 (128)
        self.diverge_0 = Diverge(input_nc=ngf, output_nc=(0, 0, z_dim))

        # 128 (128) -> 128 (128)
        self.stage_1 = nn.Sequential(
            ResnetBlock(z_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias))

        self.diverge_1 = Diverge(input_nc=z_dim, output_nc=(0, hr_nc[0], hr_nc[1]))

        self.stage_2 = ParallelBranches(num_blocks=1, inputs_nc=(hr_nc[0], hr_nc[1]),
                                        blocks_type=('HRTBlock', 'ResnetBlock'))
        self.stage_3 = ParallelBranches(num_blocks=4, inputs_nc=(hr_nc[0], hr_nc[1]),
                                        blocks_type=('HRTBlock', 'ResnetBlock'))
        self.stage_4 = ParallelBranches(num_blocks=4, inputs_nc=(hr_nc[0], hr_nc[1]),
                                        blocks_type=('HRTBlock', 'ResnetBlock'))

        self.diverge_41 = Diverge(input_nc=hr_nc[0], output_nc=(0, hr_nc[0], 0))
        self.diverge_42 = Diverge(input_nc=hr_nc[1], output_nc=(hr_nc[1], 0, 0))

        self.stage_5 = nn.Sequential(
            ResnetBlock(z_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias))
        self.diverge_5 = Diverge(input_nc=z_dim, output_nc=(ngf, 0, 0))

        self.stage_6 = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                     nn.Tanh())

    def forward(self, input, layers=[], encode_only=False):
        if encode_only:
            feats = [input]
            x = self.stage_0(input)
            _, _, x = self.diverge_0(x)

            x = self.stage_1(x)
            feats += [x]

            _, x1, x2 = self.diverge_1(x)

            x1, x2 = self.stage_2([x1, x2])
            feats += [x1, x2]
            x1, x2 = self.stage_3([x1, x2])
            feats += [x1, x2]
            # x1, x2 = self.stage_4([x1, x2])

            # x1, x2 = self.stage_4([x1, x2])
            # _, x11, _ = self.diverge_41(x1)
            # x21, _, _ = self.diverge_42(x2)
            #
            # x1 = torch.cat([x11, x21], dim=1)
            # x = self.stage_5(x1)
            # # 128 -> 256
            # x1, _, _ = self.diverge_5(x)
            # feat = self.stage_6(x1)

            return feats
        else:
            """Standard forward"""
            x = self.stage_0(input)
            _, _, x = self.diverge_0(x)

            x = self.stage_1(x)
            _, x1, x2 = self.diverge_1(x)

            x1, x2 = self.stage_2([x1, x2])
            x1, x2 = self.stage_3([x1, x2])
            x1, x2 = self.stage_4([x1, x2])

            _, x11, _ = self.diverge_41(x1)
            x21, _, _ = self.diverge_42(x2)

            x1 = torch.cat([x11, x21], dim=1)
            x = self.stage_5(x1)
            x1, _, _ = self.diverge_5(x)
            fake = self.stage_6(x1)

            return fake

            # fake = self.model(input)
            # return fake


class HRKormerGeneratorV4(nn.Module):
    """
    two pathway without fusion.
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(HRKormerGeneratorV4, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 18, 36, 72, 144
        # hr_nc = [32, 64, 128]
        hr_nc = [64, 128]
        z_dim = hr_nc[0] + hr_nc[1]

        # 256 (1) -> 256 (64)
        self.stage_0 = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                                     norm_layer(ngf),
                                     nn.ReLU(True))
        # 256 (64) -> 128 (128)
        self.diverge_0 = Diverge(input_nc=ngf, output_nc=(0, 0, z_dim))

        # 128 (128) -> 128 (128)
        self.stage_1 = nn.Sequential(
            ResnetBlock(z_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias))

        self.diverge_1 = Diverge(input_nc=z_dim, output_nc=(0, hr_nc[0], hr_nc[1]))

        self.stage_2 = ParallelBranches(num_blocks=1, inputs_nc=(hr_nc[0], hr_nc[1]),
                                        blocks_type=('ResnetBlock', 'ResnetBlock'))
        self.stage_3 = ParallelBranches(num_blocks=4, inputs_nc=(hr_nc[0], hr_nc[1]),
                                        blocks_type=('ResnetBlock', 'ResnetBlock'))
        self.stage_4 = ParallelBranches(num_blocks=4, inputs_nc=(hr_nc[0], hr_nc[1]),
                                        blocks_type=('ResnetBlock', 'ResnetBlock'))

        self.diverge_41 = Diverge(input_nc=hr_nc[0], output_nc=(0, hr_nc[0], 0))
        self.diverge_42 = Diverge(input_nc=hr_nc[1], output_nc=(hr_nc[1], 0, 0))

        self.stage_5 = nn.Sequential(
            ResnetBlock(z_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias))
        self.diverge_5 = Diverge(input_nc=z_dim, output_nc=(ngf, 0, 0))

        self.stage_6 = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                     nn.Tanh())

    def forward(self, input, layers=[], encode_only=False):
        if encode_only:
            feats = [input]
            x = self.stage_0(input)
            _, _, x = self.diverge_0(x)

            x = self.stage_1(x)
            feats += [x]

            _, x1, x2 = self.diverge_1(x)

            x1, x2 = self.stage_2([x1, x2])
            feats += [x1, x2]
            x1, x2 = self.stage_3([x1, x2])
            feats += [x1, x2]
            # x1, x2 = self.stage_4([x1, x2])

            # x1, x2 = self.stage_4([x1, x2])
            # _, x11, _ = self.diverge_41(x1)
            # x21, _, _ = self.diverge_42(x2)
            #
            # x1 = torch.cat([x11, x21], dim=1)
            # x = self.stage_5(x1)
            # # 128 -> 256
            # x1, _, _ = self.diverge_5(x)
            # feat = self.stage_6(x1)

            return feats
        else:
            """Standard forward"""
            x = self.stage_0(input)
            _, _, x = self.diverge_0(x)

            x = self.stage_1(x)
            _, x1, x2 = self.diverge_1(x)

            x1, x2 = self.stage_2([x1, x2])
            x1, x2 = self.stage_3([x1, x2])
            x1, x2 = self.stage_4([x1, x2])

            _, x11, _ = self.diverge_41(x1)
            x21, _, _ = self.diverge_42(x2)

            x1 = torch.cat([x11, x21], dim=1)
            x = self.stage_5(x1)
            x1, _, _ = self.diverge_5(x)
            fake = self.stage_6(x1)

            return fake

            # fake = self.model(input)
            # return fake


class HRKormerGeneratorV(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(HRKormerGeneratorV, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 18, 36, 72, 144
        # hr_nc = [32, 64, 128]
        hr_nc = [48, 80, 128]
        z_dim = hr_nc[0] + hr_nc[1]

        # 256 (1) -> 256 (64)
        self.stage_0 = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                                     norm_layer(ngf),
                                     nn.ReLU(True))
        # 256 (64) -> 128 (128)
        self.diverge_0 = Diverge(input_nc=ngf, output_nc=(0, 0, z_dim))

        # 128 (128) -> 128 (128)
        self.stage_1 = nn.Sequential(
            ResnetBlock(z_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias))

        self.diverge_1 = Diverge(input_nc=z_dim, output_nc=(0, hr_nc[0], hr_nc[1]))

        num_branches = [2, 3, 2]
        num_blocks = [[2, 2], [3, 3, 3], [4, 4]]
        num_inchannels = [hr_nc[0], hr_nc[1], hr_nc[2]]
        num_channels = [hr_nc[0], hr_nc[1], hr_nc[2]]
        num_heads = [[1, 2], [1, 2, 4], [1, 2]]
        num_window_sizes = [[7, 7], [7, 7, 7], [7, 7]]
        num_mlp_ratios = [[4, 4], [4, 4, 4], [4, 4]]
        reset_multi_scale_output = [True, True, True]
        drop_path = [0., 0., 0., 0.0]
        skip_connec = [False, False, False]

        self.stage_2 = nn.Sequential(HRTransBlock(num_branches=num_branches[0],
                                                  blocks=GeneralTransformerBlock,
                                                  num_blocks=num_blocks[0],
                                                  num_inchannels=num_inchannels[:num_branches[0]],
                                                  num_channels=num_channels[:num_branches[0]],
                                                  num_heads=num_heads[0],
                                                  num_window_sizes=num_window_sizes[0],
                                                  num_mlp_ratios=num_mlp_ratios[0],
                                                  multi_scale_output=reset_multi_scale_output[0],
                                                  drop_path=drop_path,
                                                  skip_connec=skip_connec[0]))

        self.diverge_21 = Diverge(input_nc=hr_nc[0], output_nc=(0, int(hr_nc[0] * 5 / 8), int(hr_nc[1] * 3 / 8)))
        self.diverge_22 = Diverge(input_nc=hr_nc[1], output_nc=(int(hr_nc[0] * 3 / 8), int(hr_nc[1] * 5 / 8), hr_nc[2]))

        self.stage_3 = nn.Sequential(HRTransBlock(num_branches=num_branches[1],
                                                  blocks=GeneralTransformerBlock,
                                                  num_blocks=num_blocks[1],
                                                  num_inchannels=num_inchannels[:num_branches[1]],
                                                  num_channels=num_channels[:num_branches[1]],
                                                  num_heads=num_heads[1],
                                                  num_window_sizes=num_window_sizes[1],
                                                  num_mlp_ratios=num_mlp_ratios[1],
                                                  multi_scale_output=reset_multi_scale_output[1],
                                                  drop_path=drop_path,
                                                  skip_connec=skip_connec[1]))

        self.diverge_31 = Diverge(input_nc=hr_nc[0], output_nc=(0, int(hr_nc[0] * 5 / 8), int(hr_nc[1] / 4)))
        self.diverge_32 = Diverge(input_nc=hr_nc[1], output_nc=(int(hr_nc[0] * 3 / 8), int(hr_nc[1] / 2), 0))
        self.diverge_33 = Diverge(input_nc=hr_nc[2], output_nc=(int(hr_nc[1] / 4), 0, 0))

        self.stage_4 = nn.Sequential(HRTransBlock(num_branches=num_branches[2],
                                                  blocks=GeneralTransformerBlock,
                                                  num_blocks=num_blocks[2],
                                                  num_inchannels=num_inchannels[:num_branches[2]],
                                                  num_channels=num_channels[:num_branches[2]],
                                                  num_heads=num_heads[2],
                                                  num_window_sizes=num_window_sizes[2],
                                                  num_mlp_ratios=num_mlp_ratios[2],
                                                  multi_scale_output=reset_multi_scale_output[2],
                                                  drop_path=drop_path,
                                                  skip_connec=skip_connec[2]))

        self.diverge_41 = Diverge(input_nc=hr_nc[0], output_nc=(0, hr_nc[0], 0))
        self.diverge_42 = Diverge(input_nc=hr_nc[1], output_nc=(hr_nc[1], 0, 0))

        self.stage_5 = nn.Sequential(
            ResnetBlock(z_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias))
        self.diverge_5 = Diverge(input_nc=z_dim, output_nc=(ngf, 0, 0))

        self.stage_6 = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                     nn.Tanh())

    def forward(self, input, layers=[], encode_only=False):
        if encode_only:
            feats = [input]

            x = self.stage_0(input)
            _, _, x = self.diverge_0(x)
            feats += [x]

            x = self.stage_1(x)
            _, x1, x2 = self.diverge_1(x)
            feats += [torch.cat([x1, F.interpolate(x2, scale_factor=2, mode='bicubic')], dim=1)]

            x1, x2 = self.stage_2([x1, x2])
            _, x11, x12 = self.diverge_21(x1)
            x21, x22, x23 = self.diverge_22(x2)

            x1 = torch.cat([x11, x21], dim=1)
            x2 = torch.cat([x12, x22], dim=1)
            x3 = x23
            feats += [
                torch.cat([x1, F.interpolate(x2, scale_factor=2, mode='bicubic'),
                           F.interpolate(x3, scale_factor=4, mode='bicubic')], dim=1)]

            x1, x2, x3 = self.stage_3([x1, x2, x3])
            _, x11, x12 = self.diverge_31(x1)
            x21, x22, _ = self.diverge_32(x2)
            x32, _, _ = self.diverge_33(x3)

            x1 = torch.cat([x11, x21], dim=1)
            x2 = torch.cat([x12, x22, x32], dim=1)
            feats += [torch.cat([x1, F.interpolate(x2, scale_factor=2, mode='bicubic')], dim=1)]

            # x1, x2 = self.stage_4([x1, x2])
            # _, x11, _ = self.diverge_41(x1)
            # x21, _, _ = self.diverge_42(x2)
            #
            # x1 = torch.cat([x11, x21], dim=1)
            # x = self.stage_5(x1)
            # # 128 -> 256
            # x1, _, _ = self.diverge_5(x)
            # feat = self.stage_6(x1)

            return feats
        else:
            """Standard forward"""
            x = self.stage_0(input)
            _, _, x = self.diverge_0(x)

            x = self.stage_1(x)
            _, x1, x2 = self.diverge_1(x)

            x1, x2 = self.stage_2([x1, x2])
            _, x11, x12 = self.diverge_21(x1)
            x21, x22, x23 = self.diverge_22(x2)

            x1 = torch.cat([x11, x21], dim=1)
            x2 = torch.cat([x12, x22], dim=1)
            x3 = x23
            x1, x2, x3 = self.stage_3([x1, x2, x3])
            _, x11, x12 = self.diverge_31(x1)
            x21, x22, _ = self.diverge_32(x2)
            x32, _, _ = self.diverge_33(x3)

            x1 = torch.cat([x11, x21], dim=1)
            x2 = torch.cat([x12, x22, x32], dim=1)
            x1, x2 = self.stage_4([x1, x2])
            _, x11, _ = self.diverge_41(x1)
            x21, _, _ = self.diverge_42(x2)

            x1 = torch.cat([x11, x21], dim=1)
            x = self.stage_5(x1)
            x1, _, _ = self.diverge_5(x)
            fake = self.stage_6(x1)

            return fake

            # fake = self.model(input)
            # return fake


class Diverge(nn.Module):
    def __init__(self, input_nc=64, output_nc=(0, 64, 128), norm_layer=nn.InstanceNorm2d, use_bias=False):
        """
        output_nc=[0, 64, 128] means
           ↗  0
        64 -> 64
           ↘  128
        """
        super().__init__()
        if output_nc[0] > 0:
            self.up_branch = nn.Sequential(
                Upsample(input_nc),
                nn.Conv2d(input_nc, output_nc[0], kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(output_nc[0]),
                nn.ReLU(True)
            )
        if output_nc[1] > 0:
            self.same_branch = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(input_nc, output_nc[1], kernel_size=3, stride=1, padding=0, bias=use_bias),
                norm_layer(output_nc[1]),
                nn.ReLU(True)
            )
        if output_nc[2] > 0:
            self.down_branch = nn.Sequential(
                nn.Conv2d(input_nc, output_nc[2], kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(output_nc[2]),
                nn.ReLU(True),
                Downsample(output_nc[2])
            )
        self.output_nc = output_nc

    def forward(self, x):
        up, same, down = None, None, None
        if self.output_nc[0] > 0:
            up = self.up_branch(x)
        if self.output_nc[1] > 0:
            same = self.same_branch(x)
        if self.output_nc[2] > 0:
            down = self.down_branch(x)
        return up, same, down


class ParallelBranches(nn.Module):
    def __init__(self, num_blocks=3, inputs_nc=(256, 256, 256),
                 blocks_type=('ResnetBlock', 'ResnetBlock', 'ResnetBlock')):
        super().__init__()
        self.blocks_type = blocks_type
        for idx, blk in enumerate(blocks_type):
            if blk == 'ResnetBlock':
                branch = []
                for i in range(num_blocks):
                    branch += [ResnetBlock(inputs_nc[idx], padding_type='reflect', norm_layer=nn.InstanceNorm2d,
                                           use_dropout=False, use_bias=False)]
                exec(f"self.branch_{idx} = nn.Sequential(*branch)")

            if blk == 'HRTBlock':
                branch = []
                for i in range(num_blocks):
                    branch += [HRTransBlock(num_branches=1,
                                            blocks=GeneralTransformerBlock,
                                            num_blocks=[1],
                                            num_inchannels=[inputs_nc[idx]],
                                            num_channels=[inputs_nc[idx]],
                                            num_heads=[idx + 1],
                                            num_window_sizes=[7],
                                            num_mlp_ratios=[4],
                                            multi_scale_output=False,
                                            drop_path=[0.],
                                            skip_connec=False)]
                exec(f"self.branch_{idx} = nn.Sequential(*branch)")

    def forward(self, x: list):
        res = []
        for i in range(len(self.blocks_type)):
            out = eval(f'self.branch_{i}')(x[i])
            res.append(out)
        return res


class HGHRTGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(HGHRTGenerator, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        idx_down = 0
        n_downsampling = 1
        # 1: 256 (64) -> 128 (128)
        for _ in range(n_downsampling):  # add downsampling layers
            mult = 2 ** idx_down
            idx_down += 1
            if (no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        # self.n_blocks = self.opt.n_blocks

        num_branches = 1
        block = GeneralTransformerBlock
        num_blocks = [1]
        num_inchannels = [128]
        num_channels = [128]
        num_heads = [2]
        num_window_sizes = [7]
        num_mlp_ratios = [4]
        reset_multi_scale_output = False
        drop_path = [0., 0.]
        skip_connec = self.opt.skip_connec

        # 2: 128 (128) -> 128 (128)
        n_blocks = 1
        for i in range(n_blocks):  # add ResNet blocks
            mult = 2 ** idx_down
            model += [HRTransBlock(num_branches,
                                   block,  # TRANSFORMER_BLOCK
                                   num_blocks,  # 2
                                   num_inchannels,  # 256
                                   num_channels,  # 256
                                   num_heads,  # 2
                                   num_window_sizes,  # 7
                                   num_mlp_ratios,  # 4
                                   reset_multi_scale_output,
                                   drop_path=drop_path,
                                   skip_connec=skip_connec)]

        n_downsampling = 1
        # 3: 128 (128) -> 64 (256)
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** idx_down
            idx_down += 1
            if (no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        # 4: 64 (256) -> 64 (256)
        n_blocks = 2
        for i in range(n_blocks):  # add ResNet blocks
            mult = 2 ** idx_down
            model += [HRTransBlock(num_branches,
                                   block,  # TRANSFORMER_BLOCK
                                   num_blocks,  # 2
                                   num_inchannels,  # 256
                                   num_channels,  # 256
                                   num_heads,  # 2
                                   num_window_sizes,  # 7
                                   num_mlp_ratios,  # 4
                                   reset_multi_scale_output,
                                   drop_path=drop_path,
                                   skip_connec=skip_connec)]

        n_upsampling = 1
        idx_up = idx_down
        # 5: 64 (256) -> 128 (128)
        for i in range(n_upsampling):  # add upsampling layers
            mult = 2 ** idx_up
            idx_up -= 1
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]

        n_blocks = 1
        # 6: 128 (128) -> 128 (128)
        for i in range(n_blocks):  # add ResNet blocks
            mult = 2 ** idx_up
            model += [HRTransBlock(num_branches,
                                   block,  # TRANSFORMER_BLOCK
                                   num_blocks,  # 2
                                   num_inchannels,  # 256
                                   num_channels,  # 256
                                   num_heads,  # 2
                                   num_window_sizes,  # 7
                                   num_mlp_ratios,  # 4
                                   reset_multi_scale_output,
                                   drop_path=drop_path,
                                   skip_connec=skip_connec)]

        n_downsampling = 1
        # 7: 128 (128) -> 64 (256)
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** (idx_down - 1)
            if (no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        # 8: 64 (256) -> 64 (256)
        n_blocks = 2
        for i in range(n_blocks):  # add ResNet blocks
            mult = 2 ** idx_down
            model += [HRTransBlock(num_branches,
                                   block,  # TRANSFORMER_BLOCK
                                   num_blocks,  # 2
                                   num_inchannels,  # 256
                                   num_channels,  # 256
                                   num_heads,  # 2
                                   num_window_sizes,  # 7
                                   num_mlp_ratios,  # 4
                                   reset_multi_scale_output,
                                   drop_path=drop_path,
                                   skip_connec=skip_connec)]

        # 9: 64 (256) -> 128 (128)
        n_upsampling = 1
        for i in range(n_upsampling):  # add upsampling layers
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]

        # 10: 128 (128) -> 128 (128)
        n_blocks = 1
        for i in range(n_blocks):  # add ResNet blocks
            mult = 2 ** (idx_down - 1)

            model += [HRTransBlock(num_branches,
                                   block,  # TRANSFORMER_BLOCK
                                   num_blocks,  # 2
                                   num_inchannels,  # 256
                                   num_channels,  # 256
                                   num_heads,  # 2
                                   num_window_sizes,  # 7
                                   num_mlp_ratios,  # 4
                                   reset_multi_scale_output,
                                   drop_path=drop_path,
                                   skip_connec=skip_connec)]

        # 11: 128 (128) -> 256 (64)
        for i in range(n_upsampling):  # add upsampling layers
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        # print(self.model)

    def forward(self, input, layers=[], encode_only=False):
        """
        input: input
        layers: nec_layers, e.g., [0, 4, 8, 12, 16]
        encode_only: when encode_only is True, for nce loss
        """
        # -1 means the last layer
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            fake = self.model(input)
            return fake


class Pathways(nn.Module):
    def __init__(self):
        super(Pathways, self).__init__()
        ngf = 64
        mult = 2
        use_bias = True
        norm_layer = nn.InstanceNorm2d
        padding_type = 'reflect'
        use_dropout = False
        self.up_branch = nn.Sequential(*[
            Upsample(ngf * mult),
            nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(int(ngf * mult / 2)),
            nn.ReLU(True),
            nn.Conv2d(int(ngf * mult / 2), ngf * mult, kernel_size=1, stride=1, padding=0, bias=use_bias),
            Downsample(ngf * mult),
        ])
        self.sam_branch = nn.Sequential(*[
            ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias)
        ])
        self.dow_branch = nn.Sequential(*[
            Downsample(ngf * mult),
            nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ngf * mult * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1, stride=1, padding=0, bias=use_bias),
            Upsample(ngf * mult),
        ])

    def forward(self, x):
        up = self.up_branch(x)
        sam = self.sam_branch(x)
        dow = self.dow_branch(x)
        return x + up + sam + dow


class HGEncoder(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(HGEncoder, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        idx_down = 0
        n_downsampling = 1
        # 1: 256 (64) -> 128 (128)
        for _ in range(n_downsampling):  # add downsampling layers
            mult = 2 ** idx_down
            idx_down += 1
            if (no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        # self.n_blocks = self.opt.n_blocks

        # 2: 128 (128) -> 128 (128)
        n_blocks = 1
        for i in range(n_blocks):  # add ResNet blocks
            mult = 2 ** idx_down
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        n_downsampling = 1
        # 3: 128 (128) -> 64 (256)
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** idx_down
            idx_down += 1
            if (no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        # 4: 64 (256) -> 64 (256)
        n_blocks = 2
        for i in range(n_blocks):  # add ResNet blocks
            mult = 2 ** idx_down
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        n_upsampling = 1
        idx_up = idx_down
        # 5: 64 (256) -> 128 (128)
        for i in range(n_upsampling):  # add upsampling layers
            mult = 2 ** idx_up
            idx_up -= 1
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]

        n_blocks = 2
        # 6: 128 (128) -> 128 (128)
        for i in range(n_blocks):  # add ResNet blocks
            mult = 2 ** idx_up
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        self.model = nn.Sequential(*model)

    def forward(self, input, layers=[], encode_only=False):
        """
        input: input
        layers: nec_layers, e.g., [0, 4, 8, 12, 16]
        encode_only: when encode_only is True, for nce loss
        """
        # -1 means the last layer
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            fake = self.model(input)
            return fake


class HGDecoder(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(HGDecoder, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []

        n_blocks = 2
        # 6: 128 (128) -> 128 (128)
        for i in range(n_blocks):  # add ResNet blocks
            mult = 2
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        n_downsampling = 1
        # 7: 128 (128) -> 64 (256)
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2
            if (no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        # 8: 64 (256) -> 64 (256)
        n_blocks = 2
        for i in range(n_blocks):  # add ResNet blocks
            mult = 4
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        # 9: 64 (256) -> 128 (128)
        n_upsampling = 1
        for i in range(n_upsampling):  # add upsampling layers
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]

        # 10: 128 (128) -> 128 (128)
        n_blocks = 1
        for i in range(n_blocks):  # add ResNet blocks
            mult = 2
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        # 11: 128 (128) -> 256 (64)
        for i in range(n_upsampling):  # add upsampling layers
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        # print(self.model)

    def forward(self, input, layers=[], encode_only=False):
        """
        input: input
        layers: nec_layers, e.g., [0, 4, 8, 12, 16]
        encode_only: when encode_only is True, for nce loss
        """
        # -1 means the last layer
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            fake = self.model(input)
            return fake


class ResnetGeneratorV5(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGeneratorV5, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        idx_down = 0
        n_downsampling = 1
        # 1: 256 (64) -> 128 (128)
        for _ in range(n_downsampling):  # add downsampling layers
            mult = 2 ** idx_down
            idx_down += 1
            if (no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        # self.n_blocks = self.opt.n_blocks

        # 2: 128 (128) -> 128 (128)
        n_blocks = 1
        for i in range(n_blocks):  # add ResNet blocks
            mult = 2 ** idx_down
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        n_downsampling = 1
        # 3: 128 (128) -> 64 (192)
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** idx_down
            idx_down += 1
            if (no_antialias):
                model += [
                    nn.Conv2d(ngf * mult, int(ngf * mult * 2.25), kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(int(ngf * mult * 2.25)),
                    nn.ReLU(True)]
            else:
                model += [
                    nn.Conv2d(ngf * mult, int(ngf * mult * 2.25), kernel_size=3, stride=1, padding=1, bias=use_bias),
                    norm_layer(int(ngf * mult * 2.25)),
                    nn.ReLU(True),
                    Downsample(int(ngf * mult * 2.25))]

        # 4: 64 (192) -> 64 (192)
        n_blocks = 2
        for i in range(n_blocks):  # add ResNet blocks
            mult = 2 ** idx_down
            model += [ResnetBlock(int(ngf * mult * 1.125), padding_type=padding_type, norm_layer=norm_layer,
                                  use_dropout=use_dropout,
                                  use_bias=use_bias),
                      HierBlock(int(ngf * mult * 1.125), padding_type=padding_type, norm_layer=norm_layer,
                                use_dropout=use_dropout,
                                use_bias=use_bias)]
        model += [ResnetBlock(int(ngf * mult * 1.125), padding_type=padding_type, norm_layer=norm_layer,
                              use_dropout=use_dropout,
                              use_bias=use_bias)]
        n_upsampling = 1
        idx_up = idx_down
        # 5: 64 (192) -> 128 (128)
        for i in range(n_upsampling):  # add upsampling layers
            mult = 2 ** idx_up
            idx_up -= 1
            if no_antialias_up:
                model += [nn.ConvTranspose2d(int(ngf * mult * 1.125), 128,
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(128),
                          nn.ReLU(True)]
            else:
                model += [Upsample(int(ngf * mult * 1.125)),
                          nn.Conv2d(int(ngf * mult * 1.125), 128,
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(128),
                          nn.ReLU(True)]
        # print("", ngf * mult)
        n_blocks = 1
        # 6: 128 (128) -> 128 (128)
        for i in range(n_blocks):  # add ResNet blocks
            mult = 2 ** idx_up
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        # 11: 128 (128) -> 256 (64)
        for i in range(n_upsampling):  # add upsampling layers
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        print(self.model)

    def forward(self, input, layers=[], encode_only=False):
        """
        input: input
        layers: nec_layers, e.g., [0, 4, 8, 12, 16]
        encode_only: when encode_only is True, for nce loss
        """
        # -1 means the last layer
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            fake = self.model(input)
            return fake


class RednetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(RednetGenerator, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model_in = [nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True)]

        n_downsampling = 2
        # for i in range(n_downsampling):  # add downsampling layers
        #     mult = 2 ** i
        #     if (no_antialias):
        #         model_in += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
        #                      norm_layer(ngf * mult * 2),
        #                      nn.ReLU(True)]
        #     else:
        #         model_in += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
        #                      norm_layer(ngf * mult * 2),
        #                      nn.ReLU(True),
        #                      Downsample(ngf * mult * 2)]

        model = []
        mult = 2 ** n_downsampling

        self.n_blocks = 6

        for i in range(self.n_blocks):  # add ResNet blocks

            # model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
            #                       use_bias=use_bias)]
            num_branches = 1
            block = GeneralTransformerBlock
            num_blocks = [1]
            num_inchannels = [256]
            num_channels = [256]
            num_heads = [4]
            num_window_sizes = [7]
            num_mlp_ratios = [4]
            reset_multi_scale_output = False
            drop_path = [0., 0.]

            # model += [nn.ModuleList([
            #     HRTransBlock(num_branches,
            #                  block,  # TRANSFORMER_BLOCK
            #                  num_blocks,  # 2
            #                  num_inchannels,  # 256
            #                  num_channels,  # 256
            #                  num_heads,  # 2
            #                  num_window_sizes,  # 7
            #                  num_mlp_ratios,  # 4
            #                  reset_multi_scale_output,
            #                  drop_path=drop_path),
            #     ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
            #                 use_bias=use_bias),
            #     MBInvertedConvLayer(in_channels=ngf * mult, out_channels=ngf * mult, kernel_size=3, stride=1, expand_ratio=6, mid_channels=None),
            #     nn.Sequential(*[ShuffleBlock(in_channels=ngf * mult // 2, out_channels=ngf * mult, kernel=7, stride=1),
            #                     ShuffleBlockX(in_channels=ngf * mult // 2, out_channels=ngf * mult, stride=1),
            #                     ShuffleBlockX(in_channels=ngf * mult // 2, out_channels=ngf * mult, stride=1),
            #                    ShuffleBlock(in_channels=ngf * mult // 2, out_channels=ngf * mult, kernel=7, stride=1)])
            #     # nn.Sequential(*[])
            # ])]
            from .blocks.involution_blocks import Involution2d
            model += [
                HRTransBlock(num_branches,
                             block,  # TRANSFORMER_BLOCK
                             num_blocks,  # 2
                             num_inchannels,  # 256
                             num_channels,  # 256
                             num_heads,  # 2
                             num_window_sizes,  # 7
                             num_mlp_ratios,  # 4
                             reset_multi_scale_output,
                             drop_path=drop_path)
            ]
        model_out = []
        # for i in range(n_downsampling):  # add upsampling layers
        #     mult = 2 ** (n_downsampling - i)
        #     if no_antialias_up:
        #         model_out += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
        #                                          kernel_size=3, stride=2,
        #                                          padding=1, output_padding=1,
        #                                          bias=use_bias),
        #                       norm_layer(int(ngf * mult / 2)),
        #                       nn.ReLU(True)]
        #     else:
        #         model_out += [Upsample(ngf * mult),
        #                       nn.Conv2d(ngf * mult, int(ngf * mult / 2),
        #                                 kernel_size=3, stride=1,
        #                                 padding=1,  # output_padding=1,
        #                                 bias=use_bias),
        #                       norm_layer(int(ngf * mult / 2)),
        #                       nn.ReLU(True)]
        model_out += [nn.ReflectionPad2d(3)]
        model_out += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_out += [nn.Tanh()]

        self.model_in = nn.Sequential(*model_in)
        self.model = nn.Sequential(*model)
        self.model_out = nn.Sequential(*model_out)

    def forward(self, input, layers=[], encode_only=False, choice=None):
        """
        input: input
        layers: nec_layers, e.g., [0, 4, 8, 12, 16]
        encode_only: when encode_only is True, for nce loss
        """
        # -1 means the last layer
        if -1 in layers:
            layers.append(len(self.model_in) + len(self.model) + len(self.model_out))
        if len(layers) > 0:
            feat = input
            feats = []
            layer_id = 0
            for layer in self.model_in:
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers
                layer_id += 1

            for i, j in enumerate(choice):
                feat = self.model[i][j](feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers
                layer_id += 1

            for layer in self.model_out:
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers
                layer_id += 1

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            x = self.model_in(input)
            # print(x.shape)
            x = window_partition(x, window_size=64)
            # print(x.shape)
            x = self.model(x)
            # for i, j in enumerate(choice):
            #     x = self.model[i][j](x)
            x = window_reverse(windows=x, window_size=64, H=256, W=256)
            # print(x.shape)
            fake = self.model_out(x)
            return fake


class ResnetDecoder(nn.Module):
    """Resnet-based decoder that consists of a few Resnet blocks + a few upsampling operations.
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', no_antialias=False):
        """Construct a Resnet-based decoder

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetDecoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = []
        n_downsampling = 2
        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if (no_antialias):
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetEncoder(nn.Module):
    """Resnet-based encoder that consists of a few downsampling + several Resnet blocks
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', no_antialias=False):
        """Construct a Resnet-based encoder

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetEncoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if (no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetBlockV2(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, dim_out, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlockV2, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

        self.pwc = nn.Conv2d(dim, dim_out, kernel_size=1, padding=0, bias=False)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return self.pwc(out)


class DSCBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(DSCBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        # conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, groups=dim, bias=use_bias),
                       norm_layer(dim),
                       nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class HierBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(HierBlock, self).__init__()
        self.dim = dim
        dim = self.dim // 3
        # self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

        upper_branch = [Upsample(dim),
                        nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
                        norm_layer(dim),
                        nn.ReLU(True),
                        #
                        ResnetBlock(dim, padding_type=padding_type, norm_layer=norm_layer,
                                    use_dropout=use_dropout,
                                    use_bias=use_bias),
                        #
                        nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
                        norm_layer(dim),
                        nn.ReLU(True),
                        Downsample(dim)
                        ]
        middle_branch = [ResnetBlock(dim, padding_type=padding_type, norm_layer=norm_layer,
                                     use_dropout=use_dropout,
                                     use_bias=use_bias),
                         ResnetBlock(dim, padding_type=padding_type, norm_layer=norm_layer,
                                     use_dropout=use_dropout,
                                     use_bias=use_bias)
                         ]
        bottom_branch = [nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
                         norm_layer(dim),
                         nn.ReLU(True),
                         Downsample(dim),
                         #
                         ResnetBlock(dim, padding_type=padding_type, norm_layer=norm_layer,
                                     use_dropout=use_dropout,
                                     use_bias=use_bias),
                         #
                         Upsample(dim),
                         nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
                         norm_layer(dim),
                         nn.ReLU(True),
                         ]

        self.upper_branch = nn.Sequential(*upper_branch)
        self.middle_branch = nn.Sequential(*middle_branch)
        self.bottom_branch = nn.Sequential(*bottom_branch)

    def forward(self, x):
        """Forward function (with skip connections)"""
        up, mid, bot = torch.chunk(x, 3, dim=1)
        out = torch.cat([self.upper_branch(up), self.middle_branch(mid), self.bottom_branch(bot)], dim=1)
        return out


class RednetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(RednetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        from .blocks.involution_blocks import Involution2d
        conv_block += [Involution2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDoDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=256, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDoDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if (no_antialias):
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True),
                        Downsample(ndf)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if (no_antialias):
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    # Downsample(ndf * nf_mult)
                ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class NLayerCDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerCDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model_in = []
        if (no_antialias):
            model_in += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0),
                nn.LeakyReLU(0.2, True)
            ]
        else:
            model_in += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(input_nc, ndf, kernel_size=4, stride=1, padding=0),
                nn.LeakyReLU(0.2, True),
                Downsample(ndf)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if (no_antialias):
                model_in += [
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=0, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                model_in += [
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=0, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)
                ]
        # print(ndf * nf_mult)
        # 256, 32, 32
        self.model_o1 = nn.Sequential(
            # 32 -> 16
            nn.ReflectionPad2d(1),
            nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=3, padding=0, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            Downsample(ndf * nf_mult),
            # 16 -> 8
            nn.ReflectionPad2d(1),
            nn.Conv2d(ndf * nf_mult, ndf * nf_mult * 2, kernel_size=3, padding=0, bias=use_bias),
            norm_layer(ndf * nf_mult * 2),
            nn.LeakyReLU(0.2, True),
            Downsample(ndf * nf_mult * 2),
            # 8 -> 1
            nn.AdaptiveAvgPool2d(1)
        )
        self.projection = nn.Sequential(
            nn.Linear(ndf * nf_mult * 2, ndf * nf_mult * 2, bias=False),
            nn.ReLU(),
            nn.Linear(ndf * nf_mult * 2, ndf * nf_mult * 2, bias=False),
        )

        self.model_in = nn.Sequential(*model_in)

    def forward(self, input):
        """Standard forward."""
        B = input.shape[0]

        x = self.model_in(input)

        features = self.model_o1(x)
        # features = self.projection(features.view(B, -1))
        features = F.normalize(self.projection(features.view(B, -1)), dim=1)
        logits = torch.einsum('ac,bc->ab', features[B // 2:], features).unsqueeze(0)

        return logits.unsqueeze(1)


class NLayerContrastiveDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerContrastiveDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model_in = []
        if (no_antialias):
            model_in += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0),
                nn.LeakyReLU(0.2, True)
            ]
        else:
            model_in += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(input_nc, ndf, kernel_size=4, stride=1, padding=0),
                nn.LeakyReLU(0.2, True),
                Downsample(ndf)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if (no_antialias):
                model_in += [
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=0, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                model_in += [
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=0, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)
                ]
        # print(ndf * nf_mult)
        # 256, 32, 32
        self.model_o1 = nn.Sequential(
            # 32 -> 16
            nn.ReflectionPad2d(1),
            nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=3, padding=0, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            Downsample(ndf * nf_mult),
            # 16 -> 8
            nn.ReflectionPad2d(1),
            nn.Conv2d(ndf * nf_mult, ndf * nf_mult * 2, kernel_size=3, padding=0, bias=use_bias),
            norm_layer(ndf * nf_mult * 2),
            nn.LeakyReLU(0.2, True),
            Downsample(ndf * nf_mult * 2),
            # 8 -> 1
            nn.AdaptiveAvgPool2d(1)
        )
        self.projection = nn.Sequential(
            nn.Linear(ndf * nf_mult * 2, ndf * nf_mult * 2, bias=False),
            nn.ReLU(),
            nn.Linear(ndf * nf_mult * 2, ndf * nf_mult * 2, bias=False),
        )
        from torchvision import models
        model_o7 = []
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        model_o7 += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        model_o7 += [nn.ReflectionPad2d(1),
                     nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=0)]  # output 1 channel prediction map

        self.model_in = nn.Sequential(*model_in)
        self.model_o7 = nn.Sequential(*model_o7)

    def forward(self, input):
        """Standard forward."""
        B = input.shape[0]

        x = self.model_in(input)

        features = self.model_o1(x)
        # features = F.normalize(self.projection(features.view(B, -1)), dim=1)
        features = self.projection(features.view(B, -1))
        logits = torch.einsum('ac,bc->ab', features[B // 2:], features).unsqueeze(0)
        return self.model_o7(x), logits.unsqueeze(1)


class NLayerMSDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerMSDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model_in = []
        if (no_antialias):
            model_in += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0),
                nn.LeakyReLU(0.2, True)
            ]
        else:
            model_in += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(input_nc, ndf, kernel_size=4, stride=1, padding=0),
                nn.LeakyReLU(0.2, True),
                Downsample(ndf)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if (no_antialias):
                model_in += [
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=0, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                model_in += [
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=0, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)
                ]
        # print(ndf * nf_mult)
        self.model_o1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ndf * nf_mult, ndf * nf_mult // 2, kernel_size=3, padding=0, bias=use_bias),
            norm_layer(ndf * nf_mult // 2),
            nn.LeakyReLU(0.2, True),
            Downsample(ndf * nf_mult // 2)
        )
        self.fc = nn.utils.spectral_norm(nn.Linear(ndf * nf_mult, 1, bias=False))

        model_o7 = []
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        model_o7 += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        model_o7 += [nn.ReflectionPad2d(1),
                     nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=0)]  # output 1 channel prediction map

        self.model_in = nn.Sequential(*model_in)
        self.model_o7 = nn.Sequential(*model_o7)

    def forward(self, input):
        """Standard forward."""
        x = self.model_in(input)

        x1 = self.model_o1(x)
        gap = F.adaptive_avg_pool2d(x1, 1)
        gmp = F.adaptive_max_pool2d(x1, 1)
        cam_logit = torch.cat([gap, gmp], 1)  # 1, 512, 1, 1
        # 1, 512, 1, 1 -> 1, 512 -> 1, 1
        # print(cam_logit.shape, self.model_o1)
        cam_logit = self.fc(cam_logit.view(cam_logit.shape[0], -1))

        # return self.model_o7(x), cam_logit
        return torch.cat([self.model_o7(x).flatten(start_dim=1), cam_logit.flatten(start_dim=1)], dim=1)
        # return self.model(input)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if (no_antialias):
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True),
                        Downsample(ndf)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if (no_antialias):
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class NLayerHDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerHDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True),
                    # Downsample(ndf)
                    ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if (no_antialias):
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class NLayerDiscriminator3D(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator3D, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if (no_antialias):
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True),
                        Downsample(ndf)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if (no_antialias):
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, input_nc, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class NiceDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=7):
        super(NiceDiscriminator, self).__init__()
        model = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                     nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]  # 1+3*2^0 =4

        for i in range(1, 2):  # 1+3*2^0 + 3*2^1 =10
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                          nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

            # Class Activation Map
        mult = 2 ** (1)
        self.fc = nn.utils.spectral_norm(nn.Linear(ndf * mult * 2, 1, bias=False))
        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.lamda = nn.Parameter(torch.zeros(1))

        Dis0_0 = []
        for i in range(2, n_layers - 4):  # 1+3*2^0 + 3*2^1 + 3*2^2 =22
            mult = 2 ** (i - 1)
            Dis0_0 += [nn.ReflectionPad2d(1),
                       nn.utils.spectral_norm(
                           nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                       nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 4 - 1)
        Dis0_1 = [nn.ReflectionPad2d(1),  # 1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 = 46
                  nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]
        mult = 2 ** (n_layers - 4)
        self.conv0 = nn.utils.spectral_norm(  # 1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 + 3*2^3= 70
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

        Dis1_0 = []
        for i in range(n_layers - 4,
                       n_layers - 2):  # 1+3*2^0 + 3*2^1 + 3*2^2 + 3*2^3=46, 1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 +3*2^4 = 94
            mult = 2 ** (i - 1)
            Dis1_0 += [nn.ReflectionPad2d(1),
                       nn.utils.spectral_norm(
                           nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                       nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        Dis1_1 = [nn.ReflectionPad2d(1),  # 1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 +3*2^4 + 3*2^5= 94 + 96 = 190
                  nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]
        mult = 2 ** (n_layers - 2)
        self.conv1 = nn.utils.spectral_norm(  # 1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 +3*2^4 + 3*2^5 + 3*2^5 = 286
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

        # self.attn = Self_Attn( ndf * mult)
        self.pad = nn.ReflectionPad2d(1)

        self.model = nn.Sequential(*model)
        self.Dis0_0 = nn.Sequential(*Dis0_0)
        self.Dis0_1 = nn.Sequential(*Dis0_1)
        self.Dis1_0 = nn.Sequential(*Dis1_0)
        self.Dis1_1 = nn.Sequential(*Dis1_1)

    def forward(self, input):
        x = self.model(input)

        x_0 = x

        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        x = torch.cat([x, x], 1)
        cam_logit = torch.cat([gap, gmp], 1)
        cam_logit = self.fc(cam_logit.view(cam_logit.shape[0], -1))
        weight = list(self.fc.parameters())[0]
        x = x * weight.unsqueeze(2).unsqueeze(3)
        x = self.conv1x1(x)

        x = self.lamda * x + x_0
        # print("lamda:",self.lamda)

        x = self.leaky_relu(x)

        heatmap = torch.sum(x, dim=1, keepdim=True)

        z = x

        x0 = self.Dis0_0(x)
        x1 = self.Dis1_0(x0)
        x0 = self.Dis0_1(x0)
        x1 = self.Dis1_1(x1)
        x0 = self.pad(x0)
        x1 = self.pad(x1)
        out0 = self.conv0(x0)
        out1 = self.conv1(x1)

        # return out0, out1, cam_logit, heatmap, z
        return torch.cat([out0.flatten(1), out1.flatten(1), cam_logit.flatten(1)], dim=1), heatmap, z


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class PatchDiscriminator(NLayerDiscriminator):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        super().__init__(input_nc, ndf, 2, norm_layer, no_antialias)

    def forward(self, input):
        B, C, H, W = input.size(0), input.size(1), input.size(2), input.size(3)
        size = 16
        Y = H // size
        X = W // size
        input = input.view(B, C, Y, size, X, size)
        input = input.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * Y * X, C, size, size)
        return super().forward(input)


class GroupedChannelNorm(nn.Module):
    def __init__(self, num_groups):
        super().__init__()
        self.num_groups = num_groups

    def forward(self, x):
        shape = list(x.shape)
        new_shape = [shape[0], self.num_groups, shape[1] // self.num_groups] + shape[2:]
        x = x.view(*new_shape)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x_norm = (x - mean) / (std + 1e-7)
        return x_norm.view(*shape)
