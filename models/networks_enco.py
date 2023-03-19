import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
import einops
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .stylegan_networks import StyleGAN2Discriminator, StyleGAN2Generator, TileStyleGAN2Discriminator

from .blocks.shuffle_blocks import ShuffleBlock, ShuffleBlockX
from .blocks.mb_blocks import MBInvertedConvLayer
from .blocks.hrt_blocks import HRTransBlock, GeneralTransformerBlock
# from .blocks.swinir_blocks import MySwinTransformerBlock as SwinTransformerBlock
from .blocks.Swin import SwinTransformerBlock, SwinTransformerBlockV2, ASPP, SwinAttentionBlock
from .blocks.dcnv2_blocks import DeformableConv2d, DynamicKSConv2d, DynamicKSConv2dV2
from .blocks.involution_blocks import Involution2d
from .blocks.mobilevit_blocks import MV2Block, MobileViTBlock
from .blocks.shuffle_transformer_blocks import Block as ShuffleTFBlock
from .blocks.CSWin import CSWinBlock
from .blocks.convnext_blocks import Block as NextBlock


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
        if opt.warmup_epochs > 0:
            def lambda_rule(epoch):
                if epoch < opt.warmup_epochs:
                    lr_l = (epoch + opt.epoch_count) / opt.warmup_epochs
                else:
                    lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
                return lr_l
        else:
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
                            no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=9, opt=opt)
    elif netG == 'd3gan':
        from models.score_sde.models.ncsnpp_generator_adagn import NCSNpp
        net = NCSNpp(opt)
    elif netG == 'sa':
        net = SAResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=9, opt=opt)
    elif netG == 'lsa':
        net = LeakySAResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                     no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=9, opt=opt)
    elif netG == 'resnet_enc':
        net = ResnetEncoder(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=5,
                            no_antialias=no_antialias, opt=opt)
    elif netG == 'resnet_dec':
        net = ResnetDecoder(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=5,
                            no_antialias=no_antialias, opt=opt)
    elif netG == 'spos':
        net = SPOSGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                            no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=opt.n_blocks, opt=opt)
    elif netG == 'nice':
        from models.blocks.nice_gan_blocks import NiceGenerator
        net = NiceGenerator(input_nc=input_nc, output_nc=output_nc, ngf=ngf, n_blocks=6, img_size=256, light=False)
    elif netG == 'swin':
        net = SwinGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                            no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=9, opt=opt)
    elif netG == 'swin_ps':
        net = SwinPSGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                              no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=9, opt=opt)
    elif netG == 'dcnv2_9blocks':
        net = DCNv2Generator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=9, opt=opt)
    elif netG == 'invo_9blocks':
        net = InvolutionGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                  no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=9, opt=opt)
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
    elif netF == 'mlp_sample_predict':
        net = PredictorF(use_mlp=True, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    elif netF == 'cam_mlp_sample':
        net = PatchCamSampleF(use_mlp=True, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc,
                              opt=opt)
    elif netF == 'mlp_sample_nl':
        net = PatchSampleNonLinearF(use_mlp=True, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids,
                                    nc=opt.netF_nc)
    elif netF == 'cam_mlp_sample_nl':
        net = PatchCamSampleNonLinearF(use_mlp=True, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids,
                                       nc=opt.netF_nc,
                                       opt=opt)
    elif netF == 'cam_mlp_sample_nls':
        net = PatchCamSampleNonLinearSamF(use_mlp=True, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids,
                                          nc=opt.netF_nc,
                                          opt=opt)
    elif netF == 'cam_mlp_sample_s':
        net = PatchCamSampleSamF(use_mlp=True, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids,
                                 nc=opt.netF_nc, opt=opt)
    elif netF == 'cam_sample_s':
        net = PatchCamSampleSamF(use_mlp=False, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids,
                                 nc=opt.netF_nc, opt=opt)
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
    elif netD == 'sa':  # default PatchGAN classifier
        net = NLayerSADiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, no_antialias=no_antialias, )
    elif netD == 'cam':  # default PatchGAN classifier
        net = NLayerCAMDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, no_antialias=no_antialias, )
    elif netD == 'd3gan':
        from models.score_sde.models.discriminator import Discriminator_large
        net = Discriminator_large(nc=opt.num_channels, ngf=opt.ngf, t_emb_dim=opt.t_emb_dim, act=nn.LeakyReLU(0.2))
    elif netD == 'nice':  # default PatchGAN classifier
        from models.blocks.nice_gan_blocks import NiceDiscriminator
        net = NiceDiscriminator(input_nc, ndf=ndf, n_layers=7)
    elif netD == 'myunet':  # default PatchGAN classifier
        net = NLayerUNetDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, no_antialias=no_antialias, )
    elif netD == 'hd':  # default PatchGAN classifier
        net = NLayerHDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, no_antialias=no_antialias, )
    elif netD == 'unet3':  # more options
        net = UNet3PlusDiscriminator(in_channels=input_nc, out_channels=1, feature_scale=4, is_deconv=True,
                                     norm_layer=norm_layer)
    elif netD == 'unet':
        from models.blocks.UNetD import UNetDiscriminator
        net = UNetDiscriminator(D_ch=64, D_wide=True, resolution=256,
                                D_kernel_size=3, D_attn='64',
                                num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
                                D_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8,
                                SN_eps=1e-12, output_dim=1, D_mixed_precision=False, D_fp16=False,
                                D_init='ortho', skip_init=False, D_param='SN', decoder_skip_connection=True)

    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, no_antialias=no_antialias, )
    elif netD == 'swin_unet':  # more options
        net = SwinUnetDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, no_antialias=no_antialias, )
    elif netD == 'basic_spos':  # default PatchGAN classifier
        net = SPOSDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, no_antialias=no_antialias, )
    elif netD == 'n_layers_spos':  # more options
        net = SPOSDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, no_antialias=no_antialias, )
    elif netD == 'n_layers_dks':  # more options
        net = DKSDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, no_antialias=no_antialias, )
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
        elif gan_mode == 'ralsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'nonsaturating']:
            self.loss = None
        elif gan_mode == "hinge":
            self.loss = None
        elif gan_mode == "rahinge":
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
        if self.gan_mode in ['ralsgan', 'rahinge']:
            bs = prediction[0].size(0)
        else:
            bs = prediction.size(0)
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'ralsgan':
            # Relativistic average LSGAN, with no activation in discriminator
            pred, ref = prediction
            target_tensor = self.get_target_tensor(pred, target_is_real)
            loss = self.loss(pred - ref.mean(), target_tensor)

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
        elif self.gan_mode == 'rahinge':
            pred, ref = prediction

            if target_is_real:
                minvalue = torch.nn.ReLU()(1.0 - (pred - ref.mean()))
                loss = torch.mean(minvalue)
            else:
                minvalue = torch.nn.ReLU()(1.0 + (pred - ref.mean()))
                loss = torch.mean(minvalue)
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
        x = x.view(1, -1, self.patch_num, self.nc)
        x = self.model(x)
        x_norm = self.l2norm(x)
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
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


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
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


class PredictorF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PredictorF, self).__init__()
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
            mlp = nn.Sequential(
                *[nn.Linear(input_nc, self.nc), nn.InstanceNorm1d(self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats):
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            x_sample = feat
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            x_sample = self.l2norm(x_sample)

            return_feats.append(x_sample)
        return return_feats


class PatchSampleNonLinearF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleNonLinearF, self).__init__()
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
            mlp = nn.Sequential(
                *[nn.Linear(input_nc, self.nc), nn.InstanceNorm1d(self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
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
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[], opt=None):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchCamSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids
        self.opt = opt

        self.oversample_ratio = 4 if self.opt.oversample_ratio is None else self.opt.oversample_ratio
        self.random_ratio = 0.5 if self.opt.random_ratio is None else self.opt.random_ratio

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
                    for b in range(cams[feat_id].shape[0]):
                        cam_b = cams[feat_id][b]

                        points_sampled = torch.randperm(H * W)[:num_points * self.oversample_ratio]

                        values, indices = torch.sort(cam_b.flatten()[points_sampled], descending=False)

                        random_num_points = int(num_points * self.random_ratio)
                        good_num_points = num_points - random_num_points

                        good_idx = points_sampled[indices[:good_num_points]]

                        if random_num_points > 0:
                            points_random = torch.randperm(H * W)[-random_num_points:]
                            patch_id = torch.cat([good_idx, points_random], dim=0)
                        else:
                            patch_id = good_idx

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


class PatchCamSampleNonLinearF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[], opt=None):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchCamSampleNonLinearF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids
        self.opt = opt

        if self.opt.prj_norm == 'BN':
            self.norm = nn.BatchNorm1d
        elif self.opt.prj_norm == 'LN':
            self.norm = nn.LayerNorm
        elif self.opt.prj_norm == 'None':
            self.norm = nn.Identity
        else:
            raise NotImplementedError

        self.oversample_ratio = 4 if self.opt.oversample_ratio is None else self.opt.oversample_ratio
        self.random_ratio = 0.5 if self.opt.random_ratio is None else self.opt.random_ratio

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            hidden_dim = self.nc
            mlp = nn.Sequential(
                *[nn.Linear(input_nc, hidden_dim, bias=False), self.norm(hidden_dim), nn.ReLU(inplace=True),
                  # nn.Linear(hidden_dim, hidden_dim, bias=False), self.norm(hidden_dim), nn.ReLU(inplace=True),
                  nn.Linear(hidden_dim, hidden_dim, bias=False),
                  self.norm(hidden_dim, affine=False) if self.opt.prj_norm == 'BN' else self.norm(hidden_dim)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def create_prj(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = self.nc
            hidden_dim = self.nc // 2
            mlp = nn.Sequential(
                *[nn.Linear(input_nc, hidden_dim, bias=False), self.norm(hidden_dim), nn.ReLU(inplace=True),
                  nn.Linear(hidden_dim, input_nc)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'prj_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None, cams=None, prj=False):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
            self.create_prj(feats)
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
                    for b in range(cams[feat_id].shape[0]):
                        cam_b = cams[feat_id][b]

                        points_sampled = torch.randperm(H * W)[:num_points * self.oversample_ratio]

                        values, indices = torch.sort(cam_b.flatten()[points_sampled], descending=False)

                        random_num_points = int(num_points * self.random_ratio)
                        good_num_points = num_points - random_num_points

                        good_idx = points_sampled[indices[:good_num_points]]

                        if random_num_points > 0:
                            # points_random = torch.randperm(H * W)[-random_num_points:]
                            points_random = points_sampled[indices[-random_num_points:]]
                            patch_id = torch.cat([good_idx, points_random], dim=0)
                        else:
                            patch_id = good_idx

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
            if prj:
                prj = getattr(self, 'prj_%d' % feat_id)
                x_sample = prj(x_sample)
            return_ids.append(id_b_list)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


class PatchCamSampleNonLinearSamF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[], opt=None):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchCamSampleNonLinearSamF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids
        self.opt = opt

        if self.opt.prj_norm == 'BN':
            self.norm = nn.BatchNorm1d
        elif self.opt.prj_norm == 'LN':
            self.norm = nn.LayerNorm
        elif self.opt.prj_norm == 'None':
            self.norm = nn.Identity
        else:
            raise NotImplementedError

        self.oversample_ratio = 4 if self.opt.oversample_ratio is None else self.opt.oversample_ratio
        self.random_ratio = 0.5 if self.opt.random_ratio is None else self.opt.random_ratio

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            hidden_dim = self.nc
            mlp = nn.Sequential(
                *[nn.Linear(input_nc, hidden_dim, bias=False), self.norm(hidden_dim), nn.ReLU(inplace=True),
                  # nn.Linear(hidden_dim, hidden_dim, bias=False), self.norm(hidden_dim), nn.ReLU(inplace=True),
                  nn.Linear(hidden_dim, hidden_dim, bias=False),
                  self.norm(hidden_dim, affine=False) if self.opt.prj_norm == 'BN' else self.norm(hidden_dim)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def create_prj(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = self.nc
            hidden_dim = input_nc // 2
            mlp = nn.Sequential(
                *[nn.Linear(input_nc, hidden_dim, bias=False), self.norm(hidden_dim), nn.ReLU(inplace=True),
                  nn.Linear(hidden_dim, input_nc)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'prj_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None, cams=None, prj=False):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
            self.create_prj(feats)
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
                    for b in range(cams[feat_id].shape[0]):
                        cam_b = cams[feat_id][b]

                        points_sampled = torch.randperm(H * W)[:num_points * self.oversample_ratio]

                        values, indices = torch.sort(cam_b.flatten()[points_sampled], descending=False)

                        random_num_points = int(num_points * self.random_ratio)
                        good_num_points = num_points - random_num_points

                        good_idx = points_sampled[indices[:good_num_points]]

                        if random_num_points > 0:
                            points_random = torch.randperm(H * W)[-random_num_points:]
                            # points_random = points_sampled[indices[-random_num_points:]]
                            patch_id = torch.cat([good_idx, points_random], dim=0)
                        else:
                            patch_id = good_idx

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
            if prj:
                prj = getattr(self, 'prj_%d' % feat_id)
                x_sample = prj(x_sample)
            return_ids.append(id_b_list)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


class PatchCamSampleSamF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[], opt=None):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchCamSampleSamF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids
        self.opt = opt

        self.oversample_ratio = 4 if self.opt.oversample_ratio is None else self.opt.oversample_ratio
        self.random_ratio = 0.5 if self.opt.random_ratio is None else self.opt.random_ratio

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(
                *[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
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
                    for b in range(cams[feat_id].shape[0]):
                        cam_b = cams[feat_id][b]

                        points_sampled = torch.randperm(H * W)[:num_points * self.oversample_ratio]

                        values, indices = torch.sort(cam_b.flatten()[points_sampled], descending=False)

                        random_num_points = int(num_points * self.random_ratio)
                        good_num_points = num_points - random_num_points

                        good_idx = points_sampled[indices[:good_num_points]]

                        if random_num_points > 0:
                            # points_random = torch.randperm(H * W)[-random_num_points:]
                            points_random = points_sampled[indices[-random_num_points:]]
                            patch_id = torch.cat([good_idx, points_random], dim=0)
                        else:
                            patch_id = good_idx

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


class SPOSGenerator(nn.Module):
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
        super(SPOSGenerator, self).__init__()
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

        self.n_blocks = n_blocks

        window_size = [8 for _ in range(9)]
        num_heads = [4, 4, 8, 8, 16, 8, 8, 4, 4]
        for i in range(self.n_blocks):  # add ResNet blocks
            model += [nn.ModuleList([
                # ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                #             use_bias=use_bias),
                # MBInvertedConvLayer(in_channels=ngf * mult, out_channels=ngf * mult, kernel_size=3, stride=1,
                #                     expand_ratio=6, mid_channels=None),

                HRTransBlock(num_branches=1, blocks=GeneralTransformerBlock, num_blocks=[2],
                             num_inchannels=[ngf * mult], num_channels=[ngf * mult],
                             num_heads=[num_heads[i]], num_window_sizes=[window_size[i]], num_mlp_ratios=[4],
                             multi_scale_output=False, drop_path=[0., 0.], skip_connec=False),
                # ShuffleBlock(in_channels=ngf * mult // 2, out_channels=ngf * mult, kernel=5, stride=1),
                # ShuffleTFBlock(dim=256, out_dim=256, num_heads=4, window_size=4, shuffle=True, mlp_ratio=4.,
                #                qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                #                drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.InstanceNorm2d, stride=False,
                #                relative_pos_embedding=True),
                nn.Sequential(
                    PreNorm(num_patches=64 * 64, dim=ngf * mult, NORM=i == 0, PE=i == 0),
                    SwinTransformerBlock(dim=ngf * mult, input_resolution=(64, 64),
                                         num_heads=num_heads[i], window_size=window_size[i],
                                         shift_size=0 if (i % 2 == 0) else window_size[i] // 2,
                                         mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                         drop=0., attn_drop=0., drop_path=0.),
                    Rearrange('B (H W) C -> B C H W', H=64)
                )
                # Involution2d(in_channels=ngf * mult, out_channels=ngf * mult, kernel_size=7, padding=3, sc=True),
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


class PreNorm(nn.Module):
    def __init__(self, num_patches=64 * 64, dim=256, NORM=False, PE=True):
        super(PreNorm, self).__init__()
        self.PE = PE

        self.rerange = Rearrange('B C H W -> B (H W) C')
        if NORM:
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = nn.Identity()

        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        trunc_normal_(self.absolute_pos_embed, std=.02)

    def forward(self, x):
        x = self.norm(self.rerange(x))
        if self.PE:
            return x + self.absolute_pos_embed
        else:
            return x


class SwinGenerator(nn.Module):
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
        super(SwinGenerator, self).__init__()
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

        self.n_blocks = n_blocks

        window_size = [4, 4, 8, 8, 16, 8, 8, 4, 4]
        window_size = [8 for _ in range(9)]
        num_heads = [4, 4, 8, 8, 16, 8, 8, 4, 4]

        for i in range(self.n_blocks):  # add ResNet blocks
            model += [
                nn.Sequential(
                    PreNorm(num_patches=64 * 64, dim=ngf * mult, NORM=i == 0, PE=i == 0),
                    SwinTransformerBlock(dim=ngf * mult, input_resolution=(64, 64),
                                         num_heads=num_heads[i], window_size=window_size[i],
                                         shift_size=0 if (i % 2 == 0) else window_size[i] // 2,
                                         mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                         drop=0., attn_drop=0., drop_path=0.),
                    Rearrange('B (H W) C -> B C H W', H=64)
                )
            ]

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

    def forward(self, input, layers=[], encode_only=False, return_layer_id=None, CUTv2=False):
        if CUTv2:
            feat = input
            feats = []
            if 0 in layers:
                feats.append(feat)
                layers = layers[1:]  # remove 0 from list
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)

            return feat, feats  # return both output and intermediate features

        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            if 0 in layers:
                feats.append(feat)
                layers = layers[1:]  # remove 0 from list
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


class PixelUnshuffle(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PixelShuffle(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(dim // 4, dim // 2, bias=False)
        self.norm = norm_layer(dim // 4)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape

        # x = x.view(B, H, W, C)

        x = einops.rearrange(x, 'B (H W) C -> B C H W', H=H)
        x = nn.PixelShuffle(upscale_factor=2)(x)

        x = einops.rearrange(x, 'B C H W -> B (H W) C')

        x = self.norm(x)
        x = self.reduction(x)

        return x


class SwinPSGenerator(nn.Module):
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
        super(SwinPSGenerator, self).__init__()
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
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True),
                      Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling

        self.n_blocks = n_blocks

        window_size = [4, 4, 8, 8, 16, 8, 8, 4, 4]
        window_size = [8 for _ in range(9)]
        num_heads = [4, 4, 8, 8, 16, 8, 8, 4, 4]

        for i in range(self.n_blocks):  # add ResNet blocks
            model += [
                nn.Sequential(
                    PreNorm(num_patches=64 * 64, dim=ngf * mult, NORM=i == 0, PE=i == 0),
                    # SwinTransformerBlockV2(dim=ngf * mult, input_resolution=(64, 64),
                    #                        num_heads=num_heads[i], window_size=window_size[i],
                    #                        shift_size=0 if (i % 2 == 0) else window_size[i] // 2,
                    #                        mlp_ratio=4, qkv_bias=True, qk_scale=None,
                    #                        drop=0., attn_drop=0., drop_path=0.),
                    ASPP(dim=ngf * mult),
                    Rearrange('B (H W) C -> B C H W', H=64)
                )
            ]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            # model += [nn.PixelShuffle(upscale_factor=2),
            #           ShuffleBlock(in_channels=ngf * mult // 8, out_channels=ngf * mult // 4, kernel=5, stride=1),
            #           Rearrange('B C H W -> B (H W) C'),
            #           nn.LayerNorm(ngf * mult // 4),
            #           nn.Linear(ngf * mult // 4, ngf * mult // 2),
            #           Rearrange('B (H W) C -> B C H W', H=128 * (i + 1)),
            #           # nn.Conv2d(ngf * mult // 4, ngf * mult // 2, 1, 1, 0),
            #           ]
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

    def forward(self, input, layers=[], encode_only=False, return_layer_id=None, CUTv2=False):
        if CUTv2:
            feat = input
            feats = []
            if 0 in layers:
                feats.append(feat)
                layers = layers[1:]  # remove 0 from list
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)

            return feat, feats  # return both output and intermediate features

        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            if 0 in layers:
                feats.append(feat)
                layers = layers[1:]  # remove 0 from list
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


class SPOSDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(SPOSDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.n_layers = n_layers

        kw = [3, 3, 4, 4]  # kernel size
        dl = [1, 2, 1, 2]  # dilation
        pd = [1, 2, 1, 2]  # padding
        sd = [1, 1, 1, 1]  # stride

        # receptive field: 1 + (4-1) * 2^0 = 4

        if (no_antialias):
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [nn.ModuleList([
                nn.Sequential(nn.Conv2d(input_nc, ndf, kernel_size=kw[0], stride=sd[0], dilation=dl[0], padding=pd[0]),
                              nn.LeakyReLU(0.2, True), Downsample(ndf)),
                nn.Sequential(nn.Conv2d(input_nc, ndf, kernel_size=kw[1], stride=sd[1], dilation=dl[1], padding=pd[1]),
                              nn.LeakyReLU(0.2, True), Downsample(ndf)),
                nn.Sequential(nn.Conv2d(input_nc, ndf, kernel_size=kw[2], stride=sd[2], dilation=dl[2], padding=pd[2]),
                              nn.LeakyReLU(0.2, True), Downsample(ndf)),
                nn.Sequential(nn.Conv2d(input_nc, ndf, kernel_size=kw[3], stride=sd[3], dilation=dl[3], padding=pd[3]),
                              nn.LeakyReLU(0.2, True), Downsample(ndf)),
            ])]
        nf_mult = 1
        nf_mult_prev = 1
        # 4 + (4-1) * 2^1 = 10
        # 10 + (4-1) * 2^2 = 22
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if (no_antialias):
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw[0], stride=sd[0], dilation=dl[0],
                                  padding=pd[0], bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True),
                        Downsample(ndf * nf_mult)),
                    nn.Sequential(
                        nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw[1], stride=sd[1], dilation=dl[1],
                                  padding=pd[1], bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True),
                        Downsample(ndf * nf_mult)),
                    nn.Sequential(
                        nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw[2], stride=sd[2], dilation=dl[2],
                                  padding=pd[2], bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True),
                        Downsample(ndf * nf_mult)),
                    nn.Sequential(
                        nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw[3], stride=sd[3], dilation=dl[3],
                                  padding=pd[3], bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True),
                        Downsample(ndf * nf_mult)),
                ])]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        # 22 + (4-1) * 2^3 = 46
        sequence += [nn.ModuleList([
            nn.Sequential(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw[0], stride=sd[0], dilation=dl[0],
                                    padding=pd[0], bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)),
            nn.Sequential(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw[1], stride=sd[1], dilation=dl[1],
                                    padding=pd[1], bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)),
            nn.Sequential(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw[2], stride=sd[2], dilation=dl[2],
                                    padding=pd[2], bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)),
            nn.Sequential(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw[3], stride=sd[3], dilation=dl[3],
                                    padding=pd[3], bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)),
        ])]
        # 46 + (4-1) * 2^3 = 70
        # # output 1 channel prediction map
        sequence += [nn.ModuleList([
            nn.Sequential(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw[0], stride=sd[0], dilation=dl[0], padding=pd[0])),
            nn.Sequential(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw[1], stride=sd[1], dilation=dl[1], padding=pd[1])),
            nn.Sequential(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw[2], stride=sd[2], dilation=dl[2], padding=pd[2])),
            nn.Sequential(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw[3], stride=sd[3], dilation=dl[3], padding=pd[3])),
        ])]
        self.model = nn.ModuleList(sequence)

    def forward(self, x, choice):
        """Standard forward."""
        for i, j in enumerate(choice):
            x = self.model[i][j](x)
        return x


class DKSDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(DKSDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = [1, 3, 5, 7]
        padw = 0
        if (no_antialias):
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [DynamicKSConv2d(input_nc, ndf, kernel_size_list=kw, stride=1, padding=padw),
                        nn.LeakyReLU(0.2, True),
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
                    DynamicKSConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size_list=kw, stride=1, padding=padw,
                                    bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            DynamicKSConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size_list=kw, stride=1, padding=padw,
                            bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            DynamicKSConv2d(ndf * nf_mult, 1, kernel_size_list=kw, stride=1,
                            padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, x, choice):
        """Standard forward."""
        idx = 0
        for m in self.model:
            if isinstance(m, DynamicKSConv2d):
                x = m(x, idx=choice[idx])
            else:
                x = m(x)
        return x


class DKSv2Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(DKSv2Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = [1, 3, 5, 7]
        padw = 0
        if (no_antialias):
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [DynamicKSConv2dV2(input_nc, ndf, kernel_size_list=kw, stride=1, padding=padw),
                        nn.LeakyReLU(0.2, True),
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
                    DynamicKSConv2dV2(ndf * nf_mult_prev, ndf * nf_mult, kernel_size_list=kw, stride=1, padding=padw,
                                      bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            DynamicKSConv2dV2(ndf * nf_mult_prev, ndf * nf_mult, kernel_size_list=kw, stride=1, padding=padw,
                              bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            DynamicKSConv2dV2(ndf * nf_mult, 1, kernel_size_list=kw, stride=1,
                              padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, x, choice):
        """Standard forward."""
        idx = 0
        for m in self.model:
            if isinstance(m, DynamicKSConv2dV2):
                x = m(x, idx=choice[idx])
            else:
                x = m(x)
        return x


class DKSv3Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(DKSv3Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = [1, 3, 5, 7]
        padw = 0
        if (no_antialias):
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [DynamicKSConv2dV2(input_nc, ndf, kernel_size_list=kw, stride=1, padding=padw),
                        nn.LeakyReLU(0.2, True),
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
                    DynamicKSConv2dV2(ndf * nf_mult_prev, ndf * nf_mult, kernel_size_list=kw, stride=1, padding=padw,
                                      bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            DynamicKSConv2dV2(ndf * nf_mult_prev, ndf * nf_mult, kernel_size_list=kw, stride=1, padding=padw,
                              bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            DynamicKSConv2dV2(ndf * nf_mult, 1, kernel_size_list=kw, stride=1,
                              padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, x, choice):
        """Standard forward."""
        idx = 0
        for m in self.model:
            if isinstance(m, DynamicKSConv2dV2):
                x = m(x, idx=choice[idx])
            else:
                x = m(x)
        return x


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

    def forward(self, input, layers=[], encode_only=False, return_layer_id=None, CUTv2=False):
        if CUTv2:
            feat = input
            feats = []
            if 0 in layers:
                feats.append(feat)
                layers = layers[1:]  # remove 0 from list
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)

            return feat, feats  # return both output and intermediate features

        if return_layer_id is not None:
            feat = input
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id == return_layer_id:
                    return feat

        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            if 0 in layers:
                feats.append(feat)
                layers = layers[1:]  # remove 0 from list
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
            # print('input:', input.shape)
            # for idx, m in enumerate(self.model):
            #     input = m(input)
            #     print(idx, m._get_name(), input.shape)
            # input: torch.Size([1, 1, 256, 256])
            # 0 ReflectionPad2d torch.Size([1, 1, 262, 262])
            # 1 Conv2d torch.Size([1, 64, 256, 256])
            # 2 InstanceNorm2d torch.Size([1, 64, 256, 256])
            # 3 ReLU torch.Size([1, 64, 256, 256])
            # 4 Conv2d torch.Size([1, 128, 256, 256])
            # 5 InstanceNorm2d torch.Size([1, 128, 256, 256])
            # 6 ReLU torch.Size([1, 128, 256, 256])
            # 7 Downsample torch.Size([1, 128, 128, 128])
            # 8 Conv2d torch.Size([1, 256, 128, 128])
            # 9 InstanceNorm2d torch.Size([1, 256, 128, 128])
            # 10 ReLU torch.Size([1, 256, 128, 128])
            # 11 Downsample torch.Size([1, 256, 64, 64])
            # 12 ResnetBlock torch.Size([1, 256, 64, 64])
            # 13 ResnetBlock torch.Size([1, 256, 64, 64])
            # 14 ResnetBlock torch.Size([1, 256, 64, 64])
            # 15 ResnetBlock torch.Size([1, 256, 64, 64])
            # 16 ResnetBlock torch.Size([1, 256, 64, 64])
            # 17 ResnetBlock torch.Size([1, 256, 64, 64])
            # 18 ResnetBlock torch.Size([1, 256, 64, 64])
            # 19 ResnetBlock torch.Size([1, 256, 64, 64])
            # 20 ResnetBlock torch.Size([1, 256, 64, 64])
            # 21 Upsample torch.Size([1, 256, 128, 128])
            # 22 Conv2d torch.Size([1, 128, 128, 128])
            # 23 InstanceNorm2d torch.Size([1, 128, 128, 128])
            # 24 ReLU torch.Size([1, 128, 128, 128])
            # 25 Upsample torch.Size([1, 128, 256, 256])
            # 26 Conv2d torch.Size([1, 64, 256, 256])
            # 27 InstanceNorm2d torch.Size([1, 64, 256, 256])
            # 28 ReLU torch.Size([1, 64, 256, 256])
            # 29 ReflectionPad2d torch.Size([1, 64, 262, 262])
            # 30 Conv2d torch.Size([1, 1, 256, 256])
            # 31 Tanh torch.Size([1, 1, 256, 256])
            fake = self.model(input)
            return fake

    def forward_D(self, input, return_layer=12):
        feat = input
        for layer_id, layer in enumerate(self.model):
            feat = layer(feat)
            if layer_id == return_layer:
                return feat


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

    def forward(self, input, layers=[], encode_only=False, return_layer_id=None, CUTv2=False):
        if CUTv2:
            feat = input
            feats = []
            if 0 in layers:
                feats.append(feat)
                layers = layers[1:]  # remove 0 from list
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)

            return feat, feats  # return both output and intermediate features

        if return_layer_id is not None:
            feat = input
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id == return_layer_id:
                    return feat

        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            if 0 in layers:
                feats.append(feat)
                layers = layers[1:]  # remove 0 from list
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
            # print('input:', input.shape)
            # for idx, m in enumerate(self.model):
            #     input = m(input)
            #     print(idx, m._get_name(), input.shape)
            # input: torch.Size([1, 1, 256, 256])
            # 0 ReflectionPad2d torch.Size([1, 1, 262, 262])
            # 1 Conv2d torch.Size([1, 64, 256, 256])
            # 2 InstanceNorm2d torch.Size([1, 64, 256, 256])
            # 3 ReLU torch.Size([1, 64, 256, 256])
            # 4 Conv2d torch.Size([1, 128, 256, 256])
            # 5 InstanceNorm2d torch.Size([1, 128, 256, 256])
            # 6 ReLU torch.Size([1, 128, 256, 256])
            # 7 Downsample torch.Size([1, 128, 128, 128])
            # 8 Conv2d torch.Size([1, 256, 128, 128])
            # 9 InstanceNorm2d torch.Size([1, 256, 128, 128])
            # 10 ReLU torch.Size([1, 256, 128, 128])
            # 11 Downsample torch.Size([1, 256, 64, 64])
            # 12 ResnetBlock torch.Size([1, 256, 64, 64])
            # 13 ResnetBlock torch.Size([1, 256, 64, 64])
            # 14 ResnetBlock torch.Size([1, 256, 64, 64])
            # 15 ResnetBlock torch.Size([1, 256, 64, 64])
            # 16 ResnetBlock torch.Size([1, 256, 64, 64])
            # 17 ResnetBlock torch.Size([1, 256, 64, 64])
            # 18 ResnetBlock torch.Size([1, 256, 64, 64])
            # 19 ResnetBlock torch.Size([1, 256, 64, 64])
            # 20 ResnetBlock torch.Size([1, 256, 64, 64])
            # 21 Upsample torch.Size([1, 256, 128, 128])
            # 22 Conv2d torch.Size([1, 128, 128, 128])
            # 23 InstanceNorm2d torch.Size([1, 128, 128, 128])
            # 24 ReLU torch.Size([1, 128, 128, 128])
            # 25 Upsample torch.Size([1, 128, 256, 256])
            # 26 Conv2d torch.Size([1, 64, 256, 256])
            # 27 InstanceNorm2d torch.Size([1, 64, 256, 256])
            # 28 ReLU torch.Size([1, 64, 256, 256])
            # 29 ReflectionPad2d torch.Size([1, 64, 262, 262])
            # 30 Conv2d torch.Size([1, 1, 256, 256])
            # 31 Tanh torch.Size([1, 1, 256, 256])
            fake = self.model(input)
            return fake

    def forward_D(self, input, return_layer=12):
        feat = input
        for layer_id, layer in enumerate(self.model):
            feat = layer(feat)
            if layer_id == return_layer:
                return feat


class SAResnetGenerator(nn.Module):
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
        super(SAResnetGenerator, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.utils.spectral_norm(nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias)),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if (no_antialias):
                model += [nn.utils.spectral_norm(
                    nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias)),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True)]
            else:
                model += [nn.utils.spectral_norm(
                    nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias)),
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
                model += [nn.utils.spectral_norm(nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                                                    kernel_size=3, stride=2,
                                                                    padding=1, output_padding=1,
                                                                    bias=use_bias)),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.utils.spectral_norm(nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                                           kernel_size=3, stride=1,
                                                           padding=1,  # output_padding=1,
                                                           bias=use_bias)),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.utils.spectral_norm(nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0))]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, layers=[], encode_only=False, return_layer_id=None, CUTv2=False):
        if CUTv2:
            feat = input
            feats = []
            if 0 in layers:
                feats.append(feat)
                layers = layers[1:]  # remove 0 from list
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)

            return feat, feats  # return both output and intermediate features

        if return_layer_id is not None:
            feat = input
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id == return_layer_id:
                    return feat

        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            if 0 in layers:
                feats.append(feat)
                layers = layers[1:]  # remove 0 from list
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
            # print('input:', input.shape)
            # for idx, m in enumerate(self.model):
            #     input = m(input)
            #     print(idx, m._get_name(), input.shape)
            # input: torch.Size([1, 1, 256, 256])
            # 0 ReflectionPad2d torch.Size([1, 1, 262, 262])
            # 1 Conv2d torch.Size([1, 64, 256, 256])
            # 2 InstanceNorm2d torch.Size([1, 64, 256, 256])
            # 3 ReLU torch.Size([1, 64, 256, 256])
            # 4 Conv2d torch.Size([1, 128, 256, 256])
            # 5 InstanceNorm2d torch.Size([1, 128, 256, 256])
            # 6 ReLU torch.Size([1, 128, 256, 256])
            # 7 Downsample torch.Size([1, 128, 128, 128])
            # 8 Conv2d torch.Size([1, 256, 128, 128])
            # 9 InstanceNorm2d torch.Size([1, 256, 128, 128])
            # 10 ReLU torch.Size([1, 256, 128, 128])
            # 11 Downsample torch.Size([1, 256, 64, 64])
            # 12 ResnetBlock torch.Size([1, 256, 64, 64])
            # 13 ResnetBlock torch.Size([1, 256, 64, 64])
            # 14 ResnetBlock torch.Size([1, 256, 64, 64])
            # 15 ResnetBlock torch.Size([1, 256, 64, 64])
            # 16 ResnetBlock torch.Size([1, 256, 64, 64])
            # 17 ResnetBlock torch.Size([1, 256, 64, 64])
            # 18 ResnetBlock torch.Size([1, 256, 64, 64])
            # 19 ResnetBlock torch.Size([1, 256, 64, 64])
            # 20 ResnetBlock torch.Size([1, 256, 64, 64])
            # 21 Upsample torch.Size([1, 256, 128, 128])
            # 22 Conv2d torch.Size([1, 128, 128, 128])
            # 23 InstanceNorm2d torch.Size([1, 128, 128, 128])
            # 24 ReLU torch.Size([1, 128, 128, 128])
            # 25 Upsample torch.Size([1, 128, 256, 256])
            # 26 Conv2d torch.Size([1, 64, 256, 256])
            # 27 InstanceNorm2d torch.Size([1, 64, 256, 256])
            # 28 ReLU torch.Size([1, 64, 256, 256])
            # 29 ReflectionPad2d torch.Size([1, 64, 262, 262])
            # 30 Conv2d torch.Size([1, 1, 256, 256])
            # 31 Tanh torch.Size([1, 1, 256, 256])
            fake = self.model(input)
            return fake

    def forward_D(self, input, return_layer=12):
        feat = input
        for layer_id, layer in enumerate(self.model):
            feat = layer(feat)
            if layer_id == return_layer:
                return feat


class LeakySAResnetGenerator(nn.Module):
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
        super(SAResnetGenerator, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.utils.spectral_norm(nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias)),
                 norm_layer(ngf),
                 nn.LeakyReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if (no_antialias):
                model += [nn.utils.spectral_norm(
                    nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias)),
                    norm_layer(ngf * mult * 2),
                    nn.LeakyReLU(True)]
            else:
                model += [nn.utils.spectral_norm(
                    nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias)),
                    norm_layer(ngf * mult * 2),
                    nn.LeakyReLU(True),
                    Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [LeakySAResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                         use_dropout=use_dropout,
                                         use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [nn.utils.spectral_norm(nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                                                    kernel_size=3, stride=2,
                                                                    padding=1, output_padding=1,
                                                                    bias=use_bias)),
                          norm_layer(int(ngf * mult / 2)),
                          nn.LeakyReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.utils.spectral_norm(nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                                           kernel_size=3, stride=1,
                                                           padding=1,  # output_padding=1,
                                                           bias=use_bias)),
                          norm_layer(int(ngf * mult / 2)),
                          nn.LeakyReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.utils.spectral_norm(nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0))]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, layers=[], encode_only=False, return_layer_id=None, CUTv2=False):
        if CUTv2:
            feat = input
            feats = []
            if 0 in layers:
                feats.append(feat)
                layers = layers[1:]  # remove 0 from list
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)

            return feat, feats  # return both output and intermediate features

        if return_layer_id is not None:
            feat = input
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id == return_layer_id:
                    return feat

        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            if 0 in layers:
                feats.append(feat)
                layers = layers[1:]  # remove 0 from list
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
            # print('input:', input.shape)
            # for idx, m in enumerate(self.model):
            #     input = m(input)
            #     print(idx, m._get_name(), input.shape)
            # input: torch.Size([1, 1, 256, 256])
            # 0 ReflectionPad2d torch.Size([1, 1, 262, 262])
            # 1 Conv2d torch.Size([1, 64, 256, 256])
            # 2 InstanceNorm2d torch.Size([1, 64, 256, 256])
            # 3 ReLU torch.Size([1, 64, 256, 256])
            # 4 Conv2d torch.Size([1, 128, 256, 256])
            # 5 InstanceNorm2d torch.Size([1, 128, 256, 256])
            # 6 ReLU torch.Size([1, 128, 256, 256])
            # 7 Downsample torch.Size([1, 128, 128, 128])
            # 8 Conv2d torch.Size([1, 256, 128, 128])
            # 9 InstanceNorm2d torch.Size([1, 256, 128, 128])
            # 10 ReLU torch.Size([1, 256, 128, 128])
            # 11 Downsample torch.Size([1, 256, 64, 64])
            # 12 ResnetBlock torch.Size([1, 256, 64, 64])
            # 13 ResnetBlock torch.Size([1, 256, 64, 64])
            # 14 ResnetBlock torch.Size([1, 256, 64, 64])
            # 15 ResnetBlock torch.Size([1, 256, 64, 64])
            # 16 ResnetBlock torch.Size([1, 256, 64, 64])
            # 17 ResnetBlock torch.Size([1, 256, 64, 64])
            # 18 ResnetBlock torch.Size([1, 256, 64, 64])
            # 19 ResnetBlock torch.Size([1, 256, 64, 64])
            # 20 ResnetBlock torch.Size([1, 256, 64, 64])
            # 21 Upsample torch.Size([1, 256, 128, 128])
            # 22 Conv2d torch.Size([1, 128, 128, 128])
            # 23 InstanceNorm2d torch.Size([1, 128, 128, 128])
            # 24 ReLU torch.Size([1, 128, 128, 128])
            # 25 Upsample torch.Size([1, 128, 256, 256])
            # 26 Conv2d torch.Size([1, 64, 256, 256])
            # 27 InstanceNorm2d torch.Size([1, 64, 256, 256])
            # 28 ReLU torch.Size([1, 64, 256, 256])
            # 29 ReflectionPad2d torch.Size([1, 64, 262, 262])
            # 30 Conv2d torch.Size([1, 1, 256, 256])
            # 31 Tanh torch.Size([1, 1, 256, 256])
            fake = self.model(input)
            return fake

    def forward_D(self, input, return_layer=12):
        feat = input
        for layer_id, layer in enumerate(self.model):
            feat = layer(feat)
            if layer_id == return_layer:
                return feat


class DCNv2Generator(nn.Module):
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
        super(DCNv2Generator, self).__init__()
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

            model += [DCNv2Block(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
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


class InvolutionGenerator(nn.Module):
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
        super(InvolutionGenerator, self).__init__()
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

            model += [Involution2d(in_channels=ngf * mult, out_channels=ngf * mult, kernel_size=7, padding=3, sc=True)]

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


class ResnetDecoder(nn.Module):
    """Resnet-based decoder that consists of a few Resnet blocks + a few upsampling operations.
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', no_antialias=False, opt=None):
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

        n_downsampling = 2
        mult = 2 ** n_downsampling

        model = [nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(int(ngf * mult / 2), ngf * mult, kernel_size=3, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(ngf * mult),
            nn.ReLU(True)
        )]

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
                 padding_type='reflect', no_antialias=False, opt=None):
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


class SAResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(SAResnetBlock, self).__init__()
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

        conv_block += [nn.utils.spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)),
                       norm_layer(dim), nn.ReLU(True)]
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
        conv_block += [nn.utils.spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class LeakySAResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(LeakySAResnetBlock, self).__init__()
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

        conv_block += [nn.utils.spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)),
                       norm_layer(dim), nn.LeakyReLU(True)]
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
        conv_block += [nn.utils.spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class DCNv2Block(nn.Module):
    """Define a Resnet DCNv2 block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(DCNv2Block, self).__init__()
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

        conv_block += [DeformableConv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim),
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
        conv_block += [DeformableConv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

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
        # for m in self.model:
        #     input = m(input)
        #     print(m._get_name(), input.shape)
        # Conv2d
        # torch.Size([2, 64, 255, 255])
        # LeakyReLU
        # torch.Size([2, 64, 255, 255])
        # Downsample
        # torch.Size([2, 64, 128, 128])
        # Conv2d
        # torch.Size([2, 128, 127, 127])
        # InstanceNorm2d
        # torch.Size([2, 128, 127, 127])
        # LeakyReLU
        # torch.Size([2, 128, 127, 127])
        # Downsample
        # torch.Size([2, 128, 64, 64])
        # Conv2d
        # torch.Size([2, 256, 63, 63])
        # InstanceNorm2d
        # torch.Size([2, 256, 63, 63])
        # LeakyReLU
        # torch.Size([2, 256, 63, 63])
        # Downsample
        # torch.Size([2, 256, 32, 32])
        # Conv2d
        # torch.Size([2, 512, 31, 31])
        # InstanceNorm2d
        # torch.Size([2, 512, 31, 31])
        # LeakyReLU
        # torch.Size([2, 512, 31, 31])
        # Conv2d
        # torch.Size([2, 1, 30, 30])
        return self.model(input)


class NLayerCAMDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerCAMDiscriminator, self).__init__()
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

        self.model[7].register_backward_hook(self.__forward_hook)
        self.model[7].register_backward_hook(self.__backward_hook)
        self.model[11].register_backward_hook(self.__forward_hook)
        self.model[11].register_backward_hook(self.__backward_hook)
        self.model[14].register_backward_hook(self.__forward_hook)
        self.model[14].register_backward_hook(self.__backward_hook)

    def forward(self, input):
        self.grads = []
        self.fmaps = []

        """Standard forward."""
        # for idx, m in enumerate(self.model):
        #     input = m(input)
        #     print(idx, m._get_name(), input.shape)
        # 0 Conv2d torch.Size([1, 64, 255, 255])
        # 1 LeakyReLU torch.Size([1, 64, 255, 255])
        # 2 Downsample torch.Size([1, 64, 128, 128])
        # 3 Conv2d torch.Size([1, 128, 127, 127])
        # 4 InstanceNorm2d torch.Size([1, 128, 127, 127])
        # 5 LeakyReLU torch.Size([1, 128, 127, 127])
        # 6 Downsample torch.Size([1, 128, 64, 64])
        # 7 Conv2d torch.Size([1, 256, 63, 63])
        # 8 InstanceNorm2d torch.Size([1, 256, 63, 63])
        # 9 LeakyReLU torch.Size([1, 256, 63, 63])
        # 10 Downsample torch.Size([1, 256, 32, 32])
        # 11 Conv2d torch.Size([1, 512, 31, 31])
        # 12 InstanceNorm2d torch.Size([1, 512, 31, 31])
        # 13 LeakyReLU torch.Size([1, 512, 31, 31])
        # 14 Conv2d torch.Size([1, 1, 30, 30])
        return self.model(input)

    def __backward_hook(self, module, grad_in, grad_out):
        self.grads.append(grad_out[0].detach())

    def __forward_hook(self, module, input, output):
        self.fmaps.append(output)

    def compute_cam(self, feature_map, grads):
        """
        feature_map: np.array [B, C, H, W]
        grads: np.array, [B, C, H, W]
        return: np.array, [B, H, W]
        """
        alpha = torch.einsum('bchw -> bc', grads)  # GAP
        cam = torch.einsum('bchw, bc -> bhw', feature_map, alpha)
        cam = (cam - cam.min()) / (cam.max() - cam.min()) * 2 - 1

        return cam


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
        if (no_antialias):
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=7, stride=1, padding=3), nn.LeakyReLU(0.2, True),
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
        # for m in self.model:
        #     input = m(input)
        #     print(m._get_name(), input.shape)
        # Conv2d
        # torch.Size([2, 64, 255, 255])
        # LeakyReLU
        # torch.Size([2, 64, 255, 255])
        # Downsample
        # torch.Size([2, 64, 128, 128])
        # Conv2d
        # torch.Size([2, 128, 127, 127])
        # InstanceNorm2d
        # torch.Size([2, 128, 127, 127])
        # LeakyReLU
        # torch.Size([2, 128, 127, 127])
        # Downsample
        # torch.Size([2, 128, 64, 64])
        # Conv2d
        # torch.Size([2, 256, 63, 63])
        # InstanceNorm2d
        # torch.Size([2, 256, 63, 63])
        # LeakyReLU
        # torch.Size([2, 256, 63, 63])
        # Downsample
        # torch.Size([2, 256, 32, 32])
        # Conv2d
        # torch.Size([2, 512, 31, 31])
        # InstanceNorm2d
        # torch.Size([2, 512, 31, 31])
        # LeakyReLU
        # torch.Size([2, 512, 31, 31])
        # Conv2d
        # torch.Size([2, 1, 30, 30])
        return self.model(input)


class NLayerSADiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerSADiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if (no_antialias):
            sequence = [nn.utils.spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
                        nn.LeakyReLU(0.2, True)]
        else:
            sequence = [nn.utils.spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw)),
                        nn.LeakyReLU(0.2, True),
                        Downsample(ndf)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if (no_antialias):
                sequence += [
                    nn.utils.spectral_norm(
                        nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw,
                                  bias=use_bias)),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.utils.spectral_norm(
                        nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw,
                                  bias=use_bias)),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        # for m in self.model:
        #     input = m(input)
        #     print(m._get_name(), input.shape)
        # Conv2d
        # torch.Size([2, 64, 255, 255])
        # LeakyReLU
        # torch.Size([2, 64, 255, 255])
        # Downsample
        # torch.Size([2, 64, 128, 128])
        # Conv2d
        # torch.Size([2, 128, 127, 127])
        # InstanceNorm2d
        # torch.Size([2, 128, 127, 127])
        # LeakyReLU
        # torch.Size([2, 128, 127, 127])
        # Downsample
        # torch.Size([2, 128, 64, 64])
        # Conv2d
        # torch.Size([2, 256, 63, 63])
        # InstanceNorm2d
        # torch.Size([2, 256, 63, 63])
        # LeakyReLU
        # torch.Size([2, 256, 63, 63])
        # Downsample
        # torch.Size([2, 256, 32, 32])
        # Conv2d
        # torch.Size([2, 512, 31, 31])
        # InstanceNorm2d
        # torch.Size([2, 512, 31, 31])
        # LeakyReLU
        # torch.Size([2, 512, 31, 31])
        # Conv2d
        # torch.Size([2, 1, 30, 30])
        return self.model(input)


class NLayerUNetDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerUNetDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 3
        padw = 1

        dims = [input_nc, ndf, ndf * 2, ndf * 4, ndf * 8, ndf * 8]

        self.down_1 = Downsample(dims[1])
        self.down_2 = Downsample(dims[2])
        self.down_3 = Downsample(dims[3])
        self.down_4 = Downsample(dims[4])

        self.up_4 = Upsample(dims[5])
        self.up_3 = Upsample(dims[3])
        self.up_2 = Upsample2(2)
        self.up_1 = Upsample(dims[1])

        # 256 -> 128
        self.left_1 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True),
        )
        # 128 -> 64
        self.left_2 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(dims[2]),
            nn.LeakyReLU(0.2, True),
        )
        # 64 -> 32
        self.left_3 = nn.Sequential(
            nn.Conv2d(dims[2], dims[3], kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(dims[3]),
            nn.LeakyReLU(0.4, True),
        )
        # 32 -> 16 -> 8
        self.left_4 = nn.Sequential(
            nn.Conv2d(dims[3], dims[4], kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(dims[4]),
            nn.LeakyReLU(0.4, True),
        )

        self.pool = nn.AdaptiveMaxPool2d(1)
        self.out_4 = nn.Sequential(
            nn.Linear(dims[4], 1)
        )
        # 32 -> 64
        self.right_3 = nn.Sequential(
            nn.Conv2d(dims[4] * 2, dims[3], kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(dims[2]),
            nn.LeakyReLU(0.4, True),

        )
        self.out_3 = nn.Conv2d(dims[3], 1, kernel_size=kw, stride=1, padding=padw)

        # 64 -> 128
        self.right_2 = nn.Sequential(
            nn.Conv2d(dims[4], dims[2], kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(dims[2]),
            nn.LeakyReLU(0.2, True),
        )
        self.out_2 = nn.Conv2d(dims[2], 1, kernel_size=kw, stride=1, padding=padw)

        # 128 -> 256
        self.right_1 = nn.Sequential(
            nn.Conv2d(dims[3], dims[1], kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(dims[1]),
            nn.LeakyReLU(0.2, True),
        )
        self.out_1 = nn.Conv2d(dims[1], 1, kernel_size=kw, stride=1, padding=padw)
        self.out_0 = nn.Conv2d(dims[1], 1, kernel_size=kw, stride=1, padding=padw)

    def forward(self, x):
        """Standard forward."""
        res = [x]
        for model, downsample in zip((self.left_1, self.left_2, self.left_3, self.left_4),
                                     (self.down_1, self.down_2, self.down_3, self.down_4)):
            x = model(x)
            res.append(x)
            x = downsample(x)
        out_4 = self.out_4(self.pool(x).flatten(1))

        x = self.up_4(x)
        x = self.right_3(torch.cat([res[-1], x], dim=1))
        out_3 = self.out_3(x)
        x = self.up_3(x)

        x = self.right_2(torch.cat([res[-2], x], dim=1))
        out_2 = self.out_2(x)
        x = self.up_2(x)

        x = self.right_1(torch.cat([res[-3], x], dim=1))
        out_1 = self.out_1(x)
        x = self.up_1(x)
        out_0 = self.out_0(x)
        # print(torch.cat([out_0.flatten(1), out_1.flatten(1), out_2.flatten(1), out_3.flatten(1), out_4.flatten(1)], dim=1).shape)
        return torch.cat([out_0.flatten(1), out_1.flatten(1), out_2.flatten(1), out_3.flatten(1), out_4.flatten(1)],
                         dim=1)


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, norm_layer=None, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if norm_layer is not None:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p, bias=use_bias),
                                     norm_layer(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p, bias=True),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x


class UNet3PlusDiscriminator(nn.Module):
    """https://github.com/ZJUGiveLab/UNet-Version/blob/master/models/UNet_3Plus.py"""

    def __init__(self, in_channels=3, out_channels=1, feature_scale=4, is_deconv=True, norm_layer=nn.InstanceNorm2d):
        super(UNet3PlusDiscriminator, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_layer = norm_layer
        self.feature_scale = feature_scale

        # filters = [64, 128, 256, 512, 1024]
        filters = [32, 64, 128, 256, 512]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.norm_layer)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.norm_layer)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.norm_layer)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.norm_layer)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.norm_layer)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = norm_layer(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = norm_layer(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = norm_layer(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = norm_layer(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = norm_layer(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = norm_layer(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = norm_layer(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = norm_layer(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = norm_layer(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = norm_layer(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = norm_layer(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = norm_layer(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = norm_layer(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = norm_layer(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = norm_layer(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = norm_layer(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = norm_layer(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = norm_layer(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = norm_layer(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = norm_layer(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = norm_layer(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = norm_layer(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = norm_layer(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = norm_layer(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32, mode='bilinear')  ###
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        # DeepSup
        self.outconv1 = nn.Conv2d(self.UpChannels, self.out_channels, 3, padding=1)
        self.outconv2 = nn.Conv2d(self.UpChannels, self.out_channels, 3, padding=1)
        self.outconv3 = nn.Conv2d(self.UpChannels, self.out_channels, 3, padding=1)
        self.outconv4 = nn.Conv2d(self.UpChannels, self.out_channels, 3, padding=1)
        self.outconv5 = nn.Conv2d(filters[4], self.out_channels, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))  # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))))  # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))))  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))))  # hd1->320*320*UpChannels

        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5)  # 16->256

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4)  # 32->256

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3)  # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2)  # 128->256

        d1 = self.outconv1(hd1)  # 256

        # return torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5)
        return (d1 + d2 + d3 + d4 + d5) / 5


class UnetDiscriminator(nn.Module):
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
        super(UnetDiscriminator, self).__init__()
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


class SwinUnetDiscriminator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False, opt=None):
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
        super(SwinUnetDiscriminator, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ndf, kernel_size=7, padding=0, bias=use_bias),
                 # norm_layer(ndf),
                 # nn.ReLU(True)
                 ]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                      norm_layer(ndf * mult * 2),
                      nn.LeakyReLU(0.2, True),
                      Downsample(ndf * mult * 2)]

        mult = 2 ** n_downsampling

        self.n_blocks = 4

        window_size = [4, 4, 8, 8, 16, 8, 8, 4, 4]
        num_heads = [4, 4, 8, 8, 16, 8, 8, 4, 4]

        for i in range(self.n_blocks):  # add ResNet blocks
            model += [
                nn.Sequential(
                    PreNorm(num_patches=64 * 64, dim=ndf * mult, PE=i == 0),
                    SwinTransformerBlock(dim=ndf * mult, input_resolution=(64, 64),
                                         num_heads=num_heads[i], window_size=window_size[i],
                                         shift_size=0 if (i % 2 == 0) else window_size[i] // 2,
                                         mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                         drop=0., attn_drop=0., drop_path=0.),
                    Rearrange('B (H W) C -> B C H W', H=64)
                )
            ]

        # for i in range(n_downsampling):  # add upsampling layers
        #     mult = 2 ** (n_downsampling - i)
        #     if no_antialias_up:
        #         model += [nn.ConvTranspose2d(ngf * mult, int(ndf * mult / 2),
        #                                      kernel_size=3, stride=2,
        #                                      padding=1, output_padding=1,
        #                                      bias=use_bias),
        #                   norm_layer(int(ndf * mult / 2)),
        #                   nn.ReLU(True)]
        #     else:
        #         model += [Upsample(ndf * mult),
        #                   nn.Conv2d(ndf * mult, int(ndf * mult / 2),
        #                             kernel_size=3, stride=1,
        #                             padding=1,  # output_padding=1,
        #                             bias=use_bias),
        #                   norm_layer(int(ndf * mult / 2)),
        #                   nn.ReLU(True)]
        model += [
            # nn.Conv2d(ndf * mult, ndf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            # norm_layer(ndf * mult * 2),
            # nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ndf * mult, 1, kernel_size=7, padding=0),
        ]

        self.model = nn.Sequential(*model)

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

    def forward(self, input, layers=[], encode_only=False):
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            if 0 in layers:
                feats.append(feat)
                layers = layers[1:]  # remove 0 from list
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
            fake = self.model(input)
            return fake


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
