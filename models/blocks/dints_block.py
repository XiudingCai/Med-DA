# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Tuple, Union
from collections import OrderedDict

import torch
from torch import nn

from monai.networks.layers.factories import Conv
from monai.networks.layers.utils import get_act_layer, get_norm_layer

from ..networks import Upsample, Downsample

# __all__ = ["FactorizedIncreaseBlock", "FactorizedReduceBlock", "P3DActiConvNormBlock", "ActiConvNormBlock"]


class FactorizedIncreaseBlock(nn.Sequential):
    """
    Up-sampling the features by two using linear interpolation and convolutions.
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        spatial_dims: int = 3,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = "INSTANCE",
    ):
        """
        Args:
            in_channel: number of input channels
            out_channel: number of output channels
            spatial_dims: number of spatial dimensions
            act_name: activation layer type and arguments.
            norm_name: feature normalization type and arguments.
        """
        super().__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._spatial_dims = spatial_dims
        if self._spatial_dims not in (2, 3):
            raise ValueError("spatial_dims must be 2 or 3.")

        conv_type = Conv[Conv.CONV, self._spatial_dims]
        mode = "trilinear" if self._spatial_dims == 3 else "bilinear"
        self.add_module("up", nn.Upsample(scale_factor=2, mode=mode, align_corners=True))
        self.add_module("acti", get_act_layer(name=act_name))
        self.add_module(
            "conv",
            conv_type(
                in_channels=self._in_channel,
                out_channels=self._out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=False,
                dilation=1,
            ),
        )
        self.add_module(
            "norm", get_norm_layer(name=norm_name, spatial_dims=self._spatial_dims, channels=self._out_channel)
        )


class FactorizedReduceBlock(nn.Module):
    """
    Down-sampling the feature by 2 using stride.
    The length along each spatial dimension must be a multiple of 2.
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        spatial_dims: int = 3,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = "INSTANCE",
    ):
        """
        Args:
            in_channel: number of input channels
            out_channel: number of output channels.
            spatial_dims: number of spatial dimensions.
            act_name: activation layer type and arguments.
            norm_name: feature normalization type and arguments.
        """
        super().__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._spatial_dims = spatial_dims
        if self._spatial_dims not in (2, 3):
            raise ValueError("spatial_dims must be 2 or 3.")

        conv_type = Conv[Conv.CONV, self._spatial_dims]

        self.act = get_act_layer(name=act_name)
        self.conv_1 = conv_type(
            in_channels=self._in_channel,
            out_channels=self._out_channel // 2,
            kernel_size=1,
            stride=2,
            padding=0,
            groups=1,
            bias=False,
            dilation=1,
        )
        self.conv_2 = conv_type(
            in_channels=self._in_channel,
            out_channels=self._out_channel - self._out_channel // 2,
            kernel_size=1,
            stride=2,
            padding=0,
            groups=1,
            bias=False,
            dilation=1,
        )
        self.norm = get_norm_layer(name=norm_name, spatial_dims=self._spatial_dims, channels=self._out_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The length along each spatial dimension must be a multiple of 2.
        """
        x = self.act(x)
        if self._spatial_dims == 3:
            out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:, 1:])], dim=1)
        else:
            out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.norm(out)
        return out


class P3DActiConvNormBlock(nn.Sequential):
    """
    -- (act) -- (conv) -- (norm) --
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        padding: int,
        mode: int = 0,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = "INSTANCE",
    ):
        """
        Args:
            in_channel: number of input channels.
            out_channel: number of output channels.
            kernel_size: kernel size to be expanded to 3D.
            padding: padding size to be expanded to 3D.
            mode: mode for the anisotropic kernels:

                - 0: ``(k, k, 1)``, ``(1, 1, k)``,
                - 1: ``(k, 1, k)``, ``(1, k, 1)``,
                - 2: ``(1, k, k)``. ``(k, 1, 1)``.

            act_name: activation layer type and arguments.
            norm_name: feature normalization type and arguments.
        """
        super().__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._p3dmode = int(mode)

        conv_type = Conv[Conv.CONV, 3]

        if self._p3dmode == 0:  # (k, k, 1), (1, 1, k)
            kernel_size0 = (kernel_size, kernel_size, 1)
            kernel_size1 = (1, 1, kernel_size)
            padding0 = (padding, padding, 0)
            padding1 = (0, 0, padding)
        elif self._p3dmode == 1:  # (k, 1, k), (1, k, 1)
            kernel_size0 = (kernel_size, 1, kernel_size)
            kernel_size1 = (1, kernel_size, 1)
            padding0 = (padding, 0, padding)
            padding1 = (0, padding, 0)
        elif self._p3dmode == 2:  # (1, k, k), (k, 1, 1)
            kernel_size0 = (1, kernel_size, kernel_size)
            kernel_size1 = (kernel_size, 1, 1)
            padding0 = (0, padding, padding)
            padding1 = (padding, 0, 0)
        else:
            raise ValueError("`mode` must be 0, 1, or 2.")

        self.add_module("acti", get_act_layer(name=act_name))
        self.add_module(
            "conv",
            conv_type(
                in_channels=self._in_channel,
                out_channels=self._in_channel,
                kernel_size=kernel_size0,
                stride=1,
                padding=padding0,
                groups=1,
                bias=False,
                dilation=1,
            ),
        )
        self.add_module(
            "conv_1",
            conv_type(
                in_channels=self._in_channel,
                out_channels=self._out_channel,
                kernel_size=kernel_size1,
                stride=1,
                padding=padding1,
                groups=1,
                bias=False,
                dilation=1,
            ),
        )
        self.add_module("norm", get_norm_layer(name=norm_name, spatial_dims=3, channels=self._out_channel))


class P2DActiConvNormBlock(nn.Sequential):
    """
    -- (act) -- (conv) -- (norm) --
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        padding: int,
        mode: int = 0,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = "INSTANCE",
    ):
        """
        Args:
            in_channel: number of input channels.
            out_channel: number of output channels.
            kernel_size: kernel size to be expanded to 3D.
            padding: padding size to be expanded to 3D.
            mode: mode for the anisotropic kernels:

                - 0: ``(1, k)``, ``(k, 1)``,
                - 1: ``(k, 1)``, ``(1, k)``.

            act_name: activation layer type and arguments.
            norm_name: feature normalization type and arguments.
        """
        super().__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._p2dmode = int(mode)

        conv_type = Conv[Conv.CONV, 2]

        if self._p2dmode == 0:  # (1, k), (k, 1)
            kernel_size0 = (1, kernel_size)
            kernel_size1 = (kernel_size, 1)
            padding0 = (0, padding)
            padding1 = (padding, 0)
        elif self._p2dmode == 1:  # (k, 1), (1, k)
            kernel_size0 = (kernel_size, 1)
            kernel_size1 = (1, kernel_size)
            padding0 = (padding, 0)
            padding1 = (0, padding)
        else:
            raise ValueError("`mode` must be 0 or 1.")

        self.add_module("acti", get_act_layer(name=act_name))
        self.add_module(
            "conv",
            conv_type(
                in_channels=self._in_channel,
                out_channels=self._in_channel,
                kernel_size=kernel_size0,
                stride=1,
                padding=padding0,
                groups=1,
                bias=False,
                dilation=1,
            ),
        )
        self.add_module(
            "conv_1",
            conv_type(
                in_channels=self._in_channel,
                out_channels=self._out_channel,
                kernel_size=kernel_size1,
                stride=1,
                padding=padding1,
                groups=1,
                bias=False,
                dilation=1,
            ),
        )
        self.add_module("norm", get_norm_layer(name=norm_name, spatial_dims=2, channels=self._out_channel))


class ActiConvNormBlock(nn.Sequential):
    """
    -- (Acti) -- (Conv) -- (Norm) --
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 3,
        padding: int = 1,
        spatial_dims: int = 3,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = "INSTANCE",
    ):
        """
        Args:
            in_channel: number of input channels.
            out_channel: number of output channels.
            kernel_size: kernel size of the convolution.
            padding: padding size of the convolution.
            spatial_dims: number of spatial dimensions.
            act_name: activation layer type and arguments.
            norm_name: feature normalization type and arguments.
        """
        super().__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._spatial_dims = spatial_dims

        conv_type = Conv[Conv.CONV, self._spatial_dims]
        self.add_module("acti", get_act_layer(name=act_name))
        self.add_module(
            "conv",
            conv_type(
                in_channels=self._in_channel,
                out_channels=self._out_channel,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=1,
                bias=False,
                dilation=1,
            ),
        )
        self.add_module(
            "norm", get_norm_layer(name=norm_name, spatial_dims=self._spatial_dims, channels=self._out_channel)
        )


class ActiDSConvNormBlock(nn.Sequential):
    """
    -- (Acti) -- (Conv) -- (Norm) --
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 3,
        padding: int = 1,
        spatial_dims: int = 2,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = "INSTANCE",
    ):
        """
        Args:
            in_channel: number of input channels.
            out_channel: number of output channels.
            kernel_size: kernel size of the convolution.
            padding: padding size of the convolution.
            spatial_dims: number of spatial dimensions.
            act_name: activation layer type and arguments.
            norm_name: feature normalization type and arguments.
        """
        super().__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._spatial_dims = spatial_dims

        conv_type = Conv[Conv.CONV, self._spatial_dims]
        self.add_module("acti", get_act_layer(name=act_name))
        self.add_module(
            "conv",
            conv_type(
                in_channels=self._in_channel,
                out_channels=self._out_channel,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=1,
                bias=False,
                dilation=1,
            ),
        )
        self.add_module(
            "norm", get_norm_layer(name=norm_name, spatial_dims=self._spatial_dims, channels=self._out_channel)
        )


class ActiMBInvConvNormBlock(nn.Module):
    """
    -- (Acti) -- (Conv) -- (Norm) --
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 3,
        padding: int = 1,
        spatial_dims: int = 2,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = "INSTANCE",
    ):
        """
        Args:
            in_channel: number of input channels.
            out_channel: number of output channels.
            kernel_size: kernel size of the convolution.
            padding: padding size of the convolution.
            spatial_dims: number of spatial dimensions.
            act_name: activation layer type and arguments.
            norm_name: feature normalization type and arguments.
        """
        super().__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._spatial_dims = spatial_dims

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(self._in_channel, feature_dim, 1, 1, 0, bias=False)),
                ('norm', get_norm_layer(name=norm_name, spatial_dims=self._spatial_dims, channels=self.feature_dim)),
                ('act', nn.ReLU6(inplace=True)),
            ]))

        pad = int(self.kernel_size // 2)
        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, feature_dim, kernel_size, 1, pad, groups=feature_dim, bias=False)),
            ('norm', get_norm_layer(name=norm_name, spatial_dims=self._spatial_dims, channels=feature_dim)),
            ('act', nn.ReLU6(inplace=True)),
        ]))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, self._out_channel, 1, 1, 0, bias=False)),
            ('norm', get_norm_layer(name=norm_name, spatial_dims=self._spatial_dims, channels=self._out_channel)),
        ]))

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x


class ConvNormActiBlock(nn.Module):
    """
    -- (Acti) -- (Conv) -- (Norm) --
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 3,
        padding: int = 1,
        spatial_dims: int = 3,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = "INSTANCE",
    ):
        """
        Args:
            in_channel: number of input channels.
            out_channel: number of output channels.
            kernel_size: kernel size of the convolution.
            padding: padding size of the convolution.
            spatial_dims: number of spatial dimensions.
            act_name: activation layer type and arguments.
            norm_name: feature normalization type and arguments.
        """
        super().__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._spatial_dims = spatial_dims

        conv_type = Conv[Conv.CONV, self._spatial_dims]

        self.model = nn.Sequential(OrderedDict([
            ('conv', conv_type(in_channels=self._in_channel, out_channels=self._out_channel,
                kernel_size=kernel_size, stride=1, padding=padding, groups=1, bias=False, dilation=1,)),
            ('norm', get_norm_layer(name=norm_name, spatial_dims=self._spatial_dims, channels=self._out_channel)),
            ('acti', get_act_layer(name=act_name)),
        ]))

    def forward(self, x):
        # return x + self.model(x)
        return self.model(x)


class ASConvNormActiBlock(nn.Module):
    """
    -- (Acti) -- (Conv) -- (Norm) --
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 3,
        padding: int = 2,
        spatial_dims: int = 2,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = "INSTANCE",
    ):
        """
        Args:
            in_channel: number of input channels.
            out_channel: number of output channels.
            kernel_size: kernel size of the convolution.
            padding: padding size of the convolution.
            spatial_dims: number of spatial dimensions.
            act_name: activation layer type and arguments.
            norm_name: feature normalization type and arguments.
        """
        super().__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._spatial_dims = spatial_dims

        conv_type = Conv[Conv.CONV, self._spatial_dims]

        self.model = nn.Sequential(OrderedDict([
            ('conv', conv_type(in_channels=self._in_channel, out_channels=self._out_channel,
                kernel_size=kernel_size, stride=1, padding=padding, groups=1, bias=False, dilation=2,)),
            ('norm', get_norm_layer(name=norm_name, spatial_dims=self._spatial_dims, channels=self._out_channel)),
            ('acti', get_act_layer(name=act_name)),
        ]))

    def forward(self, x):
        # return x + self.model(x)
        return self.model(x)


class DSConvNormActiBlock(nn.Module):
    """
    -- (Acti) -- (Conv) -- (Norm) --
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 3,
        padding: int = 1,
        spatial_dims: int = 2,
        expand_ratio=1,
        mid_channel=None,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = "INSTANCE",
    ):
        """
        Args:
            in_channel: number of input channels.
            out_channel: number of output channels.
            kernel_size: kernel size of the convolution.
            padding: padding size of the convolution.
            spatial_dims: number of spatial dimensions.
            act_name: activation layer type and arguments.
            norm_name: feature normalization type and arguments.
        """
        super().__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._spatial_dims = spatial_dims
        self._expand_ratio = expand_ratio
        self._mid_channel = mid_channel
        self._kernel_size = kernel_size

        if self._mid_channel is None:
            feature_dim = round(self._in_channel * self._expand_ratio)
        else:
            feature_dim = self._mid_channel

        if self._expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(self._in_channel, feature_dim, 1, 1, 0, bias=False)),
                ('norm', get_norm_layer(name=norm_name, spatial_dims=self._spatial_dims, channels=feature_dim)),
                ('act', nn.ReLU6(inplace=True)),
            ]))

        pad = int(self._kernel_size // 2)
        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, feature_dim, self._kernel_size, 1, pad, groups=feature_dim, bias=False)),
            ('norm', get_norm_layer(name=norm_name, spatial_dims=self._spatial_dims, channels=feature_dim)),
            ('act', nn.ReLU6(inplace=True)),
        ]))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, self._out_channel, 1, 1, 0, bias=False)),
            ('norm', get_norm_layer(name=norm_name, spatial_dims=self._spatial_dims, channels=self._out_channel)),
        ]))

    def forward(self, x):
        sc = x
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        # return x + sc
        return x


class AntialiasUpBlock(nn.Module):
    """
    Up-sampling the features by two using linear interpolation and convolutions.
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        spatial_dims: int = 2,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = "INSTANCE",
    ):
        """
        Args:
            in_channel: number of input channels
            out_channel: number of output channels
            spatial_dims: number of spatial dimensions
            act_name: activation layer type and arguments.
            norm_name: feature normalization type and arguments.
        """
        super().__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._spatial_dims = spatial_dims
        if self._spatial_dims not in (2, 3):
            raise ValueError("spatial_dims must be 2 or 3.")

        conv_type = Conv[Conv.CONV, self._spatial_dims]
        self.add_module(
            "conv",
            conv_type(
                in_channels=self._in_channel,
                out_channels=self._out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1,
                bias=False,
                dilation=1,
            ),
        )
        self.add_module(
            "norm", get_norm_layer(name=norm_name, spatial_dims=self._spatial_dims, channels=self._out_channel)
        )
        self.add_module("acti", get_act_layer(name=act_name))

        self.add_module("upsampling", Upsample(self._out_channel))


class AntialiasDownBlock(nn.Module):
    """
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        spatial_dims: int = 2,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = "INSTANCE",
    ):
        """
        Args:
            in_channel: number of input channels
            out_channel: number of output channels
            spatial_dims: number of spatial dimensions
            act_name: activation layer type and arguments.
            norm_name: feature normalization type and arguments.
        """
        super().__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._spatial_dims = spatial_dims
        if self._spatial_dims not in (2, 3):
            raise ValueError("spatial_dims must be 2 or 3.")

        conv_type = Conv[Conv.CONV, self._spatial_dims]
        self.add_module(
            "conv",
            conv_type(
                in_channels=self._in_channel,
                out_channels=self._out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1,
                bias=False,
                dilation=1,
            ),
        )
        self.add_module(
            "norm", get_norm_layer(name=norm_name, spatial_dims=self._spatial_dims, channels=self._out_channel)
        )
        self.add_module("acti", get_act_layer(name=act_name))

        self.add_module("downsampling", Downsample(self._out_channel))

