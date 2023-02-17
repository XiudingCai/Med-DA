import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np

from models.networks import *


class MPOSGeneratorV1(nn.Module):
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
        super(MPOSGeneratorV1, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # in
        self.model_in = HierDivergeHead()
        self.model_out = HierDivergeHead()

        model_in = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        num_layers = 12
        num_depths = 3
        channels = [64, 128, 256, 512]

        model_enc = HierDivergeHead(num_layers, num_depths, channels)
        model_dec = HierAggrHead(num_layers, num_depths, channels)

        model_out = [nn.ReflectionPad2d(3)]
        model_out += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_out += [nn.Tanh()]

        self.model_in = nn.Sequential(*model_in)
        self.model_enc = model_enc
        self.model_dec = model_dec
        self.model_out = nn.Sequential(*model_out)

    def forward(self, input, layers=[], encode_only=False):
        """
        input: input
        layers: nec_layers, e.g., [0, 4, 8, 12, 16]
        encode_only: when encode_only is True, for nce loss
        """
        x = self.model_in(input)
        x = self.model_enc(x, encode_only=encode_only)
        if encode_only:
            # print(len(x))
            return x
        x = self.model_dec(x)
        fake = self.model_out(x)
        return fake


class Cell_IN(nn.Module):
    def __init__(self, in_nfc, up_nfc=0, sam_nfc=0, dow_nfc=0, block=None, **kwargs):
        super(Cell_IN, self).__init__()

        self.in_nfc = in_nfc
        self.up_nfc = up_nfc
        self.sam_nfc = sam_nfc
        self.dow_nfc = dow_nfc

        self.block_sam = block(self.in_nfc, self.sam_nfc, **kwargs)
        if self.up_nfc > 0:
            self.block_up = block(self.in_nfc, self.up_nfc, **kwargs)
        if self.dow_nfc > 0:
            self.block_dow = block(self.in_nfc, self.dow_nfc, **kwargs)

            self.block_dow = nn.Sequential(*[
                nn.Conv2d(self.in_nfc, self.dow_nfc, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(self.dow_nfc),
                nn.ReLU(True),
                Downsample(self.dow_nfc)
            ])

    def forward(self, x):
        res = []
        if self.up_nfc == 0:
            res.append(None)
        else:
            res.append(self.block_up(x))

        res.append(self.block_sam(x))

        if self.dow_nfc == 0:
            res.append(None)
        else:
            res.append(self.block_dow(x))

        return res


class Cell_OUT(nn.Module):
    def __init__(self, dow_nfc=0, sam_nfc=0, up_nfc=0, out_nfc=0, block=None,  **kwargs):
        super(Cell_OUT, self).__init__()

        self.dow_nfc = dow_nfc
        self.sam_nfc = sam_nfc
        self.up_nfc = up_nfc
        self.out_nfc = out_nfc

        if self.dow_nfc > 0:
            self.block_dow = nn.Sequential(*[
                nn.Conv2d(self.dow_nfc, self.out_nfc, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(self.out_nfc),
                nn.ReLU(True),
                Downsample(self.out_nfc)
            ])

        self.block_sam = block(self.sam_nfc, self.out_nfc, **kwargs)

        if self.up_nfc > 0:
            self.block_up = nn.Sequential(*[
                nn.Conv2d(self.up_nfc, self.out_nfc, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(self.out_nfc),
                nn.ReLU(True),
                Upsample(self.out_nfc)
            ])

    # ↘ x_dow, → x_sam, ↗ x_up
    def forward(self, x_dow, x_sam, x_up):

        res = self.block_sam(x_sam)
        # print(x_sam.shape, res.shape)

        if self.up_nfc != 0:
            # print(res.shape, self.block_up(x_up).shape)
            res = res + self.block_up(x_up)

        if self.dow_nfc != 0:
            res = res + self.block_dow(x_dow)

        return res


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        self.num_layers = 12
        self.num_depths = 4

        channels = [64, 128, 256, 512]
        # in
        self.model_in = HierDivergeHead()
        self.model_out = HierDivergeHead()
        # out

    def forward(self, x):
        # x = self.model_in(x)

        inputs = [x]

        # encoder
        for layer_idx in range(self.num_depths-1):
            outputs = [torch.empty(0)]
            for depth_idx, x in enumerate(inputs):
                _, x_same, x_down = self.cell_tree_in[str((layer_idx, depth_idx))](x)
                outputs[-1] = torch.cat([outputs[-1], x_same], dim=1)
                outputs.append(x_down)
            inputs = outputs

        for x in inputs:
            print(x.shape)
        print()

        # decoder
        for layer_idx in range(self.num_depths - 1):
            outputs = []
            for depth_idx in range(len(inputs) - 1):
                x_dow = None
                x_sam = inputs[depth_idx]
                x_up = inputs[depth_idx + 1]
                x = self.cell_tree_out[str((layer_idx, depth_idx))](x_dow, x_sam, x_up)
                print(x.shape)
                outputs.append(x)
            inputs = outputs
            # print(len(outputs))
            # for x in outputs:
            #     print(layer_idx, x.shape)


class HierDivergeHead(nn.Module):
    def __init__(self, num_layers=12, num_depths=4, channels=(64, 128, 256, 512)):
        super(HierDivergeHead, self).__init__()

        self.num_layers = num_layers
        self.num_depths = num_depths

        # in
        self.cell_tree_in = nn.ModuleDict()
        for i in range(self.num_depths):
            for j in range(self.num_depths):
                in_nfc = channels[i]
                up_nfc = 0
                sam_nfc = channels[i]
                dow_nfc = channels[i]
                if i == j:
                    dow_nfc *= 2
                if i != 0:
                    sam_nfc //= 2
                kwargs = {'padding_type': 'reflect', 'norm_layer': nn.BatchNorm2d, 'use_dropout': False, 'use_bias': False}
                self.cell_tree_in[str((j, i))] = Cell_IN(in_nfc, up_nfc, sam_nfc, dow_nfc, block=ResnetBlockV2, **kwargs)

    def forward(self, x, encode_only=False):

        inputs = [x]

        if encode_only:
            feats = [x]

        # encoder
        for layer_idx in range(self.num_depths-1):
            outputs = [torch.empty(0, device=x.device)]
            for depth_idx, x in enumerate(inputs):
                _, x_same, x_down = self.cell_tree_in[str((layer_idx, depth_idx))](x)
                outputs[-1] = torch.cat([outputs[-1], x_same], dim=1)
                outputs.append(x_down)
            if encode_only:
                feats.extend(outputs)
            inputs = outputs

        if encode_only:
            return feats
        else:
            return inputs

class HierAggrHead(nn.Module):
    def __init__(self, num_layers=12, num_depths=4, channels=(64, 128, 256, 512)):
        super(HierAggrHead, self).__init__()

        self.num_layers = num_layers
        self.num_depths = num_depths

        # out
        self.cell_tree_out = nn.ModuleDict()
        for i in range(self.num_depths-1):
            for j in range(self.num_depths-1):
                dow_nfc = 0
                sam_nfc = channels[i]
                up_nfc = channels[i+1]
                out_nfc = channels[i]

                kwargs = {'padding_type': 'reflect', 'norm_layer': nn.BatchNorm2d, 'use_dropout': False, 'use_bias': False}
                self.cell_tree_out[str((j, i))] = Cell_OUT(dow_nfc, sam_nfc, up_nfc, out_nfc, block=ResnetBlockV2, **kwargs)

    def forward(self, inputs):

        # decoder
        for layer_idx in range(self.num_depths - 1):
            outputs = []
            for depth_idx in range(len(inputs) - 1):
                x_dow = None
                x_sam = inputs[depth_idx]
                x_up = inputs[depth_idx + 1]
                x = self.cell_tree_out[str((layer_idx, depth_idx))](x_dow, x_sam, x_up)
                outputs.append(x)
            inputs = outputs

        return inputs[0]


def main():
    model = MPOSGeneratorV1(3, 3)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    main()
