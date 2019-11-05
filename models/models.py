from typing import Union, Tuple
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    CONV_NAME: str = 'conv'

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 2,
                 norm_layer=None,
                 activation=None):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
                ('pad', nn.ReflectionPad2d(padding=kernel_size//2)),
                (BasicBlock.CONV_NAME, nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 bias=False,
                                                 )
                 ),
                ]
            ))
        if norm_layer is not None:
            self.features.add_module('norm', norm_layer)
        if activation is not None:
            self.features.add_module('activation', activation)

    def forward(self, x):
        return self.features(x)

    @property
    def weight(self):
        return self.features.__getattr__(BasicBlock.CONV_NAME).weight


class ResizeConv(nn.Module):
    CONV_NAME: str = 'conv_block'

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 target_size: Union[int, Tuple[int, ...]] = None,
                 scale_factor: Union[float, Tuple[float, ...]] = None,
                 mode: str = 'bilinear',
                 norm_layer=None,
                 activation=None
                 ):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
                ('up', nn.Upsample(size=target_size, scale_factor=scale_factor, mode=mode, align_corners=True)),
                ]
            ))
        self.features.add_module(ResizeConv.CONV_NAME,
                                 BasicBlock(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=1,
                                            norm_layer=norm_layer,
                                            activation=activation))

    @property
    def weight(self):
        return self.features.__getattr__(ResizeConv.CONV_NAME).weight

    def forward(self, x):
        return self.features(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9,
                 use_resize: bool = True):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            if not use_resize:
                upsampling_layer = nn.ConvTranspose2d(in_features,
                                                      out_features,
                                                      3,
                                                      stride=2, padding=1, output_padding=1)
            else:
                upsampling_layer = ResizeConv(in_features, out_features,
                                              kernel_size=3,
                                              scale_factor=2)
            model += [upsampling_layer,
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)
        # self.classifier = nn.Linear(num_features, 2)

    def forward(self, x):
        # Average pooling and flatten
        features = self.model(x)
        features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size()[0], -1)
        return features


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc: int,
                 ndf: int = 64,
                 n_layers: int = 3,
                 norm_layer: type = nn.InstanceNorm2d):
        super().__init__()
        use_bias = norm_layer == nn.InstanceNorm2d or norm_layer == nn.BatchNorm2d

        kernel_size = 4
        pad_width = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=2, padding=pad_width),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kernel_size, stride=2, padding=pad_width, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kernel_size, stride=1, padding=pad_width, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kernel_size, stride=1, padding=pad_width)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        # x: N*3*256*256
        features = self.model(x)
        # N*1*30*30
        # adaptive global average pooling
        features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size()[0], -1)
        return features
