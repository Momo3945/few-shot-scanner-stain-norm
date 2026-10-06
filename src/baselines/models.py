#!/usr/bin/env python3
"""
models.py -- learned stain-normalisation baselines (P2-11b, tab:baselines).

Network definitions for StainNet, ParamNet and the CycleGAN components used for
StainGAN, adapted from the authors' public code (neither repo states a licence, so
this file re-states only the model/loss definitions needed; training and inference
are our own scripts):

  StainNet  -- github.com/khtao/StainNet  @ 94c20b31c0784d0d49468265afdde3d131d6afc8 (models.py)
               Kang et al. 2021, "StainNet: a fast and robust stain normalization network".
  ParamNet  -- github.com/khtao/ParamNet  @ b82a2c4fce326933868860ac0da234b6e49bd027
               (source/model.py, source/resnet.py, source/image_pool.py, train.py)
               Kang et al. 2023/24, "ParamNet: a dynamic parameter network for fast
               multi-to-one stain normalization".
  StainGAN  -- Shaban et al. 2019 (ISBI). The authors' repo (xtarx/StainGAN) ships no
               trained weights, so StainGAN here is the standard CycleGAN
               (ResnetGenerator + PatchGAN, LSGAN, cycle + identity) trained by us.

All networks consume/produce float tensors in [-1, 1] (N, 3, H, W). Pure definitions,
no I/O -- mirrors the style of baseline_methods.py / metrics.py.
"""

from __future__ import annotations

import functools
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# ----------------------------------------------------------------------
# StainNet: 3 x (1x1 conv) per-pixel colour map, 32 channels (~1.3k params)
# ----------------------------------------------------------------------
class StainNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_layer=3, n_channel=32, kernel_size=1):
        super().__init__()
        layers = [nn.Conv2d(input_nc, n_channel, kernel_size, bias=True, padding=kernel_size // 2),
                  nn.ReLU(True)]
        for _ in range(n_layer - 2):
            layers += [nn.Conv2d(n_channel, n_channel, kernel_size, bias=True, padding=kernel_size // 2),
                       nn.ReLU(True)]
        layers.append(nn.Conv2d(n_channel, output_nc, kernel_size, bias=True, padding=kernel_size // 2))
        self.rgb_trans = nn.Sequential(*layers)

    def forward(self, x):
        return self.rgb_trans(x)


# ----------------------------------------------------------------------
# ParamNet: ResNet18 (no BatchNorm, as upstream) predicts the weights of a
# per-image 1x1-conv colour-mapping network, applied to the full-res input.
# ----------------------------------------------------------------------
class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Conv2d(self.inplanes, planes, 1, stride, bias=True)
        layers = [_BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers += [_BasicBlock(planes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


class ParamNet(nn.Module):
    """Upstream defaults: resnet18 backbone, resample_size=128, channels=8, layers=2."""

    def __init__(self, resample_size=128, channels=8, layers=2):
        super().__init__()
        assert channels >= 3 and isinstance(channels, int)
        if layers == 1:
            n_params = 12
        elif layers == 2:
            n_params = 7 * channels + 3
        elif layers == 3:
            n_params = 8 * channels + channels * channels + 3
        else:
            raise NotImplementedError("layers must be one of 1,2,3")
        self.backbone = _ResNet18(num_classes=n_params)
        self.resample_size = resample_size
        self.channels = channels
        self.layers = layers
        self.param = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.data_range = nn.Parameter(torch.tensor(4.5, dtype=torch.float))

    def _color_map(self, x_it, w_it):
        ch = self.channels
        x = x_it.unsqueeze(0)
        if self.layers == 1:
            return F.conv2d(x, w_it[:9].reshape(3, 3, 1, 1), w_it[9:12].reshape(3))
        if self.layers == 2:
            h = F.relu(F.conv2d(x, w_it[:3 * ch].reshape(ch, 3, 1, 1), w_it[3 * ch:4 * ch].reshape(ch)))
            return F.conv2d(h, w_it[4 * ch:7 * ch].reshape(3, ch, 1, 1), w_it[7 * ch:7 * ch + 3].reshape(3))
        h = F.relu(F.conv2d(x, w_it[:3 * ch].reshape(ch, 3, 1, 1), w_it[3 * ch:4 * ch].reshape(ch)))
        h = F.relu(F.conv2d(h, w_it[4 * ch:4 * ch + ch * ch].reshape(ch, ch, 1, 1),
                            w_it[4 * ch + ch * ch:5 * ch + ch * ch].reshape(ch)))
        return F.conv2d(h, w_it[5 * ch + ch * ch:8 * ch + ch * ch].reshape(3, ch, 1, 1),
                        w_it[8 * ch + ch * ch:8 * ch + ch * ch + 3].reshape(3))

    def forward(self, x):
        x_small = F.interpolate(x, size=self.resample_size, mode="bilinear", align_corners=True)
        w = self.data_range * self.backbone(x_small).tanh()
        out = torch.cat([self._color_map(xi, wi) for xi, wi in zip(x, w)], dim=0)
        return (out + x * self.param).tanh()


# ----------------------------------------------------------------------
# CycleGAN components (ParamNet's auxiliary generator + StainGAN)
# ----------------------------------------------------------------------
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_bias):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, bias=use_bias), norm_layer(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, bias=use_bias), norm_layer(dim))

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.InstanceNorm2d, n_blocks=6):
        super().__init__()
        use_bias = (norm_layer.func == nn.InstanceNorm2d if isinstance(norm_layer, functools.partial)
                    else norm_layer == nn.InstanceNorm2d)
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, 7, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
        for i in range(2):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]
        mult = 4
        model += [ResnetBlock(ngf * mult, norm_layer, use_bias) for _ in range(n_blocks)]
        for i in range(2):
            mult = 2 ** (2 - i)
            model += [nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 3, stride=2, padding=1,
                                         output_padding=1, bias=use_bias),
                      norm_layer(ngf * mult // 2), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    """PatchGAN."""

    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        use_bias = (norm_layer.func == nn.InstanceNorm2d if isinstance(norm_layer, functools.partial)
                    else norm_layer == nn.InstanceNorm2d)
        seq = [nn.Conv2d(input_nc, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        mult = 1
        for n in range(1, n_layers):
            prev, mult = mult, min(2 ** n, 8)
            seq += [nn.Conv2d(ndf * prev, ndf * mult, 4, 2, 1, bias=use_bias),
                    norm_layer(ndf * mult), nn.LeakyReLU(0.2, True)]
        prev, mult = mult, min(2 ** n_layers, 8)
        seq += [nn.Conv2d(ndf * prev, ndf * mult, 4, 1, 1, bias=use_bias),
                norm_layer(ndf * mult), nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf * mult, 1, 4, 1, 1)]
        self.model = nn.Sequential(*seq)

    def forward(self, x):
        return self.model(x)


def gan_loss(pred_real, pred_fake=None):
    """LSGAN. One arg: generator loss (fool D). Two args: discriminator loss."""
    if pred_fake is None:
        return F.mse_loss(pred_real, torch.ones_like(pred_real))
    return 0.5 * (F.mse_loss(pred_real, torch.ones_like(pred_real))
                  + F.mse_loss(pred_fake, torch.zeros_like(pred_fake)))


def init_weights(net, init_gain=0.02):
    def _init(m):
        name = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in name or "Linear" in name):
            init.normal_(m.weight.data, 0.0, init_gain)
            if getattr(m, "bias", None) is not None:
                init.constant_(m.bias.data, 0.0)
        elif "BatchNorm2d" in name:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(_init)


class ImagePool:
    """History buffer of generated images for discriminator updates (CycleGAN)."""

    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        out = []
        for img in images:
            img = img.detach().unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(img)
                out.append(img)
            elif random.random() > 0.5:
                k = random.randint(0, self.pool_size - 1)
                out.append(self.images[k].clone())
                self.images[k] = img
            else:
                out.append(img)
        return torch.cat(out, 0)
