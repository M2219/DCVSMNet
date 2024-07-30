from __future__ import print_function

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from torch.autograd.function import Function
from torch.nn.modules.container import Sequential
from typing import Optional


class BasicConv(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        deconv: bool = False,
        is_3d: bool = False,
        bn: bool = True,
        relu: bool = True,
        **kwargs
    ) -> None:
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(
                    in_channels, out_channels, bias=False, **kwargs
                )
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(
                    in_channels, out_channels, bias=False, **kwargs
                )
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)  # , inplace=True)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int,
        downsample: Optional[Sequential],
        pad: int,
        dilation: int,
    ) -> None:
        super(BasicBlock, self).__init__()

        self.conv1 = BasicConv(
            inplanes,
            planes,
            bn=True,
            relu=True,
            kernel_size=3,
            stride=stride,
            padding=pad,
            dilation=dilation,
        )
        self.conv2 = BasicConv(
            planes,
            planes,
            bn=True,
            relu=False,
            kernel_size=3,
            stride=1,
            padding=pad,
            dilation=dilation,
        )

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class Conv2x(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        deconv: bool = False,
        is_3d: bool = False,
        concat: bool = True,
        keep_concat: bool = True,
        bn: bool = True,
        relu: bool = True,
        keep_dispc: bool = False,
    ) -> None:
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(
                in_channels,
                out_channels,
                deconv,
                is_3d,
                bn=True,
                relu=True,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            )
        else:
            self.conv1 = BasicConv(
                in_channels,
                out_channels,
                deconv,
                is_3d,
                bn=True,
                relu=True,
                kernel_size=kernel,
                stride=2,
                padding=1,
            )

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(
                out_channels * 2,
                out_channels * mul,
                False,
                is_3d,
                bn,
                relu,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            self.conv2 = BasicConv(
                out_channels,
                out_channels,
                False,
                is_3d,
                bn,
                relu,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def forward(self, x: torch.Tensor, rem: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode="nearest")
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


def groupwise_difference(
    fea1: torch.Tensor, fea2: torch.Tensor, num_groups: int
) -> torch.Tensor:
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    fea1 = fea1.view([B, num_groups, channels_per_group, H, W])
    fea2 = fea2.view([B, num_groups, channels_per_group, H, W])
    cost = torch.pow((fea1 - fea2), 2).sum(2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_substract_volume(
    refimg_fea: torch.Tensor, targetimg_fea: torch.Tensor, maxdisp: int, num_groups: int
) -> torch.Tensor:
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_difference(
                refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i], num_groups
            )
        else:
            volume[:, :, i, :, :] = groupwise_difference(
                refimg_fea, targetimg_fea, num_groups
            )
    volume = volume.contiguous()
    return volume


def build_concat_volume(
    refimg_fea: torch.Tensor, targetimg_fea: torch.Tensor, maxdisp: int
) -> torch.Tensor:
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, :] = refimg_fea[:, :, :, :]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def groupwise_correlation(
    fea1: torch.Tensor, fea2: torch.Tensor, num_groups: int
) -> torch.Tensor:
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(
    refimg_fea: torch.Tensor, targetimg_fea: torch.Tensor, maxdisp: int, num_groups: int
) -> torch.Tensor:
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(
                refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i], num_groups
            )
        else:
            volume[:, :, i, :, :] = groupwise_correlation(
                refimg_fea, targetimg_fea, num_groups
            )
    volume = volume.contiguous()
    return volume


def norm_correlation(fea1: torch.Tensor, fea2: torch.Tensor) -> torch.Tensor:
    cost = torch.mean(
        (
            (fea1 / (torch.norm(fea1, 2, 1, True) + 1e-05))
            * (fea2 / (torch.norm(fea2, 2, 1, True) + 1e-05))
        ),
        dim=1,
        keepdim=True,
    )
    return cost


def build_norm_correlation_volume(
    refimg_fea: torch.Tensor, targetimg_fea: torch.Tensor, maxdisp: int
) -> torch.Tensor:
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = norm_correlation(
                refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i]
            )
        else:
            volume[:, :, i, :, :] = norm_correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume


def context_upsample(depth_low: torch.Tensor, up_weights: torch.Tensor) -> torch.Tensor:
    b, c, h, w = depth_low.shape

    depth_unfold = F.unfold(depth_low.reshape(b, c, h, w), 3, 1, 1).reshape(b, -1, h, w)
    depth_unfold = F.interpolate(depth_unfold, (h * 4, w * 4), mode="nearest").reshape(
        b, 9, h * 4, w * 4
    )

    depth = (depth_unfold * up_weights).sum(1)

    return depth


def regression_topk(
    cost: torch.Tensor, disparity_samples: torch.Tensor, k: int
) -> torch.Tensor:
    _, ind = cost.sort(1, True)
    pool_ind = ind[:, :k]
    cost = torch.gather(cost, 1, pool_ind)
    prob = F.softmax(cost, 1)
    disparity_samples = torch.gather(disparity_samples, 1, pool_ind)
    pred = torch.sum(disparity_samples * prob, dim=1, keepdim=True)
    return pred
