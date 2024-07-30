from __future__ import print_function

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import math
import gc
import time
import timm

from torch.nn.modules.container import Sequential
from typing import Dict, List, Tuple, Type
from torch.autograd import Variable

from .submodule import *


def count_parameters_in_MB(model):
    return (
        np.sum(
            np.prod(v.size())
            for name, v in model.named_parameters()
            if "auxiliary" not in name
        )
        / 1e6
    )


class SubModule(nn.Module):
    def __init__(self) -> None:
        super(SubModule, self).__init__()

    def weight_init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Conv3d):
                n = (
                    m.kernel_size[0]
                    * m.kernel_size[1]
                    * m.kernel_size[2]
                    * m.out_channels
                )
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class feature_extraction(nn.Module):
    def __init__(self, concat_feature_channel: int = 12) -> None:
        super(feature_extraction, self).__init__()

        self.inplanes = 32
        self.firstconv = nn.Sequential(
            BasicConv(
                3,
                32,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=2,
                dilation=1,
            ),
            BasicConv(
                32,
                32,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=1,
                dilation=1,
            ),
            BasicConv(
                32,
                32,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=1,
                dilation=1,
            ),
        )

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 2, 2)

        self.lastconv = nn.Sequential(
            BasicConv(
                320,
                128,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=1,
                dilation=1,
            ),
            nn.Conv2d(
                128,
                concat_feature_channel,
                kernel_size=1,
                padding=0,
                stride=1,
                bias=False,
            ),
        )

    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,
        blocks: int,
        stride: int,
        pad: int,
        dilation: int,
    ) -> Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        gwc_feature = torch.cat((l2, l3, l4), dim=1)
        concat_feature = self.lastconv(gwc_feature)

        return {"gwc_feature": gwc_feature, "concat_feature": concat_feature}


class hourglass_1(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super(hourglass_1, self).__init__()

        self.conv1 = nn.Sequential(
            BasicConv(
                in_channels,
                in_channels * 2,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=2,
                dilation=1,
            ),
            BasicConv(
                in_channels * 2,
                in_channels * 2,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=1,
                dilation=1,
            ),
        )

        self.conv2 = nn.Sequential(
            BasicConv(
                in_channels * 2,
                in_channels * 4,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=2,
                dilation=1,
            ),
            BasicConv(
                in_channels * 4,
                in_channels * 4,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=1,
                dilation=1,
            ),
        )

        self.conv3 = nn.Sequential(
            BasicConv(
                in_channels * 4,
                in_channels * 6,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=2,
                dilation=1,
            ),
            BasicConv(
                in_channels * 6,
                in_channels * 6,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=1,
                dilation=1,
            ),
        )

        self.conv3_up = BasicConv(
            in_channels * 6,
            in_channels * 4,
            deconv=True,
            is_3d=True,
            bn=True,
            relu=True,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )

        self.conv2_up = BasicConv(
            in_channels * 4,
            in_channels * 2,
            deconv=True,
            is_3d=True,
            bn=True,
            relu=True,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )

        self.conv1_up = BasicConv(
            in_channels * 2,
            1,
            deconv=True,
            is_3d=True,
            bn=False,
            relu=False,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )

        self.agg_0 = nn.Sequential(
            BasicConv(
                in_channels * 8,
                in_channels * 4,
                is_3d=True,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            BasicConv(
                in_channels * 4,
                in_channels * 4,
                is_3d=True,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            BasicConv(
                in_channels * 4,
                in_channels * 4,
                is_3d=True,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )

        self.agg_1 = nn.Sequential(
            BasicConv(
                in_channels * 4,
                in_channels * 2,
                is_3d=True,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            BasicConv(
                in_channels * 2,
                in_channels * 2,
                is_3d=True,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            BasicConv(
                in_channels * 2,
                in_channels * 2,
                is_3d=True,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv3_up = self.conv3_up(conv3)

        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2_up = self.conv2_up(conv2)

        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv = self.conv1_up(conv1)

        return conv3, conv2, conv1, conv


class CouplingBlock(SubModule):
    def __init__(self, cv_chan: int) -> None:
        super(CouplingBlock, self).__init__()

        self.branch_1 = nn.Sequential(
            BasicConv(
                cv_chan,
                cv_chan,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                stride=1,
                dilation=1,
            ),
            nn.Conv3d(cv_chan, cv_chan, kernel_size=1, stride=1, padding=0, bias=False),
        )

        self.branch_2 = nn.Sequential(
            BasicConv(
                cv_chan,
                cv_chan,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                stride=1,
                dilation=1,
            ),
            BasicConv(
                cv_chan,
                cv_chan,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                stride=1,
                dilation=1,
            ),
            nn.Conv3d(cv_chan, cv_chan, kernel_size=1, stride=1, padding=0, bias=False),
        )

        self.weight_init()

    def forward(self, upper: torch.Tensor, lower: torch.Tensor) -> torch.Tensor:

        f1 = self.branch_1(lower)
        f2 = f1 + upper
        out = self.branch_2(f2) + lower

        return out


class hourglass_2(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super(hourglass_2, self).__init__()

        self.conv1 = nn.Sequential(
            BasicConv(
                in_channels,
                in_channels * 2,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=2,
                dilation=1,
            ),
            BasicConv(
                in_channels * 2,
                in_channels * 2,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=1,
                dilation=1,
            ),
        )

        self.conv2 = nn.Sequential(
            BasicConv(
                in_channels * 2,
                in_channels * 4,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=2,
                dilation=1,
            ),
            BasicConv(
                in_channels * 4,
                in_channels * 4,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=1,
                dilation=1,
            ),
        )

        self.conv3 = nn.Sequential(
            BasicConv(
                in_channels * 4,
                in_channels * 6,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=2,
                dilation=1,
            ),
            BasicConv(
                in_channels * 6,
                in_channels * 6,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=1,
                dilation=1,
            ),
        )

        self.conv3_up = BasicConv(
            in_channels * 6,
            in_channels * 4,
            deconv=True,
            is_3d=True,
            bn=True,
            relu=True,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )

        self.conv2_up = BasicConv(
            in_channels * 4,
            in_channels * 2,
            deconv=True,
            is_3d=True,
            bn=True,
            relu=True,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )

        self.conv1_up = BasicConv(
            in_channels * 2,
            1,
            deconv=True,
            is_3d=True,
            bn=False,
            relu=False,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )

        self.agg_0 = nn.Sequential(
            BasicConv(
                in_channels * 8,
                in_channels * 4,
                is_3d=True,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            BasicConv(
                in_channels * 4,
                in_channels * 4,
                is_3d=True,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            BasicConv(
                in_channels * 4,
                in_channels * 4,
                is_3d=True,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )

        self.agg_1 = nn.Sequential(
            BasicConv(
                in_channels * 4,
                in_channels * 2,
                is_3d=True,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            BasicConv(
                in_channels * 2,
                in_channels * 2,
                is_3d=True,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            BasicConv(
                in_channels * 2,
                in_channels * 2,
                is_3d=True,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )

        self.coupling3 = CouplingBlock(in_channels * 6)
        self.coupling2 = CouplingBlock(in_channels * 4)
        self.coupling1 = CouplingBlock(in_channels * 2)

    def forward(
        self,
        c3_upper: torch.Tensor,
        c2_upper: torch.Tensor,
        c1_upper: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv3 = self.coupling3(c3_upper, conv3)
        conv3_up = self.conv3_up(conv3)

        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.coupling2(c2_upper, conv2)
        conv2_up = self.conv2_up(conv2)

        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.coupling1(c1_upper, conv1)
        conv = self.conv1_up(conv1)

        return conv


class DCVSMNet(nn.Module):
    def __init__(
        self,
        maxdisp: int,
        gwc: bool = True,
        norm_correlation: bool = True,
        gwc_substract: bool = False,
        concat: bool = False,
    ) -> None:
        super(DCVSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.concat_channels = 12
        self.feature_extraction = feature_extraction(
            concat_feature_channel=self.concat_channels
        )
        reduction_multiplier = 8

        self.gwc = gwc
        self.norm_correlation = norm_correlation
        self.gwc_substract = gwc_substract
        self.concat = concat

        if self.gwc and self.norm_correlation:
            print("Cost volumes: gwc and norm correlation")
            self.num_groups = 20
            self.group_stem = BasicConv(
                self.num_groups,
                reduction_multiplier,
                is_3d=True,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.agg_group = BasicConv(
                reduction_multiplier,
                reduction_multiplier,
                is_3d=True,
                kernel_size=(1, 5, 5),
                padding=(0, 2, 2),
                stride=1,
            )
            self.hourglass_1 = hourglass_1(reduction_multiplier)

            self.corr_stem = BasicConv(
                1, reduction_multiplier, is_3d=True, kernel_size=3, stride=1, padding=1
            )
            self.agg = BasicConv(
                reduction_multiplier,
                reduction_multiplier,
                is_3d=True,
                kernel_size=(1, 5, 5),
                padding=(0, 2, 2),
                stride=1,
            )
            self.hourglass_2 = hourglass_2(reduction_multiplier)

        if self.gwc_substract and self.norm_correlation:
            print("Cost volumes: gwc substract and norm correlation")
            self.num_groups = 20
            self.v1_stem = BasicConv(
                self.num_groups,
                reduction_multiplier,
                is_3d=True,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.v1_agg = BasicConv(
                reduction_multiplier,
                reduction_multiplier,
                is_3d=True,
                kernel_size=(1, 5, 5),
                padding=(0, 2, 2),
                stride=1,
            )
            self.hourglass_1 = hourglass_1(reduction_multiplier)

            self.v2_stem = BasicConv(
                1, reduction_multiplier, is_3d=True, kernel_size=3, stride=1, padding=1
            )
            self.v2_agg = BasicConv(
                reduction_multiplier,
                reduction_multiplier,
                is_3d=True,
                kernel_size=(1, 5, 5),
                padding=(0, 2, 2),
                stride=1,
            )
            self.hourglass_2 = hourglass_2(reduction_multiplier)

        if self.gwc_substract and self.concat:
            print("Cost volumes: gwc substract and concat")
            self.num_groups = 20
            self.v1_stem = BasicConv(
                self.num_groups,
                reduction_multiplier,
                is_3d=True,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.v1_agg = BasicConv(
                reduction_multiplier,
                reduction_multiplier,
                is_3d=True,
                kernel_size=(1, 5, 5),
                padding=(0, 2, 2),
                stride=1,
            )
            self.hourglass_1 = hourglass_1(reduction_multiplier)

            self.v2_stem = BasicConv(
                24, reduction_multiplier, is_3d=True, kernel_size=3, stride=1, padding=1
            )
            self.v2_agg = BasicConv(
                reduction_multiplier,
                reduction_multiplier,
                is_3d=True,
                kernel_size=(1, 5, 5),
                padding=(0, 2, 2),
                stride=1,
            )
            self.hourglass_2 = hourglass_2(reduction_multiplier)

        if self.gwc and self.gwc_substract:
            print("Cost volumes: gwc and gwc substract")
            self.num_groups = 20
            self.v1_stem = BasicConv(
                self.num_groups,
                reduction_multiplier,
                is_3d=True,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.v1_agg = BasicConv(
                reduction_multiplier,
                reduction_multiplier,
                is_3d=True,
                kernel_size=(1, 5, 5),
                padding=(0, 2, 2),
                stride=1,
            )
            self.hourglass_1 = hourglass_1(reduction_multiplier)

            self.v2_stem = BasicConv(
                self.num_groups,
                reduction_multiplier,
                is_3d=True,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.v2_agg = BasicConv(
                reduction_multiplier,
                reduction_multiplier,
                is_3d=True,
                kernel_size=(1, 5, 5),
                padding=(0, 2, 2),
                stride=1,
            )
            self.hourglass_2 = hourglass_2(reduction_multiplier)

        if self.gwc and self.concat:
            print("Cost volumes: gwc and concat")
            self.num_groups = 20
            self.v1_stem = BasicConv(
                self.num_groups,
                reduction_multiplier,
                is_3d=True,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.v1_agg = BasicConv(
                reduction_multiplier,
                reduction_multiplier,
                is_3d=True,
                kernel_size=(1, 5, 5),
                padding=(0, 2, 2),
                stride=1,
            )
            self.hourglass_1 = hourglass_1(reduction_multiplier)

            self.v2_stem = BasicConv(
                24, reduction_multiplier, is_3d=True, kernel_size=3, stride=1, padding=1
            )
            self.v2_agg = BasicConv(
                reduction_multiplier,
                reduction_multiplier,
                is_3d=True,
                kernel_size=(1, 5, 5),
                padding=(0, 2, 2),
                stride=1,
            )
            self.hourglass_2 = hourglass_2(reduction_multiplier)

        if self.norm_correlation and self.concat:
            print("Cost volumes: norm correlation and concat")
            self.v1_stem = BasicConv(
                1, reduction_multiplier, is_3d=True, kernel_size=3, stride=1, padding=1
            )
            self.v1_agg = BasicConv(
                reduction_multiplier,
                reduction_multiplier,
                is_3d=True,
                kernel_size=(1, 5, 5),
                padding=(0, 2, 2),
                stride=1,
            )
            self.hourglass_1 = hourglass_1(reduction_multiplier)

            self.v2_stem = BasicConv(
                24, reduction_multiplier, is_3d=True, kernel_size=3, stride=1, padding=1
            )
            self.v2_agg = BasicConv(
                reduction_multiplier,
                reduction_multiplier,
                is_3d=True,
                kernel_size=(1, 5, 5),
                padding=(0, 2, 2),
                stride=1,
            )
            self.hourglass_2 = hourglass_2(reduction_multiplier)

        self.cat_agg = BasicConv(
            1, 1, is_3d=True, kernel_size=(1, 5, 5), padding=(0, 2, 2), stride=1
        )

        self.stem_2 = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.stem_4 = nn.Sequential(
            BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )

        self.spx = nn.Sequential(
            nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1),
        )
        self.spx_2 = Conv2x(32, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv(60, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

    def forward(
        self, left: torch.Tensor, right: torch.Tensor, train_status: bool
    ) -> List[torch.Tensor]:

        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)

        if self.gwc and self.norm_correlation:
            gwc_volume = build_gwc_volume(
                features_left["gwc_feature"],
                features_right["gwc_feature"],
                self.maxdisp // 4,
                self.num_groups,
            )
            gwc_volume = self.group_stem(gwc_volume)
            volume_1 = self.agg_group(gwc_volume)

            corr_volume = build_norm_correlation_volume(
                features_left["concat_feature"],
                features_right["concat_feature"],
                self.maxdisp // 4,
            )
            corr_volume = self.corr_stem(corr_volume)
            volume_2 = self.agg(corr_volume)

        if self.gwc and self.gwc_substract:
            gwc_volume = build_gwc_volume(
                features_left["gwc_feature"],
                features_right["gwc_feature"],
                self.maxdisp // 4,
                self.num_groups,
            )
            gwc_volume = self.v1_stem(gwc_volume)
            volume_1 = self.v1_agg(gwc_volume)

            sub_volume = build_substract_volume(
                features_left["gwc_feature"],
                features_right["gwc_feature"],
                self.maxdisp // 4,
                self.num_groups,
            )
            sub_volume = self.v2_stem(sub_volume)
            volume_2 = self.v2_agg(sub_volume)

        if self.gwc_substract and self.norm_correlation:
            sub_volume = build_substract_volume(
                features_left["gwc_feature"],
                features_right["gwc_feature"],
                self.maxdisp // 4,
                self.num_groups,
            )
            sub_volume = self.v1_stem(sub_volume)
            volume_1 = self.v1_agg(sub_volume)

            corr_volume = build_norm_correlation_volume(
                features_left["concat_feature"],
                features_right["concat_feature"],
                self.maxdisp // 4,
            )
            corr_volume = self.v2_stem(corr_volume)
            volume_2 = self.v2_agg(corr_volume)

        if self.gwc_substract and self.concat:
            sub_volume = build_substract_volume(
                features_left["gwc_feature"],
                features_right["gwc_feature"],
                self.maxdisp // 4,
                self.num_groups,
            )
            sub_volume = self.v1_stem(sub_volume)
            volume_1 = self.v1_agg(sub_volume)

            cat_volume = build_concat_volume(
                features_left["concat_feature"],
                features_right["concat_feature"],
                self.maxdisp // 4,
            )
            cat_volume = self.v2_stem(cat_volume)
            volume_2 = self.v2_agg(cat_volume)

        if self.gwc and self.concat:
            gwc_volume = build_gwc_volume(
                features_left["gwc_feature"],
                features_right["gwc_feature"],
                self.maxdisp // 4,
                self.num_groups,
            )
            gwc_volume = self.v1_stem(gwc_volume)
            volume_1 = self.v1_agg(gwc_volume)

            cat_volume = build_concat_volume(
                features_left["concat_feature"],
                features_right["concat_feature"],
                self.maxdisp // 4,
            )
            cat_volume = self.v2_stem(cat_volume)
            volume_2 = self.v2_agg(cat_volume)

        if self.norm_correlation and self.concat:
            corr_volume = build_norm_correlation_volume(
                features_left["concat_feature"],
                features_right["concat_feature"],
                self.maxdisp // 4,
            )
            corr_volume = self.v1_stem(corr_volume)
            volume_1 = self.v1_agg(corr_volume)

            cat_volume = build_concat_volume(
                features_left["concat_feature"],
                features_right["concat_feature"],
                self.maxdisp // 4,
            )
            cat_volume = self.v2_stem(cat_volume)
            volume_2 = self.v2_agg(cat_volume)

        c3, c2, c1, cost_1 = self.hourglass_1(volume_1)
        cost_2 = self.hourglass_2(c3, c2, c1, volume_2)
        add_cost = cost_1 + cost_2

        cost = self.cat_agg(add_cost)
        stem_2x = self.stem_2(left)
        stem_4x = self.stem_4(stem_2x)
        features_left_w = torch.cat((features_left["concat_feature"], stem_4x), 1)

        xspx = self.spx_4(features_left_w)
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)

        disp_samples = torch.arange(
            0, self.maxdisp // 4, dtype=cost.dtype, device=cost.device
        )
        disp_samples = disp_samples.view(1, self.maxdisp // 4, 1, 1).repeat(
            cost.shape[0], 1, cost.shape[3], cost.shape[4]
        )
        pred = regression_topk(cost.squeeze(1), disp_samples, 2)

        pred_up = context_upsample(pred, spx_pred)

        if train_status:
            return [pred_up * 4, pred.squeeze(1) * 4]

        else:
            return [pred_up * 4]
