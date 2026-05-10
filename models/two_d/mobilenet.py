# -*- coding: utf-8 -*-
"""
MobileNetV2 图像分类模型定义文件。

本文件实现适用于 2D 图像分类任务的 MobileNetV2。

输入：
    image: [B, 3, 224, 224]

输出：
    logits: [B, num_classes]

使用示例：
    from models.two_d.mobilenet import MobileNetV2

    model = MobileNetV2(num_classes=6)
"""

from typing import List

import torch
from torch import nn


class ConvBNReLU(nn.Sequential):
    """
    Conv2d + BatchNorm2d + ReLU6 基础模块。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ):
        padding = (kernel_size - 1) // 2

        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    """
    MobileNetV2 倒残差模块。

    结构：
        1x1 pointwise conv 升维
        3x3 depthwise conv
        1x1 pointwise conv 降维

    当 stride=1 且输入输出通道一致时，使用残差连接。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
    ):
        super(InvertedResidual, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("stride 只能是 1 或 2，当前为: %d" % stride)

        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []

        if expand_ratio != 1:
            layers.append(
                ConvBNReLU(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    stride=1,
                )
            )

        layers.extend([
            ConvBNReLU(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=stride,
                groups=hidden_dim,
            ),
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        """
        if self.use_res_connect:
            return x + self.conv(x)

        return self.conv(x)


class MobileNetV2(nn.Module):
    """
    MobileNetV2 分类网络。
    """

    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 3,
        width_mult: float = 1.0,
        dropout: float = 0.2,
        init_weights: bool = True,
    ):
        """
        初始化 MobileNetV2。

        Args:
            num_classes: 分类类别数
            in_channels: 输入图像通道数，RGB 图像为 3
            width_mult: 通道宽度倍率，默认 1.0
            dropout: 分类头 dropout 概率
            init_weights: 是否初始化模型参数
        """
        super(MobileNetV2, self).__init__()

        block = InvertedResidual

        input_channel = int(32 * width_mult)
        last_channel = int(1280 * max(1.0, width_mult))

        # t: expansion ratio
        # c: output channels
        # n: block repeat number
        # s: stride
        inverted_residual_setting: List[List[int]] = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = [
            ConvBNReLU(
                in_channels=in_channels,
                out_channels=input_channel,
                kernel_size=3,
                stride=2,
            )
        ]

        for expand_ratio, output_channel, repeat_num, stride in inverted_residual_setting:
            output_channel = int(output_channel * width_mult)

            for block_idx in range(repeat_num):
                current_stride = stride if block_idx == 0 else 1

                features.append(
                    block(
                        in_channels=input_channel,
                        out_channels=output_channel,
                        stride=current_stride,
                        expand_ratio=expand_ratio,
                    )
                )

                input_channel = output_channel

        features.append(
            ConvBNReLU(
                in_channels=input_channel,
                out_channels=last_channel,
                kernel_size=1,
                stride=1,
            )
        )

        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_channel, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)

        return x

    def _initialize_weights(self) -> None:
        """
        初始化模型参数。
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0,
                    std=0.01,
                )
                nn.init.constant_(module.bias, 0)


def mobilenet_v2(
    num_classes: int = 6,
    in_channels: int = 3,
    width_mult: float = 1.0,
    dropout: float = 0.2,
    init_weights: bool = True,
) -> MobileNetV2:
    """
    构建 MobileNetV2。
    """
    model = MobileNetV2(
        num_classes=num_classes,
        in_channels=in_channels,
        width_mult=width_mult,
        dropout=dropout,
        init_weights=init_weights,
    )

    return model


def mobilenet(
    num_classes: int = 6,
    in_channels: int = 3,
    width_mult: float = 1.0,
    dropout: float = 0.2,
    init_weights: bool = True,
) -> MobileNetV2:
    """
    默认构建 MobileNetV2。

    这个函数主要是为了让训练脚本里可以直接写 model_name == "mobilenet"。
    """
    return mobilenet_v2(
        num_classes=num_classes,
        in_channels=in_channels,
        width_mult=width_mult,
        dropout=dropout,
        init_weights=init_weights,
    )


if __name__ == "__main__":
    model = MobileNetV2(num_classes=6)

    x = torch.randn(2, 3, 224, 224)
    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)