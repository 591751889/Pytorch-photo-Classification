# -*- coding: utf-8 -*-
"""
ResNet 图像分类模型定义文件。

本文件实现适用于 2D 图像分类任务的 ResNet 网络，默认使用 ResNet18 结构。

输入：
    image: [B, 3, 224, 224]

输出：
    logits: [B, num_classes]

使用示例：
    from models.two_d.resnet import ResNet

    model = ResNet(num_classes=6)
"""

from typing import Optional
from typing import Type

import torch
from torch import nn


class BasicBlock(nn.Module):
    """
    ResNet18 / ResNet34 使用的基础残差块。
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        """
        初始化 BasicBlock。

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 卷积步长
            downsample: 残差分支下采样模块
        """
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    ResNet50 / ResNet101 / ResNet152 使用的瓶颈残差块。
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        """
        初始化 Bottleneck。

        Args:
            in_channels: 输入通道数
            out_channels: 中间通道数
            stride: 卷积步长
            downsample: 残差分支下采样模块
        """
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet 主体网络。

    默认是 ResNet18：
        layers = [2, 2, 2, 2]
        block = BasicBlock

    如果想构建其他结构，建议使用下面的 resnet34 / resnet50 等函数。
    """

    def __init__(
        self,
        num_classes: int,
        block: Type[nn.Module] = BasicBlock,
        layers: Optional[list] = None,
        in_channels: int = 3,
    ):
        """
        初始化 ResNet。

        Args:
            num_classes: 分类类别数
            block: 残差块类型，BasicBlock 或 Bottleneck
            layers: 每个 stage 的残差块数量
            in_channels: 输入图像通道数，RGB 图像为 3
        """
        super(ResNet, self).__init__()

        if layers is None:
            layers = [2, 2, 2, 2]

        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.layer1 = self._make_layer(
            block=block,
            out_channels=64,
            blocks=layers[0],
            stride=1,
        )
        self.layer2 = self._make_layer(
            block=block,
            out_channels=128,
            blocks=layers[1],
            stride=2,
        )
        self.layer3 = self._make_layer(
            block=block,
            out_channels=256,
            blocks=layers[2],
            stride=2,
        )
        self.layer4 = self._make_layer(
            block=block,
            out_channels=512,
            blocks=layers[3],
            stride=2,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            in_features=512 * block.expansion,
            out_features=num_classes,
        )

        self._initialize_weights()

    def _make_layer(
        self,
        block: Type[nn.Module],
        out_channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """
        构建一个 ResNet stage。
        """
        downsample = None

        if stride != 1 or self.in_planes != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_planes,
                    out_channels=out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(
            block(
                in_channels=self.in_planes,
                out_channels=out_channels,
                stride=stride,
                downsample=downsample,
            )
        )

        self.in_planes = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    in_channels=self.in_planes,
                    out_channels=out_channels,
                    stride=1,
                    downsample=None,
                )
            )

        return nn.Sequential(*layers)

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
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(num_classes: int, in_channels: int = 3) -> ResNet:
    """
    构建 ResNet18。
    """
    return ResNet(
        num_classes=num_classes,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        in_channels=in_channels,
    )


def resnet34(num_classes: int, in_channels: int = 3) -> ResNet:
    """
    构建 ResNet34。
    """
    return ResNet(
        num_classes=num_classes,
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        in_channels=in_channels,
    )


def resnet50(num_classes: int, in_channels: int = 3) -> ResNet:
    """
    构建 ResNet50。
    """
    return ResNet(
        num_classes=num_classes,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        in_channels=in_channels,
    )


def resnet101(num_classes: int, in_channels: int = 3) -> ResNet:
    """
    构建 ResNet101。
    """
    return ResNet(
        num_classes=num_classes,
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        in_channels=in_channels,
    )


def resnet152(num_classes: int, in_channels: int = 3) -> ResNet:
    """
    构建 ResNet152。
    """
    return ResNet(
        num_classes=num_classes,
        block=Bottleneck,
        layers=[3, 8, 36, 3],
        in_channels=in_channels,
    )


if __name__ == "__main__":
    model = ResNet(num_classes=6)

    x = torch.randn(2, 3, 224, 224)
    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)