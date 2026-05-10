# -*- coding: utf-8 -*-
"""
AlexNet 图像分类模型定义文件。

输入：
    image: [B, 3, 224, 224]

输出：
    logits: [B, num_classes]

使用示例：
    from models.two_d.alexnet import AlexNet

    model = AlexNet(num_classes=6)
"""

import torch
from torch import nn


class AlexNet(nn.Module):
    """
    AlexNet 分类网络。
    """

    def __init__(
        self,
        num_classes: int = 6,
        init_weights: bool = True,
        in_channels: int = 3,
    ):
        """
        初始化 AlexNet。

        Args:
            num_classes: 分类类别数
            init_weights: 是否初始化模型参数
            in_channels: 输入图像通道数，RGB 图像为 3
        """
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=11,
                stride=4,
                padding=2,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
            ),

            nn.Conv2d(
                in_channels=64,
                out_channels=192,
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
            ),

            nn.Conv2d(
                in_channels=192,
                out_channels=384,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=384,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
            ),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
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

            elif isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0,
                    std=0.01,
                )
                nn.init.constant_(module.bias, 0)


def alexnet(
    num_classes: int = 6,
    init_weights: bool = True,
    in_channels: int = 3,
) -> AlexNet:
    """
    构建 AlexNet。
    """
    model = AlexNet(
        num_classes=num_classes,
        init_weights=init_weights,
        in_channels=in_channels,
    )

    return model


if __name__ == "__main__":
    model = AlexNet(num_classes=6)

    x = torch.randn(2, 3, 224, 224)
    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)