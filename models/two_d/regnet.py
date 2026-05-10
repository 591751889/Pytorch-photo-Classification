# -*- coding: utf-8 -*-
"""
RegNet 系列图像分类模型构建文件。

支持：
1. regnet_y_400mf
2. regnet_y_800mf

输入：
    image: [B, 3, 224, 224]

输出：
    logits: [B, num_classes]
"""

from torch import nn


def get_weights(weights_class, pretrained: bool):
    """
    根据 pretrained 参数获取 torchvision 预训练权重。
    """
    if pretrained:
        return weights_class.DEFAULT

    return None


def regnet_y_400mf(num_classes: int, pretrained: bool = False):
    """
    构建 RegNet-Y-400MF。
    """
    from torchvision.models import RegNet_Y_400MF_Weights
    from torchvision.models import regnet_y_400mf as torchvision_regnet_y_400mf

    weights = get_weights(RegNet_Y_400MF_Weights, pretrained)
    model = torchvision_regnet_y_400mf(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def regnet_y_800mf(num_classes: int, pretrained: bool = False):
    """
    构建 RegNet-Y-800MF。
    """
    from torchvision.models import RegNet_Y_800MF_Weights
    from torchvision.models import regnet_y_800mf as torchvision_regnet_y_800mf

    weights = get_weights(RegNet_Y_800MF_Weights, pretrained)
    model = torchvision_regnet_y_800mf(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model