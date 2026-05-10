# -*- coding: utf-8 -*-
"""
ConvNeXt 系列图像分类模型构建文件。

支持：
1. convnext_tiny
2. convnext_small

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


def convnext_tiny(num_classes: int, pretrained: bool = False):
    """
    构建 ConvNeXt-Tiny。
    """
    from torchvision.models import ConvNeXt_Tiny_Weights
    from torchvision.models import convnext_tiny as torchvision_convnext_tiny

    weights = get_weights(ConvNeXt_Tiny_Weights, pretrained)
    model = torchvision_convnext_tiny(weights=weights)

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    return model


def convnext_small(num_classes: int, pretrained: bool = False):
    """
    构建 ConvNeXt-Small。
    """
    from torchvision.models import ConvNeXt_Small_Weights
    from torchvision.models import convnext_small as torchvision_convnext_small

    weights = get_weights(ConvNeXt_Small_Weights, pretrained)
    model = torchvision_convnext_small(weights=weights)

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    return model