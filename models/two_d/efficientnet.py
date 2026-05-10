# -*- coding: utf-8 -*-
"""
EfficientNet 系列图像分类模型构建文件。

支持：
1. efficientnet_b0
2. efficientnet_v2_s

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


def efficientnet_b0(num_classes: int, pretrained: bool = False):
    """
    构建 EfficientNet-B0。
    """
    from torchvision.models import EfficientNet_B0_Weights
    from torchvision.models import efficientnet_b0 as torchvision_efficientnet_b0

    weights = get_weights(EfficientNet_B0_Weights, pretrained)
    model = torchvision_efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def efficientnet_v2_s(num_classes: int, pretrained: bool = False):
    """
    构建 EfficientNetV2-S。
    """
    from torchvision.models import EfficientNet_V2_S_Weights
    from torchvision.models import efficientnet_v2_s as torchvision_efficientnet_v2_s

    weights = get_weights(EfficientNet_V2_S_Weights, pretrained)
    model = torchvision_efficientnet_v2_s(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model