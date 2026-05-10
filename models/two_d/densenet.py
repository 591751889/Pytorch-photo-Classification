# -*- coding: utf-8 -*-
"""
DenseNet 系列图像分类模型构建文件。

支持：
1. densenet121
2. densenet169

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


def densenet121(num_classes: int, pretrained: bool = False):
    """
    构建 DenseNet121。
    """
    from torchvision.models import DenseNet121_Weights
    from torchvision.models import densenet121 as torchvision_densenet121

    weights = get_weights(DenseNet121_Weights, pretrained)
    model = torchvision_densenet121(weights=weights)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    return model


def densenet169(num_classes: int, pretrained: bool = False):
    """
    构建 DenseNet169。
    """
    from torchvision.models import DenseNet169_Weights
    from torchvision.models import densenet169 as torchvision_densenet169

    weights = get_weights(DenseNet169_Weights, pretrained)
    model = torchvision_densenet169(weights=weights)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    return model