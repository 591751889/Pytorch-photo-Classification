# -*- coding: utf-8 -*-
"""
图像分类数据集定义文件。

支持如下目录结构：
root_dir/
    0/
        xxx.jpg
        xxx.png
    1/
        xxx.jpg
        xxx.png
    ...

每个类别一个文件夹，文件夹名作为类别名。
"""

import os
from typing import Callable, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def sort_class_names(class_names: List[str]) -> List[str]:
    """
    对类别名排序。

    如果类别名都是数字字符串，例如 0、1、2、10，则按照数字大小排序；
    否则按照普通字符串排序。
    """
    if all(class_name.isdigit() for class_name in class_names):
        return sorted(class_names, key=lambda x: int(x))

    return sorted(class_names)


class BaseDataset(Dataset):
    """
    基础图像分类数据集。
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        data_type: str = "image",
    ):
        """
        初始化数据集。

        Args:
            root_dir: 数据根目录，例如 /autodl-fs/data/classified_images/train
            transform: 图像预处理或增强操作
            data_type: 数据类型，目前只支持 image
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data_type = data_type

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"数据目录不存在: {root_dir}")

        class_names = [
            name for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))
        ]
        self.classes = sort_class_names(class_names)
        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.classes)
        }

        self.image_paths: List[str] = []
        self.labels: List[int] = []

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            label = self.class_to_idx[class_name]

            file_names = sorted(os.listdir(class_dir))
            for file_name in file_names:
                if file_name.lower().endswith(IMAGE_EXTENSIONS):
                    image_path = os.path.join(class_dir, file_name)
                    self.image_paths.append(image_path)
                    self.labels.append(label)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"目录下没有找到图像文件: {root_dir}")

        print(f"[Dataset] root_dir: {root_dir}")
        print(f"[Dataset] classes: {self.classes}")
        print(f"[Dataset] class_to_idx: {self.class_to_idx}")
        print(f"[Dataset] image_count: {len(self.image_paths)}")

    def __len__(self) -> int:
        """
        返回数据集大小。
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        获取一个样本。

        Args:
            idx: 样本索引

        Returns:
            image: 图像张量，shape 为 [3, 224, 224]
            label: 类别标签
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class TrainData(BaseDataset):
    """
    训练集数据类，包含数据增强。
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        data_type: str = "image",
    ):
        if transform is None:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        super().__init__(
            root_dir=root_dir,
            transform=transform,
            data_type=data_type,
        )


class ValData(BaseDataset):
    """
    验证集数据类，不做随机增强。
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        data_type: str = "image",
    ):
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        super().__init__(
            root_dir=root_dir,
            transform=transform,
            data_type=data_type,
        )


class TestData(BaseDataset):
    """
    测试集数据类，不做随机增强。
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        data_type: str = "image",
    ):
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        super().__init__(
            root_dir=root_dir,
            transform=transform,
            data_type=data_type,
        )