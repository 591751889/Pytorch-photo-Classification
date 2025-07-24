import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np


# 定义基础数据集类
class BaseDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, data_type: str = 'image'):
        """
        初始化数据集
        :param root_dir: 数据根目录
        :param transform: 数据预处理变换
        :param data_type: 数据类型 ('image')
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data_type = data_type
        self.classes = sorted(os.listdir(root_dir))  # 获取类别名称
        self.image_paths = []
        self.labels = []

        # 遍历每个类别文件夹，直接将所有文件路径添加到列表中
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # 只处理图像文件
                    self.image_paths.append(os.path.join(class_dir, file_name))
                    self.labels.append(label)

    def __len__(self):
        """返回数据集大小"""
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """
        获取指定索引的数据
        :param idx: 数据索引
        :return: (image, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 使用 PIL 读取普通图像（2D 图像）
        image = Image.open(img_path).convert('RGB')  # 读取图像文件

        # 调整图像大小到 224x224
        image = image.resize((224, 224))  # 统一调整图像大小为 224x224

        # 转换为 tensor
        image = transforms.ToTensor()(image)  # 转为 tensor 类型并自动归一化到 [0, 1] 范围

        # 如果存在 transform，应用 transform
        if self.transform:
            image = self.transform(image)

        return image, label


# 训练数据类，包含数据增强
class TrainData(BaseDataset):
    def __init__(self, root_dir: str, transform=None, data_type: str = 'image'):
        # 训练数据增强操作
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomRotation(10),  # 随机旋转
            transforms.RandomResizedCrop(224),  # 随机裁剪并调整到 224x224
        ])
        super().__init__(root_dir, transform=train_transform if transform is None else transform, data_type=data_type)


# 验证数据类
class ValData(BaseDataset):
    def __init__(self, root_dir: str, transform=None, data_type: str = 'image'):
        val_transform = transforms.Compose([
            transforms.Resize(256),  # 调整大小为 256
            transforms.CenterCrop(224),  # 裁剪中央区域为 224x224
        ])
        super().__init__(root_dir, transform=val_transform if transform is None else transform, data_type=data_type)


# 测试数据类
class TestData(BaseDataset):
    def __init__(self, root_dir: str, transform=None, data_type: str = 'image'):
        test_transform = transforms.Compose([
            transforms.Resize(256),  # 调整大小为 256
            transforms.CenterCrop(224),  # 裁剪中央区域为 224x224
        ])
        super().__init__(root_dir, transform=test_transform if transform is None else transform, data_type=data_type)


# 打印一个batch的数据
def print_batch(data_loader: DataLoader):
    images, labels = next(iter(data_loader))
    print(f"Batch size: {len(images)}")
    print(f"Images shape: {images.shape}")
    print(f"Labels: {labels}")


# 使用示例
if __name__ == '__main__':
    root_dir = './flower_photos'  # 替换为你的数据集路径
    batch_size = 8
    data_type = 'image'  # 只处理二维图像

    # 直接创建数据集实例
    train_data = TrainData(root_dir=root_dir, data_type=data_type)
    val_data = ValData(root_dir=root_dir, data_type=data_type)
    test_data = TestData(root_dir=root_dir, data_type=data_type)

    # 创建训练、验证和测试数据加载器
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    # 打印训练数据的一个batch
    print_batch(train_loader)
