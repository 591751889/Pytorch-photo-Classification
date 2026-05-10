# -*- coding: utf-8 -*-
"""
图像分类任务超参数配置文件。
"""


class hparams:
    train_or_test = "train"

    data_root = "/autodl-fs/data/classified_images"
    train_dir = "/autodl-fs/data/classified_images/train"
    val_dir = "/autodl-fs/data/classified_images/valid"
    test_dir = "/autodl-fs/data/classified_images/test"

    # 注意：这里建议写实验总目录，不要写 logs/googleNet
    output_dir = "../logs/classification_experiments"

    latest_checkpoint_file = "checkpoint_latest.pt"
    ckpt = "best_model.pth"

    total_epochs = 40
    batch_size = 8
    num_workers = 4
    init_lr = 0.0002

    scheduler_step_size = 20
    scheduler_gamma = 0.8

    mode = "2d"

    # RGB 图像
    in_class = 3

    # 你的类别目录是 0、1、2、3、4、5，所以是 6 分类
    out_class = 6

    crop_or_pad_size = 224, 224, 1

    fold_arch = "*.png"

    # 一次性跑多个模型
    model_list = [
        # "googlenet",
        "resnet",
        "mobilenet",

        "efficientnet_b0",
        "efficientnet_v2_s",
        "convnext_tiny",
        "densenet121",
        "regnet_y_400mf",
        "swin_t",
    ]

    # 训练完每个模型后，是否自动加载 best_model.pth 在 test 集测试
    test_after_train = True