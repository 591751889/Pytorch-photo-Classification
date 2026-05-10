# -*- coding: utf-8 -*-
"""
图像分类多模型批量训练、验证、测试脚本。

功能：
1. 支持 train / valid / test 三个独立数据目录；
2. 支持一次性训练多个模型；
3. 每个 epoch 都计算并打印 train / valid 分类指标；
4. 每个模型单独保存 epoch_metrics.csv；
5. 每个模型单独保存 valid accuracy 最高的 best_metrics.csv；
6. 每个模型单独保存 best_model.pth；
7. 训练完每个模型后，可自动在 test 集上测试；
8. 最后生成 all_model_results.csv，用于横向比较不同模型效果。
"""

import argparse
import csv
import os
from typing import Dict
from typing import List
from typing import Tuple

import torch
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_function import TestData
from data_function import TrainData
from data_function import ValData
from hparam import hparams as hp


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    解析命令行参数。
    """
    parser.add_argument(
        "--output-root",
        type=str,
        default=hp.output_dir,
        required=False,
        help="保存所有模型实验结果的根目录",
    )
    parser.add_argument(
        "--latest-checkpoint-file",
        type=str,
        default=hp.latest_checkpoint_file,
        help="每个 epoch 保存的最新检查点文件名",
    )
    parser.add_argument(
        "--best-model-weights",
        type=str,
        default=hp.ckpt,
        help="最佳模型权重文件名",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=hp.train_dir,
        help="训练集目录",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default=hp.val_dir,
        help="验证集目录",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=hp.test_dir,
        help="测试集目录",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=hp.total_epochs,
        help="训练总轮数",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=hp.batch_size,
        help="每个 batch 的样本数",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=hp.num_workers,
        help="DataLoader 使用的 worker 数量",
    )
    parser.add_argument(
        "--init-lr",
        type=float,
        default=hp.init_lr,
        help="初始学习率",
    )
    parser.add_argument(
        "--epoch-metrics-file",
        type=str,
        default="epoch_metrics.csv",
        help="保存每个 epoch 指标的 CSV 文件名",
    )
    parser.add_argument(
        "--best-metrics-file",
        type=str,
        default="best_metrics.csv",
        help="保存最佳 epoch 指标的 CSV 文件名",
    )
    parser.add_argument(
        "--test-metrics-file",
        type=str,
        default="test_metrics.csv",
        help="保存测试集指标的 CSV 文件名",
    )
    parser.add_argument(
        "--all-model-results-file",
        type=str,
        default="all_model_results.csv",
        help="保存所有模型最佳结果汇总的 CSV 文件名",
    )
    parser.add_argument(
        "--cudnn-enabled",
        default=True,
        type=bool,
        help="是否启用 cudnn",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        default=True,
        type=bool,
        help="是否启用 cudnn benchmark",
    )

    return parser


def build_model(model_name: str) -> nn.Module:
    """
    根据模型名称构建模型。

    注意：
    这里的 import 路径需要和你自己的工程目录一致。
    如果你的模型文件路径不同，只需要改这个函数即可。
    """
    model_name = model_name.lower()

    if hp.mode != "2d":
        raise ValueError("当前分类脚本只支持 2d 模式，不支持: %s" % hp.mode)

    if model_name == "googlenet":
        from models.two_d.googlenet import googlenet

        model = googlenet(num_class=hp.out_class)

    elif model_name == "resnet":
        from models.two_d.resnet import ResNet

        model = ResNet(num_classes=hp.out_class)

    elif model_name == "alexnet":
        from models.two_d.alexnet import AlexNet

        model = AlexNet(num_classes=hp.out_class)

    elif model_name == "vgg":
        from models.two_d.vgg import vgg

        model = vgg(
            model_name="vgg16",
            num_classes=hp.out_class,
            init_weights=True,
        )

    elif model_name == "vgg11":
        from models.two_d.vgg import vgg

        model = vgg(
            model_name="vgg11",
            num_classes=hp.out_class,
            init_weights=True,
        )

    elif model_name == "vgg13":
        from models.two_d.vgg import vgg

        model = vgg(
            model_name="vgg13",
            num_classes=hp.out_class,
            init_weights=True,
        )

    elif model_name == "vgg16":
        from models.two_d.vgg import vgg

        model = vgg(
            model_name="vgg16",
            num_classes=hp.out_class,
            init_weights=True,
        )

    elif model_name == "vgg19":
        from models.two_d.vgg import vgg

        model = vgg(
            model_name="vgg19",
            num_classes=hp.out_class,
            init_weights=True,
        )

    else:
        raise ValueError("暂不支持的模型名称: %s" % model_name)

    return model

def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    构建 train / valid / test 三个 DataLoader。
    """
    train_dataset = TrainData(root_dir=args.train_dir, data_type="image")
    val_dataset = ValData(root_dir=args.val_dir, data_type="image")
    test_dataset = TestData(root_dir=args.test_dir, data_type="image")

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader

def get_logits(outputs):
    """
    兼容部分模型返回 tuple / list 的情况。

    普通模型：
        outputs = model(x)

    某些 GoogLeNet 结构可能返回：
        outputs = (logits, aux1, aux2)
    """
    if isinstance(outputs, (tuple, list)):
        return outputs[0]

    return outputs


def build_model(model_name: str) -> nn.Module:
    """
    根据模型名称构建分类模型。
    """
    model_name = model_name.lower()

    if hp.mode != "2d":
        raise ValueError("当前分类脚本只支持 2d 模式，不支持: %s" % hp.mode)

    if model_name == "googlenet":
        from models.two_d.googlenet import googlenet

        model = googlenet(num_class=hp.out_class)

    elif model_name == "resnet":
        from models.two_d.resnet import ResNet

        model = ResNet(num_classes=hp.out_class)

    elif model_name == "alexnet":
        from models.two_d.alexnet import AlexNet

        model = AlexNet(num_classes=hp.out_class)

    elif model_name == "vgg16":
        from models.two_d.vgg import vgg

        model = vgg(
            model_name="vgg16",
            num_classes=hp.out_class,
            init_weights=True,
        )

    elif model_name == "mobilenet":
        from models.two_d.mobilenet import mobilenet

        model = mobilenet(
            num_classes=hp.out_class,
            in_channels=hp.in_class,
        )

    elif model_name == "efficientnet_b0":
        from models.two_d.efficientnet import efficientnet_b0

        model = efficientnet_b0(
            num_classes=hp.out_class,
            pretrained=False,
        )

    elif model_name == "efficientnet_v2_s":
        from models.two_d.efficientnet import efficientnet_v2_s

        model = efficientnet_v2_s(
            num_classes=hp.out_class,
            pretrained=False,
        )

    elif model_name == "convnext_tiny":
        from models.two_d.convnext import convnext_tiny

        model = convnext_tiny(
            num_classes=hp.out_class,
            pretrained=False,
        )

    elif model_name == "convnext_small":
        from models.two_d.convnext import convnext_small

        model = convnext_small(
            num_classes=hp.out_class,
            pretrained=False,
        )

    elif model_name == "densenet121":
        from models.two_d.densenet import densenet121

        model = densenet121(
            num_classes=hp.out_class,
            pretrained=False,
        )

    elif model_name == "densenet169":
        from models.two_d.densenet import densenet169

        model = densenet169(
            num_classes=hp.out_class,
            pretrained=False,
        )

    elif model_name == "regnet_y_400mf":
        from models.two_d.regnet import regnet_y_400mf

        model = regnet_y_400mf(
            num_classes=hp.out_class,
            pretrained=False,
        )

    elif model_name == "regnet_y_800mf":
        from models.two_d.regnet import regnet_y_800mf

        model = regnet_y_800mf(
            num_classes=hp.out_class,
            pretrained=False,
        )

    elif model_name == "swin_t":
        from models.two_d.swin_transformer import swin_t

        model = swin_t(
            num_classes=hp.out_class,
            pretrained=False,
        )

    elif model_name == "swin_s":
        from models.two_d.swin_transformer import swin_s

        model = swin_s(
            num_classes=hp.out_class,
            pretrained=False,
        )

    else:
        raise ValueError("暂不支持的模型名称: %s" % model_name)

    return model


def compute_classification_metrics(
    all_labels: List[int],
    all_preds: List[int],
) -> Tuple[float, float, float]:
    """
    计算分类任务的 Precision、Recall、F1。
    """
    precision = precision_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )
    recall = recall_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )
    f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    return precision, recall, f1


def save_csv_rows(csv_path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    """
    追加保存多行指标到 CSV。
    """
    dir_name = os.path.dirname(csv_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for row in rows:
            writer.writerow(row)


def save_single_csv_row(csv_path: str, row: Dict, fieldnames: List[str]) -> None:
    """
    覆盖保存单行指标到 CSV。
    """
    dir_name = os.path.dirname(csv_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(csv_path, mode="w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def get_epoch_metric_fieldnames() -> List[str]:
    """
    每个模型的 epoch_metrics.csv / best_metrics.csv 字段。
    """
    return [
        "model_name",
        "epoch",
        "best_model_path",
        "train_loss",
        "train_acc",
        "train_precision",
        "train_recall",
        "train_f1",
        "val_loss",
        "val_acc",
        "val_precision",
        "val_recall",
        "val_f1",
        "class_num",
        "batch_size",
        "init_lr",
        "train_dir",
        "val_dir",
        "output_dir",
    ]


def get_test_metric_fieldnames() -> List[str]:
    """
    单个模型 test_metrics.csv 字段。
    """
    return [
        "model_name",
        "test_loss",
        "test_acc",
        "test_precision",
        "test_recall",
        "test_f1",
        "class_num",
        "batch_size",
        "init_lr",
        "test_dir",
        "output_dir",
        "best_model_path",
    ]


def get_all_model_result_fieldnames() -> List[str]:
    """
    所有模型汇总 all_model_results.csv 字段。
    """
    return [
        "model_name",
        "best_epoch",
        "best_model_path",
        "best_train_loss",
        "best_train_acc",
        "best_train_precision",
        "best_train_recall",
        "best_train_f1",
        "best_val_loss",
        "best_val_acc",
        "best_val_precision",
        "best_val_recall",
        "best_val_f1",
        "test_loss",
        "test_acc",
        "test_precision",
        "test_recall",
        "test_f1",
        "class_num",
        "batch_size",
        "init_lr",
        "output_dir",
    ]


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epoch: int,
    num_epochs: int,
    model_name: str,
) -> Tuple[float, float, float, float, float]:
    """
    训练一个 epoch，并返回分类指标。
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for step, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(x)
        outputs = get_logits(outputs)

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, dim=1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

        all_preds.extend(predicted.detach().cpu().numpy().tolist())
        all_labels.extend(y.detach().cpu().numpy().tolist())

        if (step + 1) % 10 == 0:
            print(
                "[%s] Epoch [%d/%d], Step [%d/%d], Loss: %.4f"
                % (
                    model_name,
                    epoch + 1,
                    num_epochs,
                    step + 1,
                    len(train_loader),
                    loss.item(),
                )
            )

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total * 100.0
    precision, recall, f1 = compute_classification_metrics(all_labels, all_preds)

    return epoch_loss, epoch_acc, precision, recall, f1


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    phase: str,
    model_name: str,
) -> Tuple[float, float, float, float, float]:
    """
    在验证集或测试集上评估模型。
    """
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            outputs = get_logits(outputs)

            loss = criterion(outputs, y)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            all_preds.extend(predicted.detach().cpu().numpy().tolist())
            all_labels.extend(y.detach().cpu().numpy().tolist())

    avg_loss = running_loss / len(data_loader)
    acc = correct / total * 100.0
    precision, recall, f1 = compute_classification_metrics(all_labels, all_preds)

    print(
        "[%s] %s Loss: %.4f, %s Acc: %.2f%%, Precision: %.4f, Recall: %.4f, F1: %.4f"
        % (
            model_name,
            phase,
            avg_loss,
            phase,
            acc,
            precision,
            recall,
            f1,
        )
    )

    return avg_loss, acc, precision, recall, f1


def train_single_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    args: argparse.Namespace,
) -> Dict:
    """
    训练单个模型，并返回该模型的最佳验证指标和测试指标。
    """
    print("")
    print("============================================================")
    print("Start training model: %s" % model_name)
    print("============================================================")

    model_output_dir = os.path.join(args.output_root, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    best_model_path = os.path.join(model_output_dir, args.best_model_weights)
    latest_model_path = os.path.join(model_output_dir, args.latest_checkpoint_file)
    epoch_metrics_path = os.path.join(model_output_dir, args.epoch_metrics_file)
    best_metrics_path = os.path.join(model_output_dir, args.best_metrics_file)
    test_metrics_path = os.path.join(model_output_dir, args.test_metrics_file)

    if os.path.exists(epoch_metrics_path):
        os.remove(epoch_metrics_path)

    model = build_model(model_name)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=args.init_lr)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=model_output_dir)

    best_acc = -1.0
    best_record = None
    metric_fieldnames = get_epoch_metric_fieldnames()

    for epoch in range(args.epochs):
        train_loss, train_acc, train_precision, train_recall, train_f1 = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
            num_epochs=args.epochs,
            model_name=model_name,
        )

        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            phase="Validation",
            model_name=model_name,
        )

        print(
            "[%s] Epoch [%d/%d], Summary -> "
            "Train Loss: %.4f, Train Acc: %.2f%%, Train Precision: %.4f, Train Recall: %.4f, Train F1: %.4f, "
            "Val Loss: %.4f, Val Acc: %.2f%%, Val Precision: %.4f, Val Recall: %.4f, Val F1: %.4f"
            % (
                model_name,
                epoch + 1,
                args.epochs,
                train_loss,
                train_acc,
                train_precision,
                train_recall,
                train_f1,
                val_loss,
                val_acc,
                val_precision,
                val_recall,
                val_f1,
            )
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Precision/train", train_precision, epoch)
        writer.add_scalar("Recall/train", train_recall, epoch)
        writer.add_scalar("F1/train", train_f1, epoch)

        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("Precision/val", val_precision, epoch)
        writer.add_scalar("Recall/val", val_recall, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)

        metric_record = {
            "model_name": model_name,
            "epoch": epoch + 1,
            "best_model_path": best_model_path,
            "train_loss": "%.6f" % train_loss,
            "train_acc": "%.6f" % train_acc,
            "train_precision": "%.6f" % train_precision,
            "train_recall": "%.6f" % train_recall,
            "train_f1": "%.6f" % train_f1,
            "val_loss": "%.6f" % val_loss,
            "val_acc": "%.6f" % val_acc,
            "val_precision": "%.6f" % val_precision,
            "val_recall": "%.6f" % val_recall,
            "val_f1": "%.6f" % val_f1,
            "class_num": hp.out_class,
            "batch_size": args.batch,
            "init_lr": args.init_lr,
            "train_dir": args.train_dir,
            "val_dir": args.val_dir,
            "output_dir": model_output_dir,
        }

        save_csv_rows(
            csv_path=epoch_metrics_path,
            rows=[metric_record],
            fieldnames=metric_fieldnames,
        )

        torch.save(model.state_dict(), latest_model_path)
        print("[%s] Saved latest checkpoint: %s" % (model_name, latest_model_path))

        if val_acc > best_acc:
            best_acc = val_acc
            best_record = metric_record

            torch.save(model.state_dict(), best_model_path)

            save_single_csv_row(
                csv_path=best_metrics_path,
                row=best_record,
                fieldnames=metric_fieldnames,
            )

            print("[%s] Saved best model: %s" % (model_name, best_model_path))
            print("[%s] Saved best metrics: %s" % (model_name, best_metrics_path))
            print("[%s] Current best val acc: %.2f%%" % (model_name, best_acc))

    writer.close()

    test_loss = ""
    test_acc = ""
    test_precision = ""
    test_recall = ""
    test_f1 = ""

    if hp.test_after_train:
        print("[%s] Load best model and test on test set." % model_name)
        state_dict = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state_dict)

        test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            phase="Test",
            model_name=model_name,
        )

        test_row = {
            "model_name": model_name,
            "test_loss": "%.6f" % test_loss,
            "test_acc": "%.6f" % test_acc,
            "test_precision": "%.6f" % test_precision,
            "test_recall": "%.6f" % test_recall,
            "test_f1": "%.6f" % test_f1,
            "class_num": hp.out_class,
            "batch_size": args.batch,
            "init_lr": args.init_lr,
            "test_dir": args.test_dir,
            "output_dir": model_output_dir,
            "best_model_path": best_model_path,
        }

        save_single_csv_row(
            csv_path=test_metrics_path,
            row=test_row,
            fieldnames=get_test_metric_fieldnames(),
        )

        print("[%s] Saved test metrics: %s" % (model_name, test_metrics_path))

    if best_record is None:
        raise RuntimeError("模型 %s 没有得到任何 best_record，请检查训练集或验证集。" % model_name)

    all_model_result = {
        "model_name": model_name,
        "best_epoch": best_record["epoch"],
        "best_model_path": best_model_path,
        "best_train_loss": best_record["train_loss"],
        "best_train_acc": best_record["train_acc"],
        "best_train_precision": best_record["train_precision"],
        "best_train_recall": best_record["train_recall"],
        "best_train_f1": best_record["train_f1"],
        "best_val_loss": best_record["val_loss"],
        "best_val_acc": best_record["val_acc"],
        "best_val_precision": best_record["val_precision"],
        "best_val_recall": best_record["val_recall"],
        "best_val_f1": best_record["val_f1"],
        "test_loss": "%.6f" % test_loss if test_loss != "" else "",
        "test_acc": "%.6f" % test_acc if test_acc != "" else "",
        "test_precision": "%.6f" % test_precision if test_precision != "" else "",
        "test_recall": "%.6f" % test_recall if test_recall != "" else "",
        "test_f1": "%.6f" % test_f1 if test_f1 != "" else "",
        "class_num": hp.out_class,
        "batch_size": args.batch,
        "init_lr": args.init_lr,
        "output_dir": model_output_dir,
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("============================================================")
    print("Finish model: %s" % model_name)
    print("Best val acc: %s" % all_model_result["best_val_acc"])
    print("Test acc: %s" % all_model_result["test_acc"])
    print("============================================================")

    return all_model_result


def test_single_model(
    model_name: str,
    test_loader: DataLoader,
    args: argparse.Namespace,
) -> Dict:
    """
    只测试单个已经训练好的模型。
    """
    print("")
    print("============================================================")
    print("Start testing model: %s" % model_name)
    print("============================================================")

    model_output_dir = os.path.join(args.output_root, model_name)
    best_model_path = os.path.join(model_output_dir, args.best_model_weights)
    test_metrics_path = os.path.join(model_output_dir, args.test_metrics_file)

    if not os.path.exists(best_model_path):
        raise FileNotFoundError("没有找到模型权重文件: %s" % best_model_path)

    model = build_model(model_name)
    model = model.to(device)

    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        phase="Test",
        model_name=model_name,
    )

    test_row = {
        "model_name": model_name,
        "test_loss": "%.6f" % test_loss,
        "test_acc": "%.6f" % test_acc,
        "test_precision": "%.6f" % test_precision,
        "test_recall": "%.6f" % test_recall,
        "test_f1": "%.6f" % test_f1,
        "class_num": hp.out_class,
        "batch_size": args.batch,
        "init_lr": args.init_lr,
        "test_dir": args.test_dir,
        "output_dir": model_output_dir,
        "best_model_path": best_model_path,
    }

    save_single_csv_row(
        csv_path=test_metrics_path,
        row=test_row,
        fieldnames=get_test_metric_fieldnames(),
    )

    result = {
        "model_name": model_name,
        "best_epoch": "",
        "best_model_path": best_model_path,
        "best_train_loss": "",
        "best_train_acc": "",
        "best_train_precision": "",
        "best_train_recall": "",
        "best_train_f1": "",
        "best_val_loss": "",
        "best_val_acc": "",
        "best_val_precision": "",
        "best_val_recall": "",
        "best_val_f1": "",
        "test_loss": "%.6f" % test_loss,
        "test_acc": "%.6f" % test_acc,
        "test_precision": "%.6f" % test_precision,
        "test_recall": "%.6f" % test_recall,
        "test_f1": "%.6f" % test_f1,
        "class_num": hp.out_class,
        "batch_size": args.batch,
        "init_lr": args.init_lr,
        "output_dir": model_output_dir,
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Finish testing model: %s" % model_name)

    return result


def main() -> None:
    """
    主函数。
    """
    parser = argparse.ArgumentParser(description="PyTorch 图像分类多模型批量训练 / 测试")
    parser = parse_training_args(parser)
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    os.makedirs(args.output_root, exist_ok=True)

    print("Device: %s" % device)
    print("Train dir: %s" % args.train_dir)
    print("Val dir: %s" % args.val_dir)
    print("Test dir: %s" % args.test_dir)
    print("Class num: %d" % hp.out_class)
    print("Output root: %s" % args.output_root)
    print("Model list: %s" % hp.model_list)

    train_loader, val_loader, test_loader = build_dataloaders(args)

    all_model_results_path = os.path.join(
        args.output_root,
        args.all_model_results_file,
    )

    if os.path.exists(all_model_results_path):
        os.remove(all_model_results_path)

    all_results = []

    if hp.train_or_test == "train":
        for model_name in hp.model_list:
            result = train_single_model(
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                args=args,
            )
            all_results.append(result)

            save_csv_rows(
                csv_path=all_model_results_path,
                rows=[result],
                fieldnames=get_all_model_result_fieldnames(),
            )

    elif hp.train_or_test == "test":
        for model_name in hp.model_list:
            result = test_single_model(
                model_name=model_name,
                test_loader=test_loader,
                args=args,
            )
            all_results.append(result)

            save_csv_rows(
                csv_path=all_model_results_path,
                rows=[result],
                fieldnames=get_all_model_result_fieldnames(),
            )

    else:
        raise ValueError(
            "hp.train_or_test 只能是 'train' 或 'test'，当前是: %s"
            % hp.train_or_test
        )

    print("")
    print("============================================================")
    print("All model results saved to: %s" % all_model_results_path)
    print("============================================================")

    for result in all_results:
        print(
            "Model: %s, Best Val Acc: %s, Test Acc: %s"
            % (
                result["model_name"],
                result["best_val_acc"],
                result["test_acc"],
            )
        )


if __name__ == "__main__":
    main()