import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from data_function import TrainData, ValData
from hparam import hparams as hp
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_training_args(parser):
    """
    解析命令行参数。
    """
    parser.add_argument('-o', '--output_dir', type=str, default=hp.output_dir, required=False, help='保存检查点的目录')
    parser.add_argument('--latest-checkpoint-file', type=str, default=hp.latest_checkpoint_file,
                        help='每个epoch保存最新的检查点文件')

    # 训练配置
    training = parser.add_argument_group('训练设置')
    training.add_argument('--epochs', type=int, default=hp.total_epochs, help='训练总轮次')

    training.add_argument('--batch', type=int, default=hp.batch_size, help='每批次的样本数')
    parser.add_argument(
        '-k',
        "--ckpt",
        type=str,
        default=hp.ckpt,
        help="恢复训练的检查点路径",
    )
    parser.add_argument("--init-lr", type=float, default=hp.init_lr, help="初始学习率")
    parser.add_argument("--local_rank", type=int, default=0, help="分布式训练中的本地rank")
    training.add_argument('--amp-run', action='store_true', help='启用自动混合精度(AMP)')
    training.add_argument('--cudnn-enabled', default=True, help='启用cudnn')
    training.add_argument('--cudnn-benchmark', default=True, help='启用cudnn的优化算法')
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true',
                          help='禁用BatchNorm层权重的均匀初始化')

    parser.add_argument(
        '--best-model-weights',
        type=str,
        default=hp.ckpt,
        help='最佳模型权重路径'
    )

    return parser


def train(model, train_loader, val_loader, optimizer, criterion, num_epochs=10, output_dir="./output"):
    """
    训练模型并在每个epoch结束后进行验证。
    参数:
        model: 需要训练的模型。
        train_loader: 用于训练数据的DataLoader。
        val_loader: 用于验证数据的DataLoader。
        optimizer: 模型的优化器。
        criterion: 损失函数。
        num_epochs: 训练的总轮数。
        output_dir: 保存模型和日志的目录
    """
    writer = SummaryWriter(log_dir=output_dir)  # 设置TensorBoard日志路径

    best_acc = 0.0  # 用于保存最佳准确率
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0  # 初始化累计损失
        correct = 0  # 正确分类的数量
        total = 0  # 总样本数量
        all_preds = []  # 用于存储所有预测结果
        all_labels = []  # 用于存储所有真实标签

        # 训练过程
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)  # 将输入和标签转移到指定设备

            optimizer.zero_grad()  # 清除上一轮的梯度
            outputs = model(x)  # 前向传播

            loss = criterion(outputs, y)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数

            running_loss += loss.item()  # 累加损失

            # 计算准确率
            _, predicted = torch.max(outputs, 1)  # 获取最大值的索引作为预测结果
            total += y.size(0)  # 累计样本数
            correct += (predicted == y).sum().item()  # 累计正确的预测数

            all_preds.extend(predicted.cpu().numpy())  # 保存预测值
            all_labels.extend(y.cpu().numpy())  # 保存真实标签

            if (i + 1) % 10 == 0:  # 每10个批次输出一次损失值
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # 计算每个epoch的平均损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total * 100

        # 计算更多分类指标
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

        # 将指标写入TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        writer.add_scalar('Precision/train', precision, epoch)
        writer.add_scalar('Recall/train', recall, epoch)
        writer.add_scalar('F1/train', f1, epoch)

        # 在每个epoch结束后进行验证
        val_loss, val_acc, val_precision, val_recall, val_f1 = validate(model, val_loader, criterion)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"Saved best model with accuracy: {best_acc:.2f}%")

        # 将验证结果写入TensorBoard
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Precision/val', val_precision, epoch)
        writer.add_scalar('Recall/val', val_recall, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)


def validate(model, val_loader, criterion):
    """
    在验证集上验证模型，并计算更多的分类指标。
    参数:
        model: 需要验证的模型。
        val_loader: 用于验证数据的DataLoader。
        criterion: 损失函数。
    """
    model.eval()  # 设置模型为评估模式
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 在验证过程中不计算梯度
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)  # 将输入和标签转移到指定设备

            outputs = model(x)  # 前向传播
            loss = criterion(outputs, y)  # 计算损失

            val_loss += loss.item()  # 累加验证损失

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total * 100

    # 计算更多分类指标
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(
        f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

    return avg_val_loss, val_acc, precision, recall, f1


def load_best_model(model, model_path, device):
    """
    加载最佳模型权重。
    """
    if os.path.exists(model_path):
        print(f"加载最佳模型权重：{model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("加载成功！")
    else:
        print("没有找到最佳模型权重，跳过加载。")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='PyTorch 医学分割训练')
    parser = parse_training_args(parser)
    args = parser.parse_args()

    # 设置cudnn相关选项
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 根据模式选择合适的模型
    if hp.mode == '2d':
        from models.two_d.googlenet import googlenet
        model = googlenet(num_class=5)  # 选择GoogLeNet模型

    # 设定优化器
    optimizer = Adam(model.parameters(), lr=args.init_lr)

    # 将模型移至GPU
    model.cuda()

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 数据集路径
    root_dir = './flower_photos'  # 替换为你的数据集路径

    data_type = 'image'  # 只处理二维图像

    # 加载训练数据集
    train_dataset = TrainData(root_dir=root_dir, data_type=data_type)
    val_dataset = ValData(root_dir=root_dir, data_type=data_type)  # 假设你有一个验证集的数据加载类
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

    if hp.train_or_test == 'train':  # 判断是否进行训练
        # 调用训练函数
        train(model, train_loader, val_loader, optimizer, criterion, num_epochs=args.epochs, output_dir=args.output_dir)

    elif hp.train_or_test == 'test':  # 判断是否进行测试
        # 加载最佳模型并进行测试
        load_best_model(model, os.path.join(args.output_dir,args.best_model_weights),device)
        # 测试代码
        test_loader = val_loader  # 假设测试和验证使用相同的数据
        accuracy = validate(model, test_loader, criterion)
        print(f"Test Accuracy: {accuracy[1]:.2f}%")


if __name__ == '__main__':
    main()
