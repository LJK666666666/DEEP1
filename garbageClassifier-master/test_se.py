'''
测试 ResNet 模型在指定数据集上的准确率
'''
import os
import argparse
from collections import defaultdict
import torch
import torch.nn as nn
import config
from nets.NetsTorch import NetsTorch
from utils.datasets import ImageFolder

def test(weights_path, use_se, test_dir):
    # 默认测试目录
    default_test_dir = 'Garbage data/test2'
    is_default_dir = (test_dir == default_test_dir)

    # 检查文件是否存在
    if not os.path.exists(weights_path):
        print(f'错误: 权重文件不存在: {weights_path}')
        return
    if not os.path.exists(test_dir):
        print(f'错误: 测试目录不存在: {test_dir}')
        return

    use_cuda = torch.cuda.is_available()
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    print(f'权重文件: {weights_path}')
    print(f'使用SE注意力: {use_se}')
    print(f'测试目录: {test_dir}')

    # 加载模型
    model = NetsTorch(
        net_name=config.net_name,
        pretrained=False,
        num_classes=config.num_classes,
        use_se=use_se
    )
    model.load_state_dict(torch.load(weights_path, map_location='cpu'), strict=False)

    if use_cuda:
        model = model.cuda()
    model.eval()

    # 加载测试数据集
    dataset_test = ImageFolder(data_dir=test_dir, image_size=config.image_size, is_train=False)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=32,  # 批次大小
        shuffle=False,
        num_workers=0
    )

    print(f'测试数据集大小: {len(dataset_test)}')
    print(f'类别: {dataset_test.classes}')
    print(f'\n{"="*60}')
    print(f'{"序号":<5} {"图片名":<30} {"真实标签":<15} {"预测标签":<15} {"结果"}')
    print(f'{"="*60}')

    # 测试
    n_correct = 0
    n_total = 0

    # 记录每个类别已输出的数量（用于非默认目录时限制输出）
    class_output_count = defaultdict(int)
    max_output_per_class = 3
    sample_idx = 0  # 全局样本索引

    with torch.no_grad():
        for batch_i, (imgs, labels) in enumerate(dataloader_test):
            imgs = imgs.type(FloatTensor)
            labels = labels.type(FloatTensor)
            preds = model(imgs)

            pred_indices = preds.max(-1)[1]
            batch_size = imgs.size(0)

            for j in range(batch_size):
                pred_idx = pred_indices[j].item()
                true_idx = int(labels[j].item())

                pred_label = dataset_test.classes[pred_idx]
                true_label = dataset_test.classes[true_idx]

                # 获取图片文件名
                img_path = dataset_test.image_paths[dataset_test.indexes[sample_idx]]
                img_name = os.path.basename(img_path)

                is_correct = pred_idx == true_idx
                result = "✓" if is_correct else "✗"

                # 判断是否输出该条记录
                if is_default_dir:
                    # 默认目录：输出所有
                    print(f'{sample_idx+1:<5} {img_name:<30} {true_label:<15} {pred_label:<15} {result}')
                else:
                    # 非默认目录：每个类别只输出前3张
                    if class_output_count[true_label] < max_output_per_class:
                        print(f'{sample_idx+1:<5} {img_name:<30} {true_label:<15} {pred_label:<15} {result}')
                        class_output_count[true_label] += 1

                n_correct += int(is_correct)
                n_total += 1
                sample_idx += 1

    acc = (n_correct / n_total) * 100
    print(f'\n========== 测试结果 ==========')
    print(f'正确数: {n_correct}/{n_total}')
    print(f'准确率: {acc:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='测试垃圾分类模型')
    parser.add_argument('-w', '--weights', type=str, default='model_saved/epoch_200.pkl',
                        help='模型权重路径 (默认: model_saved/epoch_200.pkl)')
    parser.add_argument('--use_se', action='store_true',
                        help='使用SE注意力机制')
    parser.add_argument('-d', '--data', type=str, default='test2',
                        choices=['train', 'test', 'test2'],
                        help='测试数据集: train, test, test2 (默认: test2)')
    args = parser.parse_args()

    # 构建测试目录路径
    test_dir = f'Garbage data/{args.data}'

    test(args.weights, args.use_se, test_dir)
