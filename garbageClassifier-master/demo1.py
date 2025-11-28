'''
Function:
	demo1.py - 供 QT 调用的垃圾分类模块
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# 获取当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import config
from utils.utils import loadClasses
from nets.NetsTorch import NetsTorch


# 垃圾分类映射字典
GARBAGE_CATEGORIES = {
    # 可回收物
    'cardboard': '可回收物',
    'glass': '可回收物',
    'metal': '可回收物',
    'paper': '可回收物',
    'plastic': '可回收物',
    # 厨余垃圾
    'leftovers': '厨余垃圾',
    'pericarp': '厨余垃圾',
    # 有害垃圾
    'cell': '有害垃圾',
    'power_bank': '有害垃圾',
    # 其他垃圾
    'trash': '其他垃圾',
    'cigarette': '其他垃圾',
}


class GarbageClassifier:
    """垃圾分类器类，供 QT 调用"""

    def __init__(self, weights_path=None):
        """
        初始化分类器
        Args:
            weights_path: 模型权重路径，如果为 None 则使用 config 中的默认路径
        """
        self.use_cuda = torch.cuda.is_available()
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        # 加载类别名称
        cls_path = os.path.join(SCRIPT_DIR, config.clsnamespath)
        self.classes = loadClasses(cls_path)

        # 加载模型
        self.model = NetsTorch(
            net_name=config.net_name,
            pretrained=False,
            num_classes=config.num_classes
        )

        # 加载权重
        if weights_path is None:
            weights_path = os.path.join(SCRIPT_DIR, config.weightspath)

        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location='cpu'), strict=False)
        else:
            print(f"警告: 模型权重文件不存在: {weights_path}")

        if self.use_cuda:
            self.model = self.model.cuda()
        self.model.eval()

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def classify(self, image_path):
        """
        对图片进行分类
        Args:
            image_path: 图片路径
        Returns:
            dict: 包含分类结果的字典
                - class_name: 具体物品类别 (如 glass, paper 等)
                - category: 垃圾大类 (可回收物/厨余垃圾/有害垃圾/其他垃圾)
                - confidence: 置信度
                - success: 是否成功识别
        """
        try:
            # 加载图片
            img = Image.open(image_path).convert('RGB')
            img_input = self.transform(img)
            img_input = img_input.type(self.FloatTensor).unsqueeze(0)

            # 推理
            with torch.no_grad():
                preds = self.model(img_input)

            preds = nn.Softmax(-1)(preds).cpu()
            max_prob, max_prob_id = preds.view(-1).max(0)
            confidence = max_prob.item()
            class_idx = max_prob_id.item()
            class_name = self.classes[class_idx]

            # 获取垃圾大类
            category = GARBAGE_CATEGORIES.get(class_name, '其他垃圾')

            if confidence > config.conf_thresh:
                return {
                    'class_name': class_name,
                    'category': category,
                    'confidence': confidence,
                    'success': True
                }
            else:
                return {
                    'class_name': class_name,
                    'category': category,
                    'confidence': confidence,
                    'success': False,
                    'message': '置信度过低，无法确定分类'
                }
        except Exception as e:
            return {
                'class_name': None,
                'category': None,
                'confidence': 0,
                'success': False,
                'message': str(e)
            }


def get_garbage_category(class_name):
    """
    根据具体类别获取垃圾大类
    Args:
        class_name: 具体物品类别
    Returns:
        str: 垃圾大类
    """
    return GARBAGE_CATEGORIES.get(class_name, '其他垃圾')


def classify_image(image_path, weights_path=None):
    """
    便捷函数：对单张图片进行分类
    Args:
        image_path: 图片路径
        weights_path: 模型权重路径（可选）
    Returns:
        dict: 分类结果
    """
    classifier = GarbageClassifier(weights_path)
    return classifier.classify(image_path)


# 兼容原有命令行调用方式
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="垃圾分类器")
    parser.add_argument('-i', '--image', dest='image', help='待分类的图片路径')
    args = parser.parse_args()

    if args.image:
        classifier = GarbageClassifier()
        result = classifier.classify(args.image)

        if result['success']:
            print(f"[物品类别]: {result['class_name']}")
            print(f"[垃圾分类]: {result['category']}")
            print(f"[置信度]: {result['confidence']:.4f}")
        else:
            print(f"识别失败: {result.get('message', '未知错误')}")