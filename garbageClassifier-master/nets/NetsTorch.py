'''
Function:
	nets defined in the torchvision module, support alexnet, resnet and vgg.
	支持带有 SE 注意力机制的 ResNet
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import torch
import torchvision
import torch.nn as nn
import sys
import os

# 导入本地带有 SE 注意力机制的 ResNet (思考题要求)
# 使用 importlib 显式导入本地模块，避免与系统 torchvision 冲突
import importlib.util
_local_resnet_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'torchvision_backup', 'models', 'resnet.py')
_spec = importlib.util.spec_from_file_location("local_resnet", _local_resnet_path)
_local_resnet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_local_resnet)

se_resnet18 = _local_resnet.resnet18
se_resnet34 = _local_resnet.resnet34
se_resnet50 = _local_resnet.resnet50
se_resnet101 = _local_resnet.resnet101
se_resnet152 = _local_resnet.resnet152


'''nets defined in the torchvision module.'''
class NetsTorch(nn.Module):
	def __init__(self, net_name, pretrained, num_classes, use_se=False, **kwargs):
		"""
		Args:
			net_name: 网络名称
			pretrained: 是否使用预训练权重
			num_classes: 分类类别数
			use_se: 是否使用 SE 注意力机制 (仅对 ResNet 有效)
		"""
		super(NetsTorch, self).__init__()
		net_name = net_name.lower()

		if net_name == 'alexnet':
			self.net_used = torchvision.models.alexnet(pretrained=pretrained)
		elif net_name == 'vgg11':
			self.net_used = torchvision.models.vgg11(pretrained=pretrained)
		elif net_name == 'vgg11_bn':
			self.net_used = torchvision.models.vgg11_bn(pretrained=pretrained)
		elif net_name == 'vgg13':
			self.net_used = torchvision.models.vgg13(pretrained=pretrained)
		elif net_name == 'vgg13_bn':
			self.net_used = torchvision.models.vgg13_bn(pretrained=pretrained)
		elif net_name == 'vgg16':
			self.net_used = torchvision.models.vgg16(pretrained=pretrained)
		elif net_name == 'vgg16_bn':
			self.net_used = torchvision.models.vgg16_bn(pretrained=pretrained)
		elif net_name == 'vgg19':
			self.net_used = torchvision.models.vgg19(pretrained=pretrained)
		elif net_name == 'vgg19_bn':
			self.net_used = torchvision.models.vgg19_bn(pretrained=pretrained)
		elif net_name == 'resnet18':
			if use_se:
				self.net_used = se_resnet18(pretrained=pretrained, use_se=True)
			else:
				self.net_used = torchvision.models.resnet18(pretrained=pretrained)
		elif net_name == 'resnet34':
			if use_se:
				self.net_used = se_resnet34(pretrained=pretrained, use_se=True)
			else:
				self.net_used = torchvision.models.resnet34(pretrained=pretrained)
		elif net_name == 'resnet50':
			if use_se:
				# 使用带有 SE 注意力机制的 ResNet50
				self.net_used = se_resnet50(pretrained=pretrained, use_se=True)
			else:
				self.net_used = torchvision.models.resnet50(pretrained=pretrained)
		elif net_name == 'resnet101':
			if use_se:
				self.net_used = se_resnet101(pretrained=pretrained, use_se=True)
			else:
				self.net_used = torchvision.models.resnet101(pretrained=pretrained)
		elif net_name == 'resnet152':
			if use_se:
				self.net_used = se_resnet152(pretrained=pretrained, use_se=True)
			else:
				self.net_used = torchvision.models.resnet152(pretrained=pretrained)
		elif net_name == 'inception_v3':
			self.net_used = torchvision.models.inception_v3(pretrained=pretrained)
		else:
			raise ValueError('Unsupport NetsTorch.net_name <%s>...' % net_name)

		# 修改最后的分类层以适应自定义类别数
		if net_name in ['alexnet', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']:
			self.net_used.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes)
		elif net_name in ['inception_v3', 'resnet50', 'resnet101', 'resnet152']:
			self.net_used.fc = nn.Linear(in_features=2048, out_features=num_classes)
		elif net_name in ['resnet18', 'resnet34']:
			self.net_used.fc = nn.Linear(in_features=512, out_features=num_classes)

	def forward(self, x):
		x = self.net_used(x)
		return x