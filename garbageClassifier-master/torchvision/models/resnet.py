"""
ResNet with SE (Squeeze-and-Excitation) Attention Mechanism

基于 PyTorch 官方 ResNet 实现，添加了 SE 注意力模块
参考: https://arxiv.org/abs/1709.01507 (Squeeze-and-Excitation Networks)
"""
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional


# 预训练模型 URL
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
}


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (SE 注意力模块)

    通过全局平均池化获取通道级别的全局信息，
    然后通过两个全连接层学习通道间的依赖关系，
    最后用 sigmoid 激活函数得到通道权重，对原特征进行加权。

    Args:
        channel: 输入特征的通道数
        reduction: 压缩比例，用于减少全连接层的参数量，默认为16
    """

    def __init__(self, channel: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        # 全局平均池化，将 H×W 压缩为 1×1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 两个全连接层组成的 "Excitation" 部分
        self.fc = nn.Sequential(
            # 第一个全连接层：压缩通道数
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            # 第二个全连接层：恢复通道数
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()  # 输出 0-1 之间的权重
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Squeeze: 全局平均池化
        y = self.avg_pool(x).view(b, c)
        # Excitation: 通过全连接层学习通道权重
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: 用学习到的权重对原特征进行缩放
        return x * y.expand_as(x)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 卷积 with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 卷积"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    ResNet 基础残差块 (用于 ResNet-18, ResNet-34)
    添加了 SE 注意力模块
    """
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_se: bool = True,
        se_reduction: int = 16
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # 两个 3x3 卷积层
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        # SE 注意力模块
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(planes, se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 应用 SE 注意力
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    ResNet 瓶颈残差块 (用于 ResNet-50, ResNet-101, ResNet-152)
    添加了 SE 注意力模块
    """
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_se: bool = True,
        se_reduction: int = 16
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        # 1x1, 3x3, 1x1 卷积
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # SE 注意力模块 (应用在最后的通道数上)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(planes * self.expansion, se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 应用 SE 注意力
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    带有 SE 注意力机制的 ResNet

    Args:
        block: 残差块类型 (BasicBlock 或 Bottleneck)
        layers: 每个 stage 的残差块数量
        num_classes: 分类类别数
        use_se: 是否使用 SE 注意力模块
        se_reduction: SE 模块的压缩比例
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_se: bool = True,
        se_reduction: int = 16
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.use_se = use_se
        self.se_reduction = se_reduction

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 个 stage
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,
                            use_se=self.use_se, se_reduction=self.se_reduction))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, use_se=self.use_se,
                                se_reduction=self.se_reduction))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # 前向传播
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    use_se: bool = True,
    **kwargs: Any
) -> ResNet:
    """ResNet 模型构建函数"""
    model = ResNet(block, layers, use_se=use_se, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        # 使用 strict=False 允许加载预训练权重时忽略 SE 模块的权重不匹配
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, use_se: bool = True, **kwargs: Any) -> ResNet:
    """
    ResNet-18 with SE attention

    Args:
        pretrained: 是否加载预训练权重
        progress: 是否显示下载进度
        use_se: 是否使用 SE 注意力模块
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, use_se, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, use_se: bool = True, **kwargs: Any) -> ResNet:
    """
    ResNet-34 with SE attention
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, use_se, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, use_se: bool = True, **kwargs: Any) -> ResNet:
    """
    ResNet-50 with SE attention

    这是实验中使用的主要网络。
    SE 模块会在每个残差块的输出上应用通道注意力。
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, use_se, **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, use_se: bool = True, **kwargs: Any) -> ResNet:
    """
    ResNet-101 with SE attention
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, use_se, **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, use_se: bool = True, **kwargs: Any) -> ResNet:
    """
    ResNet-152 with SE attention
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, use_se, **kwargs)
