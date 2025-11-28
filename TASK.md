# 实验一：目标分类网络应用

## 1.1 实验要求

1. 开发环境为 Ubuntu 18.04 和 PyTorch 1.8.2。  
2. 按照实验具体要求对网络进行修改，以提高网络的性能。  
3. 使用指定的数据集进行网络的训练。  
4. 保存实验结果，包括关键实验参数、最终模型、运行截图。  
5. 提交一份针对该实验的报告，格式参照 markdown 模板。

## 1.2 实验步骤

### 1. 选择合适的分类网络

在 `garbageClassifier-master` 文件夹下的 `config.py` 中选择 ResNet50 作为分类网络。

### 2. 添加补充数据集

按照资料中的 `README` 文件说明，添加补充数据集。

### 3. 对网络进行训练

添加完数据后，运行以下命令进行训练：

```bash
python3 train.py
```

训练结果将保存在 `model_saved` 目录中。

### 4. 测试模型

- 在 `config.py` 中修改测试时需要加载的模型路径。
- 运行以下命令进行测试（`xxx` 替换为待测试图片的路径）：

```bash
python3 demo.py -i xxx
```

### 5. 使用 QT 对算法模型进行集成

**具体要求：**

- 使用 QT 完成对算法模型的调用及结果读取。需将 `demo.py` 的调用逻辑改写为适合 QT 调用的形式，并重命名为 `demo1.py`。
- 设计 QT 界面，实现选择图片、结果显示等功能。
- 输出该垃圾图片中物品所属类别，包括以下四类之一：
  - 可回收物
  - 厨余垃圾
  - 有害垃圾
  - 其他垃圾  
- 要求具有较好的显示效果。

### 6. 资料

- 垃圾分类题目概述及资料汇总：  
  [https://github.com/wusaifei/garbage_classify](https://github.com/wusaifei/garbage_classify)
- PyTorch 版本参考代码：  
  [https://github.com/CharlesPikachu/garbageClassifier](https://github.com/CharlesPikachu/garbageClassifier)

## 1.3 思考题

**对 ResNet 网络进行改进，如加入 SE 注意力机制。**

操作步骤如下：

1. 将位于 `/home/b401-24/.local/lib/python3.6/site-packages` 目录下的 `torchvision` 文件夹复制一份到 `garbageClassifier-master` 目录中。
   > 注：`b401-24` 是实验室某台计算机的名称，其他机器名称后缀可能略有差异。

2. 修改 `garbageClassifier-master/torchvision/models/resnet.py` 文件，在其中加入 SE（Squeeze-and-Excitation）注意力机制。  
   - 参考教程：[https://zhuanlan.zhihu.com/p/99261200](https://zhuanlan.zhihu.com/p/99261200)

3. 在 `resnet.py` 文件中的 `def _resnet` 函数里，将以下代码：
   ```python
   model.load_state_dict(state_dict)
   ```
   修改为：
   ```python
   model.load_state_dict(state_dict, strict=False)
   ```

4. 在 `nets` 文件夹下的 `netstorch.py` 中添加如下导入语句：
   ```python
   from torchvision.models import resnet50
   ```

5. 重复步骤 3（重新训练模型）。
