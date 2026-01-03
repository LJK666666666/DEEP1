# 垃圾分类实验运行指引

## 进入项目

```bash
cd garbageClassifier-master
```

## 训练与推理命令

```bash
# 普通训练（不启用 SE 注意力）
python train.py

# 启用 SE 注意力机制训练
python train.py --use_se

# 使用官方 demo 进行单张推理
python demo.py -i 路径/到/图片.jpg

# 使用为 Qt 重构后的分类器（命令行模式）
python demo1.py -i 路径/到/图片.jpg

# 启用 qt_deep_learning 的 txt 通信模式（通常由 Qt 程序自动调用）
python demo1.py --qt-bridge

python qt_garbage_classifier.py
```

## Qt（qt_deep_learning）联调流程

1. 构建 C++ Qt 前端（示例使用 qmake，亦可在 Qt Creator 中打开 `qt_deep_learning/smart_video/external_program/external_program.pro` 直接构建）：
   ```bash
   cd qt_deep_learning/smart_video/external_program
   qmake external_program.pro
   nmake        # 或者 mingw32-make / make，依据 Qt 套件决定
   ```
2. 运行生成的可执行文件（通常位于 `qt_deep_learning/smart_video/build-*/external_program.exe`，也可以在 Qt Creator 中直接点击运行）。
3. Qt 程序的交互逻辑：
   - 点击“选择文件”后，程序会把图片路径写入 `qt_deep_learning/smart_video/txt/file.txt`；
   - 点击“开始检测”后，Qt 会自动执行：
     ```bash
     cd garbageClassifier-master
     python demo1.py --qt-bridge
     ```
     该模式会读取 `file.txt`，完成推理后把结果写入 `qt_deep_learning/smart_video/txt/test.txt`，并把 `flag.txt` 置为 `true`；
   - Qt 轮询 `flag.txt`，当其为 `true` 时，刷新界面并显示 `test.txt` 中的文本结果以及原图。
4. 如需单独验证 Qt 通信是否正常，可在命令行手动运行：
   ```bash
   cd garbageClassifier-master
   python demo1.py --qt-bridge
   ```
   然后检查 `qt_deep_learning/smart_video/txt/test.txt` 与界面显示是否一致。

> 提示：若修改了 `qt_deep_learning/smart_video/txt` 目录位置，可在 Qt 项目中重新定位该目录；`demo1.py` 会自动按仓库默认结构查找。

codex resume 019ad0b8-3fe8-7170-97e0-4d673503ce84