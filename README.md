# 1. 进入项目目录
cd garbageClassifier-master

# 2. 训练模型
python train.py

# 3. 测试单张图片
python demo.py -i 图片路径.jpg

# 4. 运行 QT 界面
python qt_garbage_classifier.py



# 1. 普通训练 (不使用 SE)
python train.py

# 2. 使用 SE 注意力机制训练
python train.py --use_se

# 3. 测试
python demo.py -i 图片路径.jpg

# 4. QT 界面
python qt_garbage_classifier.py

