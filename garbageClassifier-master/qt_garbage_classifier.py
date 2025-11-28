'''
Function:
    QT 界面垃圾分类应用
Description:
    使用 PyQt5 实现垃圾分类的图形界面
    支持选择图片、显示分类结果
'''
import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QFrame, QGroupBox
)
from PyQt5.QtGui import QPixmap, QFont, QColor, QPalette
from PyQt5.QtCore import Qt

# 添加当前目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from demo1 import GarbageClassifier, GARBAGE_CATEGORIES


class GarbageClassifierApp(QMainWindow):
    """垃圾分类 QT 应用主窗口"""

    # 垃圾类别对应的颜色
    CATEGORY_COLORS = {
        '可回收物': '#3498db',   # 蓝色
        '厨余垃圾': '#27ae60',   # 绿色
        '有害垃圾': '#e74c3c',   # 红色
        '其他垃圾': '#95a5a6',   # 灰色
    }

    def __init__(self):
        super().__init__()
        self.classifier = None
        self.current_image_path = None
        self.init_ui()
        self.init_classifier()

    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle('智能垃圾分类系统')
        self.setMinimumSize(800, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f6fa;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                border-radius: 5px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
            QLabel {
                color: #2c3e50;
            }
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

        # 主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

        # 标题
        title_label = QLabel('智能垃圾分类系统')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont('Microsoft YaHei', 24, QFont.Bold))
        title_label.setStyleSheet('color: #2c3e50; margin-bottom: 20px;')
        main_layout.addWidget(title_label)

        # 内容区域
        content_layout = QHBoxLayout()
        content_layout.setSpacing(30)

        # 左侧：图片显示区
        image_group = QGroupBox('图片预览')
        image_layout = QVBoxLayout(image_group)

        self.image_label = QLabel('请选择一张垃圾图片')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(350, 350)
        self.image_label.setStyleSheet('''
            QLabel {
                background-color: white;
                border: 2px dashed #bdc3c7;
                border-radius: 10px;
                font-size: 16px;
                color: #7f8c8d;
            }
        ''')
        image_layout.addWidget(self.image_label)

        # 选择图片按钮
        self.select_btn = QPushButton('选择图片')
        self.select_btn.clicked.connect(self.select_image)
        image_layout.addWidget(self.select_btn, alignment=Qt.AlignCenter)

        content_layout.addWidget(image_group)

        # 右侧：结果显示区
        result_group = QGroupBox('分类结果')
        result_layout = QVBoxLayout(result_group)
        result_layout.setSpacing(20)

        # 垃圾大类显示
        self.category_label = QLabel('--')
        self.category_label.setAlignment(Qt.AlignCenter)
        self.category_label.setFont(QFont('Microsoft YaHei', 28, QFont.Bold))
        self.category_label.setMinimumHeight(80)
        self.category_label.setStyleSheet('''
            QLabel {
                background-color: #ecf0f1;
                border-radius: 10px;
                padding: 20px;
            }
        ''')
        result_layout.addWidget(self.category_label)

        # 具体类别显示
        detail_frame = QFrame()
        detail_layout = QVBoxLayout(detail_frame)

        self.class_name_label = QLabel('物品类别: --')
        self.class_name_label.setFont(QFont('Microsoft YaHei', 14))
        detail_layout.addWidget(self.class_name_label)

        self.confidence_label = QLabel('置信度: --')
        self.confidence_label.setFont(QFont('Microsoft YaHei', 14))
        detail_layout.addWidget(self.confidence_label)

        self.status_label = QLabel('')
        self.status_label.setFont(QFont('Microsoft YaHei', 12))
        self.status_label.setStyleSheet('color: #7f8c8d;')
        detail_layout.addWidget(self.status_label)

        result_layout.addWidget(detail_frame)

        # 分类按钮
        self.classify_btn = QPushButton('开始分类')
        self.classify_btn.setEnabled(False)
        self.classify_btn.clicked.connect(self.classify_image)
        self.classify_btn.setStyleSheet('''
            QPushButton {
                background-color: #27ae60;
                font-size: 16px;
                padding: 15px 30px;
            }
            QPushButton:hover {
                background-color: #219a52;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        ''')
        result_layout.addWidget(self.classify_btn, alignment=Qt.AlignCenter)

        result_layout.addStretch()
        content_layout.addWidget(result_group)

        main_layout.addLayout(content_layout)

        # 底部说明
        footer_label = QLabel('垃圾分类: 可回收物(蓝) | 厨余垃圾(绿) | 有害垃圾(红) | 其他垃圾(灰)')
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setStyleSheet('color: #7f8c8d; font-size: 12px;')
        main_layout.addWidget(footer_label)

    def init_classifier(self):
        """初始化分类器"""
        try:
            self.status_label.setText('正在加载模型...')
            QApplication.processEvents()
            self.classifier = GarbageClassifier()
            self.status_label.setText('模型加载完成')
        except Exception as e:
            self.status_label.setText(f'模型加载失败: {str(e)}')
            self.status_label.setStyleSheet('color: #e74c3c;')

    def select_image(self):
        """选择图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            '选择垃圾图片',
            '',
            'Images (*.png *.jpg *.jpeg *.bmp *.gif)'
        )

        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.classify_btn.setEnabled(True)
            self.reset_result()

    def display_image(self, image_path):
        """显示图片"""
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(
            330, 330,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setStyleSheet('''
            QLabel {
                background-color: white;
                border: 2px solid #3498db;
                border-radius: 10px;
            }
        ''')

    def reset_result(self):
        """重置结果显示"""
        self.category_label.setText('--')
        self.category_label.setStyleSheet('''
            QLabel {
                background-color: #ecf0f1;
                border-radius: 10px;
                padding: 20px;
            }
        ''')
        self.class_name_label.setText('物品类别: --')
        self.confidence_label.setText('置信度: --')
        self.status_label.setText('请点击"开始分类"按钮')

    def classify_image(self):
        """对图片进行分类"""
        if not self.current_image_path or not self.classifier:
            return

        self.status_label.setText('正在识别...')
        QApplication.processEvents()

        result = self.classifier.classify(self.current_image_path)

        if result['success']:
            category = result['category']
            color = self.CATEGORY_COLORS.get(category, '#95a5a6')

            self.category_label.setText(category)
            self.category_label.setStyleSheet(f'''
                QLabel {{
                    background-color: {color};
                    color: white;
                    border-radius: 10px;
                    padding: 20px;
                }}
            ''')

            self.class_name_label.setText(f"物品类别: {result['class_name']}")
            self.confidence_label.setText(f"置信度: {result['confidence']:.2%}")
            self.status_label.setText('识别成功')
            self.status_label.setStyleSheet('color: #27ae60;')
        else:
            self.category_label.setText('识别失败')
            self.category_label.setStyleSheet('''
                QLabel {
                    background-color: #e74c3c;
                    color: white;
                    border-radius: 10px;
                    padding: 20px;
                }
            ''')
            self.class_name_label.setText('物品类别: --')
            self.confidence_label.setText('置信度: --')
            self.status_label.setText(result.get('message', '未知错误'))
            self.status_label.setStyleSheet('color: #e74c3c;')


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = GarbageClassifierApp()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
