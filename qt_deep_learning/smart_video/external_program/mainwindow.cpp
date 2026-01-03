#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QThread>
#include <QImage>
#include <QPixmap>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // 设置界面名称
    setWindowTitle("调用外部程序");

    smartVideoDir = locateSmartVideoRoot();
    if (!smartVideoDir.isEmpty()) {
        QDir smartDir(smartVideoDir);
        txtDir = smartDir.filePath("txt");
        flagFilePath = QDir(txtDir).filePath("flag.txt");
        imageFilePath = QDir(txtDir).filePath("file.txt");
        resultFilePath = QDir(txtDir).filePath("test.txt");

        QDir classifierDirCandidate(smartDir);
        bool ok = classifierDirCandidate.cdUp();
        ok = ok && classifierDirCandidate.cdUp();
        ok = ok && classifierDirCandidate.cd("garbageClassifier-master");
        if (ok) {
            classifierDir = classifierDirCandidate.absolutePath();
        }
    } else {
        QMessageBox::warning(this, tr("路径错误"), tr("未找到 smart_video 目录，Qt 将无法和 Python 通信。"));
    }

    // 将信号函数与显示图片的槽函数进行连接
    connect(this, SIGNAL(showImageSignal()), this, SLOT(show_image()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

// 显示图片
void MainWindow::show_image()
{
    if (flagFilePath.isEmpty() || imageFilePath.isEmpty()) {
        return;
    }

    while (true)
    {
        QString result = ReadCommunication(flagFilePath);
        if (result == "true" || result.isEmpty()) {
            break;
        }
        QThread::msleep(50);
    }

    QString picpath = ReadCommunication(imageFilePath);
    if (picpath.isEmpty()) {
        QMessageBox::warning(this, tr("显示失败"), tr("无法读取图片路径。"));
        return;
    }

    QImage img_mainicon;
    if (!img_mainicon.load(picpath)) {
        QMessageBox::warning(this, tr("显示失败"), tr("无法加载图片: %1").arg(picpath));
        return;
    }

    img_mainicon = img_mainicon.scaled(ui->image_show->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->image_show->setScaledContents(true);
    ui->image_show->setPixmap(QPixmap::fromImage(img_mainicon));
    ui->image_show->show();
}

// 开始检测或停止检测
void MainWindow::on_startorend_clicked()
{
    if (!start_or_stop)
    {
        if (filename.isEmpty()) {
            QMessageBox::information(this, tr("提示"), tr("请先选择要识别的图片。"));
            return;
        }
        if (classifierDir.isEmpty()) {
            QMessageBox::critical(this, tr("路径错误"), tr("未找到 garbageClassifier-master 目录。"));
            return;
        }

        ui->state_Edit->setText("开始检测！");

        QProcess process;
        process.setWorkingDirectory(classifierDir);
        process.start("python", QStringList() << "demo1.py" << "--qt-bridge");
        if (!process.waitForFinished(-1)) {
            QMessageBox::warning(this, tr("运行失败"), tr("执行 demo1.py 超时。"));
            return;
        }
        if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
            const QString err = QString::fromLocal8Bit(process.readAllStandardError());
            QMessageBox::warning(this, tr("运行失败"), tr("执行 demo1.py 失败: %1").arg(err));
        }

        QString result_image = ReadCommunication(resultFilePath);
        ui->lineEdit->setText(result_image);
        emit showImageSignal();
        start_or_stop = true;
    }
    else // 再次点击，结束
    {
        ui->state_Edit->setText("停止检测！");
        ui->image_show->clear();
        start_or_stop = false;
    }
}

// 获取读取图片的路径并将其写入通信文件
void MainWindow::on_file_name_button_clicked()
{
    QString s = QFileDialog::getOpenFileName(
                this, "选择图片文件",
                        QDir::homePath(),
                 "图片文件((*.png *jpg *bmp);;");
    if (!s.isEmpty())
    {
        ui->file_name_Edit->setText(s);
        filename = s;
        if (!imageFilePath.isEmpty()) {
            WriteCommunication(imageFilePath, filename);
        }
    }
}

/*
 * 函数名称：ReadCommunication
 * 函数功能：读取通信文件内容
 * 输入参数：通信文件的路径
 * 输出参数：无
 * 返回值：读取到的内容
 */
QString MainWindow::ReadCommunication(const QString &path)
{
    QFile communication_file(path);
    if (!communication_file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "无法打开文件:" << path;
        return QString();
    }
    QString content = QString::fromUtf8(communication_file.readAll()).trimmed();
    communication_file.close();
    return content;
}

/*
 * 函数名称：WriteCommunication
 * 函数功能：写入通信文件
 * 输入参数：通信文件的路径、内容
 * 输出参数：无
 * 返回值：无
 */
void MainWindow::WriteCommunication(const QString &path, const QString &str)
{
    QFile communication_file(path);
    if (!communication_file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qWarning() << "无法写入文件:" << path;
        return;
    }
    communication_file.write(str.toUtf8());
    communication_file.close();
}

QString MainWindow::locateSmartVideoRoot() const
{
    QDir dir(QCoreApplication::applicationDirPath());
    for (int i = 0; i < 5; ++i) {
        if (dir.dirName() == "smart_video" && dir.exists("txt")) {
            return dir.absolutePath();
        }
        if (dir.exists("smart_video/txt")) {
            return dir.filePath("smart_video");
        }
        if (!dir.cdUp()) {
            break;
        }
    }
    return QString();
}
