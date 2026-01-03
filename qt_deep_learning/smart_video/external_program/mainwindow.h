#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QProcess>
#include <QMessageBox>
#include <QDialog>
#include <QFileDialog>
#include <QDebug>
#include <QString>
#include <QDir>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    QString filename;                 //存储读取图片的路径
    QProcess * detect_exe;            //创建外部程序
    bool start_or_stop = false;       //false:停止状态  true:开始状态

private slots:
    //槽函数

    void show_image();//显示函数
    void on_startorend_clicked();//检测开关
    void on_file_name_button_clicked();//读取文件
    QString ReadCommunication(const QString &path);//读取通信文件
    void WriteCommunication(const QString &path, const QString &str);//写入通信文件


private:
    Ui::MainWindow *ui;
    QString smartVideoDir;
    QString txtDir;
    QString flagFilePath;
    QString imageFilePath;
    QString resultFilePath;
    QString classifierDir;

    QString locateSmartVideoRoot() const;

signals:
    //信号函数
    void showImageSignal();
};

#endif // MAINWINDOW_H
