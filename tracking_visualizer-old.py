import os
import sys

# 确保使用UTF-8编码
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import json
from pathlib import Path
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QSlider, QComboBox, QGroupBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QProgressBar, QMessageBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QTextEdit, QSplitter, QTreeWidget,
    QTreeWidgetItem, QHeaderView, QDockWidget, QSizePolicy , QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QTextCursor
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl

from tracking_system import TrackingSystem

# 999999999999999999999
# from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
#                            QHBoxLayout, QGridLayout,  # 确保添加 QGridLayout
#                            QGroupBox, QLabel, QComboBox, QPushButton,
#                            QTextEdit, QProgressBar, QTabWidget, QSpinBox,
#                            QDoubleSpinBox, QSplitter, QMessageBox, QVideoWidget)
# from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl
# from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent


class TrackingThread(QThread):
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(int, str)

    def __init__(self, tracking_system, seq_path):
        super().__init__()
        self.tracking_system = tracking_system
        self.seq_path = seq_path

    def run(self):
        success, message = self.tracking_system.run_full_pipeline(self.seq_path)
        self.finished.emit(success, message)


class VisualizationThread(QThread):
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(int, str)

    def __init__(self, tracking_system, trail, fps):
        super().__init__()
        self.tracking_system = tracking_system
        self.trail = trail
        self.fps = fps

    def run(self):
        success, message = self.tracking_system.run_visualization(trail=self.trail, fps=self.fps)
        self.finished.emit(success, message)


class MainWindow(QMainWindow):
    # def __init__(self):
    #     super().__init__()
    #     self.tracking_system = TrackingSystem()
    #     self.init_ui()
    #     self.tracking_thread = None
    #     self.visualization_thread = None
    #     self.media_player = None

    def __init__(self):
        super().__init__()
        self.tracking_system = TrackingSystem()
        self.init_ui()
        self.tracking_thread = None
        self.visualization_thread = None
        self.media_player = None

    def init_ui(self):
        self.setWindowTitle("基于动态特征优化与非线性建模的多目标跟踪系统")
        self.setGeometry(100, 100, 1400, 1000)

        # 主布局
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # # 顶部控制区
        # control_group = QGroupBox("控制面板")
        # control_layout = QHBoxLayout()
        # control_group.setLayout(control_layout)

        # 改为网格布局，支持行列排版9999999999999999999999999999
        control_group = QGroupBox("控制面板")
        control_layout = QGridLayout()  # 关键：改为网格布局
        control_layout.setSpacing(10)  # 组件间距
        control_layout.setContentsMargins(10, 10, 10, 10)  # 边距
        control_group.setLayout(control_layout)


        # # 序列选择
        # self.seq_combo = QComboBox()
        # self.seq_combo.setMinimumWidth(200)
        # control_layout.addWidget(QLabel("选择序列:"))
        # control_layout.addWidget(self.seq_combo)
        #
        # # 加载序列按钮
        # self.load_seq_btn = QPushButton("加载序列")
        # self.load_seq_btn.clicked.connect(self.load_sequences)
        # control_layout.addWidget(self.load_seq_btn)
        #
        # # 新增：YOLO模型选择
        # control_layout.addWidget(QLabel("YOLO模型:"))
        # self.yolo_combo = QComboBox()
        # # 从配置中获取默认模型，并添加可选模型（根据实际权重文件添加）
        # default_yolo = self.tracking_system.config['yolo_model']
        # self.yolo_combo.addItem(os.path.basename(default_yolo), default_yolo)
        # # 可手动添加其他模型路径（例如）
        # self.yolo_combo.addItem("yolo11n.pt", str(Path(default_yolo).parent / "yolo11n.pt"))
        # control_layout.addWidget(self.yolo_combo)
        # self.yolo_combo.addItem("yolov8n.pt", str(Path(default_yolo).parent / "yolov8n.pt"))
        # control_layout.addWidget(self.yolo_combo)
        #
        # # 新增：跟踪方法选择
        # control_layout.addWidget(QLabel("跟踪方法:"))
        # self.track_method_combo = QComboBox()
        # # 预设可选方法（当前是bytetrack，可添加其他如ocsort等）
        # track_methods = ["bytetrack_with_", "bytetrack", "botsort"]
        # default_method = self.tracking_system.config['tracking_method']
        # self.track_method_combo.addItems(track_methods)
        # self.track_method_combo.setCurrentText(default_method)  # 默认选中当前方法
        # control_layout.addWidget(self.track_method_combo)
        #
        #
        #
        #
        #
        # # 处理按钮
        # self.process_btn = QPushButton("处理序列")
        # self.process_btn.clicked.connect(self.process_sequence)
        # self.process_btn.setEnabled(False)
        # control_layout.addWidget(self.process_btn)
        #
        # # 可视化按钮
        # self.visualize_btn = QPushButton("可视化结果")
        # self.visualize_btn.clicked.connect(self.visualize_tracks)
        # self.visualize_btn.setEnabled(False)
        # control_layout.addWidget(self.visualize_btn)
        #
        # # 尾迹长度设置
        # control_layout.addWidget(QLabel("尾迹长度:"))
        # self.trail_spin = QSpinBox()
        # self.trail_spin.setRange(1, 100)
        # self.trail_spin.setValue(30)
        # control_layout.addWidget(self.trail_spin)
        #
        # # FPS设置
        # control_layout.addWidget(QLabel("FPS:"))
        # self.fps_spin = QDoubleSpinBox()
        # self.fps_spin.setRange(1, 60)
        # self.fps_spin.setValue(25.0)
        # control_layout.addWidget(self.fps_spin)
        #
        # main_layout.addWidget(control_group)

        # 第0行：序列选择和加载按钮
        control_layout.addWidget(QLabel("选择序列:"), 0, 0)  # 行0，列0
        self.seq_combo = QComboBox()
        self.seq_combo.setMinimumWidth(200)
        control_layout.addWidget(self.seq_combo, 0, 1)  # 行0，列1

        self.load_seq_btn = QPushButton("加载序列")
        self.load_seq_btn.clicked.connect(self.load_sequences)
        control_layout.addWidget(self.load_seq_btn, 0, 2)  # 行0，列2

        # 第1行：处理按钮和可视化按钮
        self.process_btn = QPushButton("处理序列")
        self.process_btn.clicked.connect(self.process_sequence)
        self.process_btn.setEnabled(False)
        control_layout.addWidget(self.process_btn, 1, 0)  # 行1，列0

        self.visualize_btn = QPushButton("可视化结果")
        self.visualize_btn.clicked.connect(self.visualize_tracks)
        self.visualize_btn.setEnabled(False)
        control_layout.addWidget(self.visualize_btn, 1, 1)  # 行1，列1

        # 第2行：尾迹长度和FPS设置
        control_layout.addWidget(QLabel("尾迹长度:"), 2, 0)  # 行2，列0
        self.trail_spin = QSpinBox()
        self.trail_spin.setRange(1, 100)
        self.trail_spin.setValue(30)
        control_layout.addWidget(self.trail_spin, 2, 1)  # 行2，列1

        control_layout.addWidget(QLabel("FPS:"), 2, 2)  # 行2，列2
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(25.0)
        control_layout.addWidget(self.fps_spin, 2, 3)  # 行2，列3

        # 第3行：YOLO模型选择（单独一行）
        control_layout.addWidget(QLabel("YOLO模型:"), 3, 0)  # 行3，列0
        self.yolo_combo = QComboBox()
        self.yolo_combo.setMinimumWidth(200)
        # 从配置中加载默认模型
        default_yolo = self.tracking_system.config['yolo_model']
        self.yolo_combo.addItem(os.path.basename(default_yolo), default_yolo)
        # 可添加其他模型（示例）
        self.yolo_combo.addItem("yolov8n.pt", str(Path(default_yolo).parent / "yolov8n.pt"))
        control_layout.addWidget(self.yolo_combo, 3, 1, 1, 3)  # 行3，列1-3（跨3列，避免过短）

        # 第4行：跟踪方法选择（单独一行）
        control_layout.addWidget(QLabel("跟踪方法:"), 4, 0)  # 行4，列0
        self.track_method_combo = QComboBox()
        track_methods = ["bytetrack", "ocsort", "botsort", "deepocsort"]  # 支持的方法
        default_method = self.tracking_system.config['tracking_method']
        self.track_method_combo.addItems(track_methods)
        self.track_method_combo.setCurrentText(default_method)
        control_layout.addWidget(self.track_method_combo, 4, 1, 1, 3)  # 行4，列1-3（跨3列）






        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # 状态标签
        self.status_label = QLabel("就绪")
        main_layout.addWidget(self.status_label)

        # 结果展示区
        result_tabs = QTabWidget()
        main_layout.addWidget(result_tabs, 1)

        # 检测结果标签页
        det_tab = QWidget()
        det_layout = QVBoxLayout()
        det_tab.setLayout(det_layout)
        self.det_text = QTextEdit()
        self.det_text.setReadOnly(True)
        self.det_text.setFont(QFont("Courier New", 10))
        det_layout.addWidget(self.det_text)
        result_tabs.addTab(det_tab, "检测结果")

        # 跟踪结果标签页
        track_tab = QWidget()
        track_layout = QVBoxLayout()
        track_tab.setLayout(track_layout)
        self.track_text = QTextEdit()
        self.track_text.setReadOnly(True)
        self.track_text.setFont(QFont("Courier New", 10))
        track_layout.addWidget(self.track_text)
        result_tabs.addTab(track_tab, "跟踪结果")

        # 视频展示区
        video_group = QGroupBox("跟踪可视化")
        video_layout = QVBoxLayout()
        video_group.setLayout(video_layout)

        # 创建视频播放器
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(800, 450)
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 可扩展
        video_layout.addWidget(self.video_widget)

        # 播放控制按钮
        player_controls = QHBoxLayout()
        self.play_btn = QPushButton("播放")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.play_video)
        player_controls.addWidget(self.play_btn)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_video)
        player_controls.addWidget(self.stop_btn)

        self.open_btn = QPushButton("打开文件夹")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self.open_output_folder)
        player_controls.addWidget(self.open_btn)

        video_layout.addLayout(player_controls)
        main_layout.addWidget(video_group)

        # # 使用分割器
        # main_splitter = QSplitter(Qt.Vertical)
        # main_splitter.addWidget(result_tabs)
        # main_splitter.addWidget(video_group)
        # main_splitter.setSizes([400, 600])  # 上部分400像素，下部分600像素
        #
        # main_layout.addWidget(main_splitter, 1)  # 占据所有空间

        # 使用分割器（改为水平方向）
        main_splitter = QSplitter(Qt.Horizontal)  # 关键：Vertical→Horizontal
        main_splitter.addWidget(result_tabs)  # 左侧：检测/跟踪结果
        main_splitter.addWidget(video_group)  # 右侧：可视化视频
        main_splitter.setSizes([600, 800])  # 左侧600像素，右侧800像素（根据屏幕调整）

        main_layout.addWidget(main_splitter, 1)  # 占据所有空间


        # 加载序列
        self.load_sequences()



    def load_sequences(self):
        self.seq_combo.clear()
        sequences = self.tracking_system.get_sequence_list()
        if not sequences:
            self.status_label.setText("未找到序列")
            return

        for seq in sequences:
            self.seq_combo.addItem(f"{seq['name']} ({seq['frame_count']}帧)", seq['path'])

        self.process_btn.setEnabled(True)
        self.status_label.setText(f"找到 {len(sequences)} 个序列")

    def process_sequence(self):
        seq_path = self.seq_combo.currentData()
        if not seq_path:
            QMessageBox.warning(self, "警告", "请先选择序列")
            return

        # 禁用按钮
        self.process_btn.setEnabled(False)
        self.load_seq_btn.setEnabled(False)
        self.status_label.setText(f"处理序列: {Path(seq_path).name}...")

        # 启动处理线程
        self.tracking_thread = TrackingThread(self.tracking_system, seq_path)
        self.tracking_thread.finished.connect(self.on_tracking_finished)
        self.tracking_thread.start()

        # 更新进度
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(500)

    def update_progress(self):
        self.progress_bar.setValue(self.tracking_system.progress)

    def on_tracking_finished(self, success, message):
        self.progress_timer.stop()
        self.load_seq_btn.setEnabled(True)
        self.process_btn.setEnabled(True)

        if success:
            self.status_label.setText("处理完成")
            self.progress_bar.setValue(100)

            # 打印文件路径
            print(f"尝试加载检测文件: {self.tracking_system.det_file_path}")
            print(f"尝试加载跟踪文件: {self.tracking_system.track_file_path}")

            # 显示结果
            self.det_text.setText(self.tracking_system.load_detections())
            self.track_text.setText(self.tracking_system.load_tracks())

            # 启用可视化按钮
            self.visualize_btn.setEnabled(True)
        else:
            self.status_label.setText(f"错误: {message}")
            QMessageBox.critical(self, "处理失败", message)

    def visualize_tracks(self):
        if not self.tracking_system.track_file_path:
            QMessageBox.warning(self, "警告", "请先处理序列")
            return

        # 获取参数
        trail = self.trail_spin.value()
        fps = self.fps_spin.value()

        # 禁用按钮
        self.visualize_btn.setEnabled(False)
        self.status_label.setText("正在生成可视化视频...")

        # 启动可视化线程
        self.visualization_thread = VisualizationThread(self.tracking_system, trail, fps)
        self.visualization_thread.finished.connect(self.on_visualization_finished)
        self.visualization_thread.start()

    def on_visualization_finished(self, success, message):
        self.visualize_btn.setEnabled(True)

        if success:
            self.status_label.setText(f"可视化完成: {message}")
            self.tracking_system.visualization_path = Path(message)

            # 启用视频播放按钮
            self.play_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.open_btn.setEnabled(True)
        else:
            self.status_label.setText(f"可视化失败: {message}")
            QMessageBox.critical(self, "可视化失败", message)

    def play_video(self):
        print(f"尝试播放视频: {self.tracking_system.visualization_path}")
        if not os.path.exists(self.tracking_system.visualization_path):
            print("错误：视频文件不存在")
        else:
            print("视频文件存在")


        """播放生成的视频"""
        if not self.tracking_system.visualization_path:
            QMessageBox.warning(self, "警告", "尚未生成可视化视频")
            return

        # 初始化媒体播放器
        if not self.media_player:
            self.media_player = QMediaPlayer()
            self.media_player.setVideoOutput(self.video_widget)

        # 设置媒体源
        video_url = QUrl.fromLocalFile(str(self.tracking_system.visualization_path))
        print("self.tracking_system.visualization_path : ",self.tracking_system.visualization_path)
        self.media_player.setMedia(QMediaContent(video_url))
        self.media_player.play()




    def stop_video(self):
        """停止播放视频"""
        if self.media_player:
            self.media_player.stop()

    def open_output_folder(self):
        """打开输出文件夹"""
        if not self.tracking_system.visualization_path:
            QMessageBox.warning(self, "警告", "尚未生成可视化视频")
            return

        output_dir = self.tracking_system.visualization_path.parent

        # Windows系统打开文件夹
        if sys.platform == 'win32':
            os.startfile(str(output_dir))
        else:
            # 其他系统（如Mac、Linux）的处理
            import subprocess
            try:
                if sys.platform == 'darwin':
                    subprocess.Popen(['open', str(output_dir)])
                else:
                    subprocess.Popen(['xdg-open', str(output_dir)])
            except Exception as e:
                QMessageBox.warning(self, "错误", f"无法打开文件夹: {str(e)}")

    def selectSequence(self):
        """选择要处理的序列"""
        # 获取配置中的源目录
        source_dir = self.tracking_system.config['source']
        dialog = SequenceSelectorDialog(source_dir, self)
        if dialog.exec_() == QDialog.Accepted:
            selected_seq = dialog.getSelectedSequence()
            if selected_seq:
                # 构建完整的序列路径
                seq_path = Path(source_dir) / selected_seq
                self.sequence_path_label.setText(f"已选择序列: {seq_path}")
                self.process_button.setEnabled(True)
                self.current_seq_path = str(seq_path)  # 存储完整的序列路径

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
