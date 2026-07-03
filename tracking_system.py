import os
import subprocess
import re
import json
from pathlib import Path
import sys
import platform


class TrackingSystem:
    def __init__(self):
        base_dir = Path(r"D:\boxmot")

        self.config = {
            'yolo_model': str(base_dir / 'tracking' / 'weights' / 'spo_yolo11s_best.pt'),
            'reid_model': str(base_dir / 'tracking' / 'weights' / 'osnet_x0_25_msmt17.pt'),
            'tracking_method': 'bytetrack',
            # 修改为序列目录的父目录
            'source': str(base_dir / 'assets' / 'MOT17-mini' / 'test'),
            'project': str(base_dir / 'runs'),
            'verbose': True,
            'classes': 0  # 只检测行人
        }


        # 确保输出目录存在
        os.makedirs(self.config['project'], exist_ok=True)

        # 处理状态
        self.current_stage = "idle"
        self.current_seq = None
        self.progress = 0

        # 结果数据
        self.detections = {}
        self.tracks = {}
        self.det_file_path = ""
        self.track_file_path = ""
        self.visualization_path = ""

    def run_full_pipeline(self, seq_path):
        """运行完整的检测和跟踪流程"""
        # 验证序列路径
        seq_path = Path(seq_path)
        if not seq_path.exists():
            return False, f"序列路径不存在: {seq_path}"

        # 验证权重文件路径
        yolo_path = Path(self.config['yolo_model'])
        if not yolo_path.exists():
            return False, f"YOLO模型文件不存在: {yolo_path}"

        reid_path = Path(self.config['reid_model'])
        if not reid_path.exists():
            return False, f"ReID模型文件不存在: {reid_path}"

        self.current_seq = seq_path.name
        self.progress = 0
        self.current_stage = "detection"

        # 确保项目路径存在
        project_path = Path(self.config['project'])
        project_path.mkdir(parents=True, exist_ok=True)

        print(f"处理序列: {self.current_seq}")
        print(f"YOLO模型: {yolo_path}")
        print(f"ReID模型: {reid_path}")
        print(f"项目路径: {project_path}")

        # 打印详细的路径信息
        print(f"序列路径: {seq_path}")
        print(f"序列路径是否存在: {seq_path.exists()}")
        print(f"序列目录内容: {list(seq_path.glob('*'))}")

        # 构建命令 - 使用序列目录作为源
        # cmd = [
        #     sys.executable,
        #     str(Path(__file__).parent / "val.py"),
        #     "--yolo-model", self.config['yolo_model'],
        #     "--reid-model", self.config['reid_model'],
        #     "--tracking-method", self.config['tracking_method'],
        #     # 传递序列目录 (如 v_iF9bKPWdZlc_c001)
        #     "--source", str(seq_path),
        #     "--project", str(project_path),
        #     "--exist-ok",
        #     "--verbose"
        # ]
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "val.py"),
            "--yolo-model", self.config['yolo_model'],
            "--reid-model", self.config['reid_model'],
            "--tracking-method", self.config['tracking_method'],
            "--source", str(seq_path),
            "--project", str(project_path),
            "--exist-ok",
            "--verbose",
            "--classes", "0",
            # 强制使用CPU
            "--device", "cpu"
        ]

        # 只检测行人
        if self.config.get('classes', 0) == 0:
            cmd.extend(["--classes", "0"])

        # Windows下需要设置PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent)
        env["PYTHONIOENCODING"] = "utf-8"  # 处理中文路径

        print("执行命令:", " ".join(cmd))

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, env=env, encoding='utf-8', check=True)
            print("val.py stdout:", result.stdout)
            print("val.py stderr:", result.stderr)
        except subprocess.CalledProcessError as e:
            error_msg = f"处理失败: {e.stderr}\n命令: {e.cmd}\n返回码: {e.returncode}"
            print(error_msg)
            return False, error_msg

        # 更新跟踪文件路径
        self.track_file_path = (
                project_path / 'mot' /
                f"{Path(self.config['yolo_model']).stem}_{Path(self.config['reid_model']).stem}_{self.config['tracking_method']}" /
                f"{self.current_seq}.txt"
        )

        print(f"跟踪文件路径: {self.track_file_path}")
        if not self.track_file_path.exists():
            print("警告：跟踪文件未生成")

        # 添加检测文件路径
        self.det_file_path = (
                project_path / 'dets_n_embs' / yolo_path.stem / 'dets' / f"{self.current_seq}.txt"
        )

        self.progress = 100
        self.current_stage = "completed"
        return True, "处理完成"

    def run_visualization(self, trail=30, fps=25):
        """运行可视化，生成带尾迹的视频"""
        if not self.track_file_path or not self.track_file_path.exists():
            return False, "尚未生成跟踪结果"

        # 确保序列目录存在
        seq_dir = Path(self.config['source']) / self.current_seq / 'img1'
        if not seq_dir.exists():
            return False, f"序列目录不存在: {seq_dir}"

        # 输出视频路径
        out_video_dir = Path(self.config['project']) / 'track_visual'
        os.makedirs(out_video_dir, exist_ok=True)
        out_video_path = out_video_dir / f"{self.current_seq}_vis.mp4"

        # 调用visualize_track.py
        # vis_cmd = [
        #     sys.executable,
        #     str(Path(__file__).parent / "visualize_track.py"),
        #     "--seq-dir", str(seq_dir),
        #     "--mot-txt", str(self.track_file_path),
        #     "--out-video", str(out_video_path),
        #     "--trail", str(trail),
        #     "--fps", str(fps)
        # ]

        # 新增逐帧输出目录
        out_frames_dir = Path(self.config['project']) / 'track_visual_frames' / self.current_seq
        os.makedirs(out_frames_dir, exist_ok=True)  # 确保目录存在

        vis_cmd = [
            sys.executable,
            str(Path(__file__).parent / "visualize_track.py"),
            "--seq-dir", str(seq_dir),
            "--mot-txt", str(self.track_file_path),
            "--out-video", str(out_video_path),  # 保留视频输出（可选，可删除）
            "--out-dir", str(out_frames_dir),  # 新增：逐帧输出目录
            "--trail", str(trail),
            "--fps", str(fps)
        ]


        result = subprocess.run(vis_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return False, result.stderr

        self.visualization_path = out_video_path
        return True, str(out_video_path)

    def get_sequence_list(self):
        """获取序列列表"""
        source_path = Path(self.config['source'])
        if not source_path.exists():
            return []

        sequences = []
        for item in source_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                img_dir = item / 'img1'
                if img_dir.exists():
                    # 计算帧数
                    frame_count = len(list(img_dir.glob('*.jpg'))) + len(list(img_dir.glob('*.png')))
                    if frame_count > 0:
                        sequences.append({
                            'name': item.name,
                            'path': str(item),
                            'frame_count': frame_count
                        })
        return sequences

    # def load_detections(self):
    #     """加载检测文件内容"""
    #     if not self.det_file_path or not os.path.exists(self.det_file_path):
    #         return "尚未生成检测文件"
    #
    #     try:
    #         with open(self.det_file_path, 'r') as f:
    #             return f.read()
    #     except Exception as e:
    #         return f"读取检测文件失败: {str(e)}"
    #
    # def load_tracks(self):
    #     """加载跟踪文件内容"""
    #     if not self.track_file_path or not os.path.exists(self.track_file_path):
    #         return "尚未生成跟踪文件"
    #
    #     try:
    #         with open(self.track_file_path, 'r') as f:
    #             return f.read()
    #     except Exception as e:
    #         return f"读取跟踪文件失败: {str(e)}"

    def load_detections(self):
        with open(self.det_file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def load_tracks(self):
        with open(self.track_file_path, 'r', encoding='utf-8') as f:
            return f.read()
