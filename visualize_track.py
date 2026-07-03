import os
import glob
import cv2
import argparse
import numpy as np
from collections import defaultdict, deque
import sys
import platform
import webbrowser

def parse_args():
    p = argparse.ArgumentParser(
        description="可视化 MOT 跟踪结果，支持视频或逐帧输出，带点状大小渐变尾迹"
    )
    p.add_argument("--seq-dir", required=True,
                   help="序列目录，包含按帧命名的图片（如 img1/000001.jpg）")
    p.add_argument("--mot-txt", required=True,
                   help="跟踪结果文件（MOT 格式 .txt）")
    p.add_argument("--out-video", default=None,
                   help="输出视频文件路径，如 output.mp4；不需要则不传")
    p.add_argument("--out-dir", default=None,
                   help="输出帧图片目录；不需要则不传")
    p.add_argument("--trail", type=int, default=20,
                   help="轨迹尾迹长度（帧数），视频模式下生效")
    p.add_argument("--fps", type=float, default=30.0,
                   help="输出视频的帧率")
    return p.parse_args()


def load_mot_results(txt_path):
    """
    读取 MOT 结果文件，返回 ndarray，列顺序至少：
    [frame, id, x, y, w, h, ...]
    """
    # 检测文件分隔符
    with open(txt_path, 'r') as f:
        first_line = f.readline()
        delimiter = ',' if ',' in first_line else ' '

    data = np.loadtxt(txt_path, delimiter=delimiter)
    if data.size == 0:
        return np.zeros((0, 6), dtype=int)
    if data.ndim == 1:
        data = data[None, :]
    return data.astype(int)


def main():
    args = parse_args()

    # 1. 准备输入帧列表
    img_paths = sorted(glob.glob(os.path.join(args.seq_dir, "*.jpg")))
    if not img_paths:
        img_paths = sorted(glob.glob(os.path.join(args.seq_dir, "*.png")))
    assert img_paths, f"在 {args.seq_dir} 中未找到 jpg/png 图片"

    # 2. 读取跟踪结果
    mot = load_mot_results(args.mot_txt)
    mot_by_frame = {}
    for row in mot:
        frm = row[0]
        mot_by_frame.setdefault(frm, []).append(row)
    for k in mot_by_frame:
        mot_by_frame[k] = np.stack(mot_by_frame[k], axis=0)

    # 3. 初始化尾迹历史：id -> deque of center points
    trails = defaultdict(lambda: deque(maxlen=args.trail))

    # 4. 准备输出
    writer = None
    if args.out_video:
        h, w = cv2.imread(img_paths[0]).shape[:2]
        # Windows兼容的视频编解码器
        if platform.system() == 'Windows':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4V在Windows上兼容性更好
            # fourcc = cv2.VideoWriter_fourcc(*'H264')  # MP4V在Windows上兼容性更好
        else:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # macOS/Linux使用avc1
        writer = cv2.VideoWriter(args.out_video, fourcc, args.fps, (w, h))
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    # 5. 随机为每个 ID 分配固定颜色
    np.random.seed(42)
    color_map = {}

    # 最大点半径（像素）
    MAX_RADIUS = 5

    # 6. 遍历帧
    for idx, img_path in enumerate(img_paths, start=1):
        im = cv2.imread(img_path)
        if im is None:
            print(f"警告: 无法读取图像 {img_path}")
            continue

        frame_no = idx

        dets = mot_by_frame.get(frame_no, np.zeros((0, 6), dtype=int))

        # 绘制检测框与 ID，并记录 & 绘制点状尾迹
        for det in dets:
            if len(det) < 6:
                continue

            _, track_id, x, y, w, h = det[:6]
            x, y, w, h = int(x), int(y), int(w), int(h)
            if track_id not in color_map:
                color_map[track_id] = tuple(map(int, np.random.randint(0, 255, 3)))
            c = color_map[track_id]

            # 绘框 & ID
            cv2.rectangle(im, (x, y), (x + w, y + h), c, 2)
            cv2.putText(im, f"ID-{track_id}", (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

            # 更新尾迹中心点
            cx, cy = x + w // 2, y + h // 2
            trails[track_id].append((cx, cy))

            # 点状尾迹：越新的点越大
            pts = list(trails[track_id])
            L = len(pts)
            for i, (px, py) in enumerate(pts):
                # i=0 最旧 -> 最小； i=L-1 最新 -> 最大
                radius = max(1, int(MAX_RADIUS * (i + 1) / L))
                cv2.circle(im, (px, py), radius, c, -1)

        # 写视频或保存帧
        if writer:
            writer.write(im)
        if args.out_dir:
            basename = os.path.basename(img_path)
            cv2.imwrite(os.path.join(args.out_dir, basename), im)

        if not writer and args.out_dir:
            print(f"Saved frame {frame_no} -> {os.path.join(args.out_dir, basename)}")

    # 7. 结束
    if writer:
        writer.release()
        print(f"[完成] 视频已保存到 {args.out_video}")
    if args.out_dir:
        print(f"[完成] 图片帧已保存到 {args.out_dir}")


if __name__ == "__main__":
    main()
