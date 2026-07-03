import cv2
import os
import argparse
from pathlib import Path


def video_to_frames(video_path, output_dir, prefix='frame_', start_index=1, format='jpg', quality=95):
    """
    将视频转换为图像帧

    参数:
    video_path (str): 输入视频文件路径
    output_dir (str): 输出图像帧的目录
    prefix (str): 文件名前缀 (默认为 'frame_')
    start_index (int): 起始编号 (默认为1)
    format (str): 图像格式 (默认为 'jpg')
    quality (int): 图像质量 (1-100, 仅对jpg有效)
    """
    # 检查视频文件是否存在
    if not os.path.isfile(video_path):
        print(f"错误: 视频文件 '{video_path}' 不存在")
        return

    # 创建输出目录（如果不存在）
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 '{video_path}'")
        return

    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"视频信息: {os.path.basename(video_path)}")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps:.2f} FPS")
    print(f"  总帧数: {total_frames}")
    print(f"输出目录: {output_dir}")
    print(f"格式: {format}, 质量: {quality}")
    print("开始转换...")

    # 处理参数设置
    count = start_index
    success = True

    # 进度计数器
    processed_frames = 0

    while success:
        # 读取一帧
        success, frame = cap.read()

        if not success:
            break

        # 生成文件名 (6位数字编号)
        filename = f"{prefix}{count:06d}.{format}"
        output_path = os.path.join(output_dir, filename)

        # 保存图像
        if format.lower() in ['jpg', 'jpeg']:
            cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        else:
            cv2.imwrite(output_path, frame)

        # 更新计数器
        count += 1
        processed_frames += 1

        # 每处理100帧打印一次进度
        if processed_frames % 100 == 0:
            print(f"已处理 {processed_frames}/{total_frames} 帧 ({processed_frames / total_frames * 100:.1f}%)")

    # 释放视频资源
    cap.release()

    print(f"\n转换完成! 共生成 {processed_frames} 张图像")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='将视频转换为图像帧序列')
    parser.add_argument('--video', type=str, required=True, help='输入视频文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出图像目录路径')
    parser.add_argument('--prefix', type=str, default='', help='文件名前缀 (默认为 )')
    parser.add_argument('--start', type=int, default=1, help='起始编号 (默认为1)')
    parser.add_argument('--format', type=str, default='jpg', choices=['jpg', 'png', 'bmp'], help='图像格式 (默认为jpg)')
    parser.add_argument('--quality', type=int, default=95, help='JPEG图像质量 (1-100, 默认为95)')

    args = parser.parse_args()

    # 运行转换
    video_to_frames(
        video_path=args.video,
        output_dir=args.output,
        prefix=args.prefix,
        start_index=args.start,
        format=args.format,
        quality=args.quality
    )
