import cv2
import numpy as np
import os


def draw_tracking_boxes(
        track_result_path,
        img_path,
        img_width,
        img_height,
        target_frame,
        output_path,
        box_thickness=2,
        text_thickness=1,
        text_scale=0.6
):
    """
    多目标跟踪框绘制（批量版核心函数）
    特性：透明ID背景、同ID跨帧颜色一致、支持中文路径、参数可自定义
    """
    # 读取图片（支持中文路径）
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"图片文件不存在：{img_path}")
    try:
        with open(img_path, 'rb') as f:
            img_data = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法解码图片：{img_path}（文件损坏或格式不支持）")
    except Exception as e:
        raise ValueError(f"读取图片失败：{str(e)}")

    # 解析跟踪结果
    track_data = []
    with open(track_result_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [x for x in line.split(',') if x]
            if len(parts) < 9:
                continue
            try:
                frame = int(parts[0])
                track_id = int(parts[1])
                x = int(float(parts[2]))
                y = int(float(parts[3]))
                w = int(float(parts[4]))
                h = int(float(parts[5]))
                confidence = float(parts[8])
            except (ValueError, IndexError):
                continue
            if frame == target_frame and 0 <= x < img_width and 0 <= y < img_height:
                track_data.append((track_id, x, y, w, h, confidence))

    # 绘制（核心优化：透明背景+同ID同色）
    for track_id, x, y, w, h, conf in track_data:
        # 基于ID固定随机色（跨帧/跨图片都一致）
        np.random.seed(track_id)
        color = tuple(np.random.randint(0, 256, 3).tolist())  # BGR格式
        np.random.seed(None)

        # 绘制跟踪框
        cv2.rectangle(img, (x, y), (x + w, y + h), color, box_thickness)
        # 绘制ID文字（无背景，透明显示）
        id_text = f"ID:{track_id} (%.2f)" % conf
        text_pos_y = y - 5 if y - 5 > 0 else y + 15  # 避免超出图片
        cv2.putText(
            img, id_text, (x + 3, text_pos_y),
            cv2.FONT_HERSHEY_SIMPLEX, text_scale,
            color, text_thickness
        )

    # 保存图片（支持中文路径）
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    try:
        ext = os.path.splitext(output_path)[1]
        result, img_encoded = cv2.imencode(ext, img)
        if result:
            with open(output_path, 'wb') as f:
                img_encoded.tofile(f)
            print(f"✅ 帧{target_frame}绘制完成！保存至：{output_path}")
        else:
            raise ValueError("图片编码失败")
    except Exception as e:
        raise ValueError(f"保存图片失败：{str(e)}")

    print(f"📊 检测到{len(track_data)}个跟踪目标\n")
    return img


if __name__ == "__main__":
    # -------------------------- 批量配置参数（修改这里即可）--------------------------
    TRACK_RESULT_PATH = r"D:\boxmot\runs\mot\v_aAb0psypDj4_c008--spo_yolo11s_best_osnet_x0_25_msmt17_bytetrack\v_aAb0psypDj4_c008.txt"  # 跟踪结果文件路径（统一）
    IMG_WIDTH = 1920  # 图片实际宽度（统一）
    IMG_HEIGHT = 1080  # 图片实际高度（统一）
    OUTPUT_FOLDER = r"D:\研究生\研究生\硕士毕业论文\实验\预答辩图\加第二章改进后"  # 批量输出文件夹路径
    BOX_THICKNESS = 3  # 框粗细（统一调整）
    TEXT_THICKNESS = 2  # 文字粗细（统一调整）
    TEXT_SCALE = 0.6  # 文字大小（统一调整）

    # 3张图片配置：[(图片路径, 对应帧号), ...]
    BATCH_CONFIG = [
        (r"D:\研究生\研究生\硕士毕业论文\实验\预答辩图\原图\v_aAb0psypDj4_c008--000108.jpg", 108),
        (r"D:\研究生\研究生\硕士毕业论文\实验\预答辩图\原图\v_aAb0psypDj4_c008--000125.jpg", 125),
        (r"D:\研究生\研究生\硕士毕业论文\实验\预答辩图\原图\v_aAb0psypDj4_c008--000142.jpg", 142)
    ]
    # -----------------------------------------------------------------------------

    # 确保输出文件夹存在
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        print(f"📁 已创建输出文件夹：{OUTPUT_FOLDER}")

    # 批量处理3张图片
    for idx, (img_path, target_frame) in enumerate(BATCH_CONFIG, 1):
        try:
            # 生成输出路径（保留原文件名）
            img_filename = os.path.basename(img_path)
            output_path = os.path.join(OUTPUT_FOLDER, img_filename)

            # 执行绘制
            draw_tracking_boxes(
                track_result_path=TRACK_RESULT_PATH,
                img_path=img_path,
                img_width=IMG_WIDTH,
                img_height=IMG_HEIGHT,
                target_frame=target_frame,
                output_path=output_path,
                box_thickness=BOX_THICKNESS,
                text_thickness=TEXT_THICKNESS,
                text_scale=TEXT_SCALE
            )
        except Exception as e:
            print(f"❌ 第{idx}张图片处理失败：{str(e)}\n")
            continue

    print("🎉 批量处理完成！所有成功绘制的图片已保存至：", OUTPUT_FOLDER)