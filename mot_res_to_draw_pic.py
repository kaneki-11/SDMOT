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
        text_scale=0.6  # 新增：ID文字大小缩放系数（默认0.6，越大文字越大）
):
    """
    在指定图片上绘制对应帧的多目标跟踪框和ID（优化版）
    :param track_result_path: 跟踪结果文件路径（txt格式）
    :param img_path: 图片路径
    :param img_width: 图片宽度
    :param img_height: 图片高度
    :param target_frame: 目标帧号
    :param output_path: 绘制后图片的保存路径
    :param box_thickness: 跟踪框粗细（默认2）
    :param text_thickness: ID文字粗细（默认1）
    :param text_scale: ID文字大小缩放系数（默认0.6）
    :return: 绘制后的图片对象
    """
    # 读取图片
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"图片文件不存在：{img_path}")
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图片：{img_path}")

    # 读取跟踪结果并解析目标帧数据
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

    # 绘制跟踪框和ID（颜色随机且ID与框同色）
    for track_id, x, y, w, h, conf in track_data:
        # 基于ID生成固定随机颜色（同一ID始终同一颜色）
        np.random.seed(track_id)
        color = tuple(np.random.randint(0, 256, 3).tolist())  # BGR格式
        np.random.seed(None)  # 重置随机种子

        # 绘制跟踪框
        cv2.rectangle(img, (x, y), (x + w, y + h), color, box_thickness)
        # 绘制ID背景（半透明黑色）
        id_text = f"ID:{track_id} (%.2f)" % conf
        # 基于文字缩放系数计算背景大小
        text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)[0]
        text_bg_x1 = x
        text_bg_y1 = y - 20 if y - 20 > 0 else y  # 避免背景超出图片顶部
        text_bg_x2 = x + text_size[0] + 6
        text_bg_y2 = y
        cv2.rectangle(img, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), (0, 0, 0), -1)
        # 绘制ID文字（与框同色，支持大小缩放）
        cv2.putText(
            img, id_text, (x + 3, y - 5 if y - 5 > 0 else y + 15),  # 文字位置适配
            cv2.FONT_HERSHEY_SIMPLEX, text_scale,
            color, text_thickness
        )

    # 确保保存目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # 保存图片
    cv2.imwrite(output_path, img)
    print(f"✅ 绘制完成！图片已保存至：{output_path}")
    print(f"📊 帧{target_frame}共检测到{len(track_data)}个跟踪目标")
    return img


if __name__ == "__main__":
    # -------------------------- 配置参数（直接修改这里即可）--------------------------
    TRACK_RESULT_PATH = r"D:\boxmot\runs\mot\dancetrack0017--dance_yolo11s_best_osnet_x0_25_msmt17_bytetrack\dancetrack0017.txt"  # 跟踪结果文件路径
    IMG_PATH = r"D:\研究生\研究生\硕士毕业论文\实验\预答辩图\原图\dancetrack0017--00000114.jpg"  # 原始图片路径（示例：88帧对应的图片）
    IMG_WIDTH = 1920  # 图片实际宽度
    IMG_HEIGHT = 1080  # 图片实际高度
    TARGET_FRAME = 114  # 要查找的帧号（示例：88帧）
    OUTPUT_PATH = r"D:\研究生\研究生\硕士毕业论文\实验\预答辩图\加第二章改进前\dancetrack0017--00000114.jpg"  # 指定保存路径（支持多级目录）
    BOX_THICKNESS = 3  # 跟踪框粗细（可调整）
    TEXT_THICKNESS = 2  # ID文字粗细（可调整）
    TEXT_SCALE = 0.8  # ID文字大小（可调整，默认0.6，建议范围0.4-1.2）
    # -----------------------------------------------------------------------------

    try:
        # 执行绘制
        result_img = draw_tracking_boxes(
            track_result_path=TRACK_RESULT_PATH,
            img_path=IMG_PATH,
            img_width=IMG_WIDTH,
            img_height=IMG_HEIGHT,
            target_frame=TARGET_FRAME,
            output_path=OUTPUT_PATH,
            box_thickness=BOX_THICKNESS,
            text_thickness=TEXT_THICKNESS,
            text_scale=TEXT_SCALE  # 传递文字大小参数
        )

        # 可选：显示图片
        cv2.imshow(f"Tracking Frame {TARGET_FRAME}", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"❌ 执行出错：{str(e)}")